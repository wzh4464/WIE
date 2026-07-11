#!/usr/bin/env python3
"""
LOO Epoch-wise Valuation Script

Computes Leave-One-Out (LOO) epoch-wise valuation similar to wie_all_epochs.
For each epoch and each training sample, evaluates the performance difference
when that sample is removed during training.

Usage:
    python scripts/loo_epoch_wise_valuation.py --save_dir loo_similar --seed 42 --relabel 0
"""

import argparse
import os
import sys
import numpy as np
import torch
import logging
from pathlib import Path
from typing import List
import gc

# Add the project root to the path to import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Make the src/ layout importable when this script is run by path
# (python scripts/foo.py) without the editable install (pixi run install).
try:
    import wie  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wie.training.dataset_config import DATASET_NETWORK_CONFIG
from wie.data.modules import fetch_data_module
from wie.models.networks import get_network
from wie.utils.paths import resolve_output_dir


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def load_data_and_config(target: str, model: str, seed: int, relabel_percentage: int):
    """Load dataset and configuration."""
    config = DATASET_NETWORK_CONFIG[(target, model)]

    # Initialize data module using the registry
    data_module = fetch_data_module(target, seed=seed)

    # Get data sizes from config
    n_tr = config["n_tr"]
    n_val = config["n_val"]
    n_test = config.get("n_test", 1000)

    # Fetch data
    (x_tr, y_tr), (x_val, y_val), (x_test, y_test) = data_module.fetch(
        n_tr, n_val, n_test, seed=seed
    )

    # Convert to tensors
    x_tr = torch.tensor(x_tr, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    return x_tr, y_tr, x_val, y_val, config, n_tr


def load_counterfactual_models(save_dir: str, seed: int, relabel_percentage: int):
    """Load the LOO counterfactual models."""
    base_dir = str(resolve_output_dir(save_dir))
    models_path = os.path.join(
        base_dir,
        "records",
        f"relabel_{relabel_percentage:03d}_pct_counterfactual_models_{seed:03d}.pt",
    )

    if not os.path.exists(models_path):
        raise FileNotFoundError(f"Counterfactual models not found at: {models_path}")

    logger = logging.getLogger(__name__)
    logger.info(f"Loading counterfactual models from: {models_path}")

    # Load with weights_only=False to handle NetList objects
    counterfactual_models = torch.load(
        models_path, map_location="cpu", weights_only=False
    )

    logger.info(f"Loaded {len(counterfactual_models)} counterfactual model sets")
    logger.info(f"Each set contains {len(counterfactual_models[0])} epoch models")

    return counterfactual_models


def compute_model_performance(
    model: torch.nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    loss_fn,
    device: str,
) -> float:
    """Compute validation loss for a given model."""
    model.eval()
    model.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    with torch.no_grad():
        outputs = model(x_val)
        loss = loss_fn(outputs, y_val)
        return loss.item()


def compute_loo_epoch_wise_valuation(
    counterfactual_models: List,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    baseline_model: torch.nn.Module,
    loss_fn,
    device: str,
    logger,
) -> List[np.ndarray]:
    """
    Compute LOO epoch-wise valuation similar to wie_all_epochs.

    For each epoch, we compare the final baseline model with counterfactual models
    trained without each sample for that many epochs.

    Args:
        counterfactual_models: List of NetList objects, one per training sample
        x_val, y_val: Validation data
        baseline_model: Final model trained on full dataset
        loss_fn: Loss function
        device: Device to use for computation
        logger: Logger instance

    Returns:
        List of numpy arrays, one per epoch, containing valuation for each training sample
    """
    n_tr = len(counterfactual_models)
    n_epochs = len(counterfactual_models[0])

    logger.info(
        f"Computing LOO epoch-wise valuation for {n_tr} samples across {n_epochs} epochs"
    )

    # Compute baseline performance (final model trained on full dataset)
    baseline_loss = compute_model_performance(
        baseline_model, x_val, y_val, loss_fn, device
    )
    logger.info(f"Baseline (full data) final model loss = {baseline_loss:.6f}")

    # Compute LOO valuation for each epoch
    all_epoch_valuations = []

    for epoch_idx in range(n_epochs):
        logger.info(f"Computing LOO valuation for epoch {epoch_idx}")
        epoch_valuations = np.zeros(n_tr, dtype=np.float64)

        for sample_idx in range(n_tr):
            try:
                # Get the counterfactual model for this sample at this epoch
                # (model trained without this sample for epoch_idx+1 epochs)
                counterfactual_model = counterfactual_models[sample_idx].get_model(
                    epoch_idx
                )

                # Compute performance of counterfactual model
                counterfactual_loss = compute_model_performance(
                    counterfactual_model, x_val, y_val, loss_fn, device
                )

                # LOO valuation = baseline_loss - counterfactual_loss
                # Positive value means removing the sample hurts performance
                # Negative value means removing the sample improves performance
                valuation = baseline_loss - counterfactual_loss
                epoch_valuations[sample_idx] = valuation

                if sample_idx % 100 == 0:
                    logger.debug(
                        f"Epoch {epoch_idx}, Sample {sample_idx}: "
                        f"baseline={baseline_loss:.6f}, counterfactual={counterfactual_loss:.6f}, "
                        f"valuation={valuation:.6f}"
                    )

                # Clear GPU memory periodically
                if sample_idx % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.error(
                    f"Error computing valuation for epoch {epoch_idx}, sample {sample_idx}: {e}"
                )
                epoch_valuations[sample_idx] = 0.0
                continue

        all_epoch_valuations.append(epoch_valuations)
        logger.info(
            f"Epoch {epoch_idx}: computed valuations, "
            f"mean={np.mean(epoch_valuations):.6f}, "
            f"std={np.std(epoch_valuations):.6f}"
        )

        # Clear memory after each epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_epoch_valuations


def load_baseline_model(
    save_dir: str,
    target: str,
    model: str,
    seed: int,
    relabel_percentage: int,
    device: str,
    logger,
) -> torch.nn.Module:
    """Load baseline model trained on full dataset (final epoch)."""

    # Load the final epoch model
    base_dir = str(resolve_output_dir(save_dir))
    final_model_path = os.path.join(
        base_dir, "records", f"epoch_final_{seed:03d}.pt"
    )

    if not os.path.exists(final_model_path):
        raise FileNotFoundError(
            f"Final baseline model not found at: {final_model_path}"
        )

    logger.info(f"Loading final baseline model from: {final_model_path}")

    # Load model state dict
    model_state = torch.load(final_model_path, map_location="cpu")

    # Create model instance
    config = DATASET_NETWORK_CONFIG[(target, model)]
    input_dim = config.get("input_dim", (28, 28))
    model_instance = get_network(model, input_dim, logger=logger)
    model_instance.load_state_dict(model_state)
    model_instance.to(device)

    logger.info("Loaded final baseline model")
    return model_instance


def save_results(
    results: List[np.ndarray], save_dir: str, seed: int, relabel_percentage: int, logger
):
    """Save LOO epoch-wise valuation results."""
    output_dir = os.path.join(str(resolve_output_dir(save_dir)), "loo_valuations")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"loo_epoch_wise_valuation_{relabel_percentage:03d}_pct_{seed:03d}.pt",
    )

    torch.save(results, output_file)
    logger.info(f"LOO epoch-wise valuations saved to: {output_file}")

    # Also save as numpy arrays for easier analysis
    numpy_file = os.path.join(
        output_dir,
        f"loo_epoch_wise_valuation_{relabel_percentage:03d}_pct_{seed:03d}.npz",
    )

    np.savez(numpy_file, **{f"epoch_{i}": arr for i, arr in enumerate(results)})
    logger.info(
        f"LOO epoch-wise valuations also saved as numpy arrays to: {numpy_file}"
    )


def main():
    parser = argparse.ArgumentParser(description="LOO Epoch-wise Valuation")
    parser.add_argument("--save_dir", type=str, required=True, help="Save directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--relabel", type=int, default=0, help="Relabel percentage")
    parser.add_argument("--target", type=str, default="mnist", help="Target dataset")
    parser.add_argument("--model", type=str, default="dnn", help="Model type")

    args = parser.parse_args()
    logger = setup_logging()

    logger.info(f"Starting LOO epoch-wise valuation for {args.target}-{args.model}")
    logger.info(f"Parameters: seed={args.seed}, relabel={args.relabel}")

    # Setup device
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    try:
        # Load data and configuration
        x_tr, y_tr, x_val, y_val, config, n_tr = load_data_and_config(
            args.target, args.model, args.seed, args.relabel
        )
        logger.info(f"Loaded data: n_tr={n_tr}, n_val={x_val.shape[0]}")

        # Load counterfactual models
        counterfactual_models = load_counterfactual_models(
            args.save_dir, args.seed, args.relabel
        )

        # Load baseline model
        baseline_model = load_baseline_model(
            args.save_dir,
            args.target,
            args.model,
            args.seed,
            args.relabel,
            device,
            logger,
        )

        # Setup loss function
        if args.target in ["mnist", "cifar10", "emnist"]:
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()

        # Compute LOO epoch-wise valuation
        valuations = compute_loo_epoch_wise_valuation(
            counterfactual_models, x_val, y_val, baseline_model, loss_fn, device, logger
        )

        # Save results
        save_results(valuations, args.save_dir, args.seed, args.relabel, logger)

        # Print summary statistics
        logger.info("\nLOO Epoch-wise Valuation Summary:")
        logger.info("=" * 50)
        for epoch_idx, epoch_vals in enumerate(valuations):
            logger.info(
                f"Epoch {epoch_idx:2d}: mean={np.mean(epoch_vals):8.6f}, "
                f"std={np.std(epoch_vals):8.6f}, "
                f"min={np.min(epoch_vals):8.6f}, "
                f"max={np.max(epoch_vals):8.6f}"
            )

        logger.info("LOO epoch-wise valuation computation completed successfully!")

    except Exception as e:
        logger.error(
            f"Error in LOO epoch-wise valuation computation: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
