#!/usr/bin/env python3
"""
LOO Similarity Analysis Script

Computes influence correlations with Leave-One-Out (LOO) across different time windows
and methods for MNIST-DNN configuration. Generates a CSV with correlation results.

Usage:
    python scripts/loo_similarity.py --save_dir results --seed 42 --relabel 0
"""

import argparse
import os
import sys
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import logging
from pathlib import Path
import torch

# Add the project root to the path to import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Make the src/ layout importable when this script is run by path
# (python scripts/foo.py) without the editable install (pixi run install).
try:
    import wie  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wie.infl import InfluenceCalculatorFactory
from wie.training.dataset_config import DATASET_NETWORK_CONFIG
from wie.utils.paths import resolve_output_dir


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def check_training_results(
    save_dir: str, target: str, model: str, seed: int, relabel_percentage: int
):
    """Check if training results exist in the specified directory."""
    # Check for outputs directory structure using centralized path resolution
    outputs_path = str(resolve_output_dir(save_dir))
    if os.path.exists(outputs_path):
        global_info_path = os.path.join(outputs_path, f"global_info_{seed:03d}.json")
        return os.path.exists(global_info_path)

    # Check for old structure
    results_path = os.path.join(
        save_dir,
        target,
        model,
        f"seed_{seed}_relabel_{relabel_percentage}",
        "results.pt",
    )
    return os.path.exists(results_path)


def get_time_windows(num_epochs: int):
    """Define early, middle, and late time windows."""
    # Define time points
    t1 = num_epochs // 3  # Early period end
    t2 = 2 * num_epochs // 3  # Middle period end
    t3 = num_epochs  # Late period end (full training)

    return {"early": (0, t1), "middle": (t1, t2), "late": (t2, t3), "full": (0, t3)}


def compute_influence_for_method(
    method_name: str,
    infl_type: str,
    save_dir: str,
    target: str,
    model: str,
    seed: int,
    relabel_percentage: int,
    window_start: int = None,
    window_end: int = None,
):
    """Compute influence using specified method and optional time window."""

    # Get dataset config
    config = DATASET_NETWORK_CONFIG[(target, model)]

    # Setup calculator kwargs with required key and gpu parameters
    calc_kwargs = {
        "key": f"{target}_{model}_{seed}_{relabel_percentage}",  # Create a unique key
        "gpu": 0,  # GPU index
        "target": target,
        "model_type": model,
        "seed": seed,
        "save_dir": save_dir,  # resolved under outputs/ by the calculator's make_base_dir
        "relabel_percentage": relabel_percentage,
        "n_tr": config["n_tr"],
        "n_val": config["n_val"],
        "batch_size": config["batch_size"],
        "device": "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # For WIE All Epochs, we don't need window parameters as it computes all epochs

    # Create calculator and compute influence
    try:
        calculator = InfluenceCalculatorFactory.create(infl_type, **calc_kwargs)
        influence_scores = calculator.calculate()
        return influence_scores
    except Exception as e:
        raise RuntimeError(f"Failed to compute {method_name} influence: {str(e)}")


def compute_correlation(scores1, scores2, method="pearson"):
    """Compute correlation between two sets of influence scores."""
    if method == "pearson":
        corr, p_value = pearsonr(scores1, scores2)
    elif method == "spearman":
        corr, p_value = spearmanr(scores1, scores2)
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    return corr, p_value


def main():
    parser = argparse.ArgumentParser(description="LOO Similarity Analysis")
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Base save directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--relabel", type=int, default=0, help="Relabel percentage")
    parser.add_argument("--target", type=str, default="mnist", help="Target dataset")
    parser.add_argument("--model", type=str, default="dnn", help="Model type")
    parser.add_argument(
        "--correlation_method",
        type=str,
        default="pearson",
        choices=["pearson", "spearman"],
        help="Correlation method",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="loo_similarity_results.csv",
        help="Output CSV file name",
    )

    args = parser.parse_args()
    logger = setup_logging()

    logger.info(f"Starting LOO similarity analysis for {args.target}-{args.model}")
    logger.info(f"Parameters: seed={args.seed}, relabel={args.relabel}")

    # Get configuration
    config = DATASET_NETWORK_CONFIG[(args.target, args.model)]
    num_epochs = config["num_epoch"]

    # Define time windows
    windows = get_time_windows(num_epochs)
    logger.info(f"Time windows: {windows}")

    # Check if training results exist
    if not check_training_results(
        args.save_dir, args.target, args.model, args.seed, args.relabel
    ):
        logger.error("Training results not found. Please run training first.")
        return

    # Methods to evaluate (using available registered types)
    methods = {
        "TIM_All_Epochs": "wie_all_epochs",
        "LAVA": "lava",
        "TD_Influence": "td_influence",
        "SGD": "sgd",
        "TRACIN": "tracin",
    }

    # Results storage
    results = []
    all_scores = {}

    # Compute influence scores for all methods and windows
    for method_name, infl_type in methods.items():
        logger.info(f"Processing method: {method_name}")
        all_scores[method_name] = {}

        for window_name, (start, end) in windows.items():
            try:
                # Compute influence scores for current method
                logger.info(
                    f"Computing {method_name} scores for {window_name} window [{start}, {end}]"
                )
                scores = compute_influence_for_method(
                    method_name,
                    infl_type,
                    args.save_dir,
                    args.target,
                    args.model,
                    args.seed,
                    args.relabel,
                    start,
                    end,
                )
                all_scores[method_name][window_name] = scores
                logger.info(
                    f"{method_name} {window_name}: computed {len(scores)} influence scores"
                )

            except Exception as e:
                logger.error(f"Failed to compute {method_name} for {window_name}: {e}")
                all_scores[method_name][window_name] = None

    # Now compute correlations between methods (using TIM_All_Epochs as reference)
    reference_method = "TIM_All_Epochs"
    if reference_method in all_scores:
        for method_name in methods.keys():
            if method_name == reference_method:
                continue

            logger.info(
                f"Computing correlations for {method_name} vs {reference_method}"
            )

            for window_name, (start, end) in windows.items():
                ref_scores = all_scores[reference_method].get(window_name)
                method_scores = all_scores[method_name].get(window_name)

                if ref_scores is not None and method_scores is not None:
                    try:
                        # Compute correlation with reference method
                        corr, p_value = compute_correlation(
                            ref_scores, method_scores, args.correlation_method
                        )

                        # Compute delta loss notation
                        delta_loss = f"Δl_{end - start}" if start > 0 else f"l_{end}"

                        # Store result
                        result = {
                            "Method": method_name,
                            "Window": f"[{start}, {end}]"
                            if start > 0
                            else f"[0, {end}]",
                            "Window_Name": window_name,
                            "Delta_Loss": delta_loss,
                            f"Correlation_with_{reference_method}": round(corr, 3),
                            "P_Value": round(p_value, 4),
                            "Start_Epoch": start,
                            "End_Epoch": end,
                        }
                        results.append(result)

                        logger.info(
                            f"{method_name} {window_name}: correlation with {reference_method} = {corr:.3f}, p = {p_value:.4f}"
                        )

                    except Exception as e:
                        logger.error(
                            f"Failed to compute correlation for {method_name} {window_name}: {e}"
                        )
                        # Add failed result
                        result = {
                            "Method": method_name,
                            "Window": f"[{start}, {end}]",
                            "Window_Name": window_name,
                            "Delta_Loss": "N/A",
                            f"Correlation_with_{reference_method}": "Failed",
                            "P_Value": "N/A",
                            "Start_Epoch": start,
                            "End_Epoch": end,
                        }
                        results.append(result)
                else:
                    logger.warning(
                        f"Skipping correlation for {method_name} {window_name} (missing scores)"
                    )

        # Add reference method self-correlations (should be 1.0)
        for window_name, (start, end) in windows.items():
            if all_scores[reference_method].get(window_name) is not None:
                delta_loss = f"Δl_{end - start}" if start > 0 else f"l_{end}"
                result = {
                    "Method": reference_method,
                    "Window": f"[{start}, {end}]" if start > 0 else f"[0, {end}]",
                    "Window_Name": window_name,
                    "Delta_Loss": delta_loss,
                    f"Correlation_with_{reference_method}": 1.000,
                    "P_Value": 0.0000,
                    "Start_Epoch": start,
                    "End_Epoch": end,
                }
                results.append(result)

    # Create DataFrame and save
    df = pd.DataFrame(results)

    if len(results) > 0:
        # Sort by method and window for better readability
        method_order = ["TIM_All_Epochs", "LAVA", "TD_Influence", "SGD", "TRACIN"]
        window_order = ["early", "middle", "late", "full"]

        df["Method_Order"] = df["Method"].map(
            {m: i for i, m in enumerate(method_order)}
        )
        df["Window_Order"] = df["Window_Name"].map(
            {w: i for i, w in enumerate(window_order)}
        )
        df = df.sort_values(["Method_Order", "Window_Order"]).drop(
            ["Method_Order", "Window_Order"], axis=1
        )
    else:
        logger.warning("No results computed - creating empty DataFrame")
        df = pd.DataFrame(
            columns=[
                "Method",
                "Window",
                "Window_Name",
                "Delta_Loss",
                "Correlation_with_TIM_All_Epochs",
                "P_Value",
                "Start_Epoch",
                "End_Epoch",
            ]
        )

    # Save results
    output_path = os.path.join(args.save_dir, args.output_file)
    df.to_csv(output_path, index=False)

    logger.info(f"Results saved to: {output_path}")

    # Print summary table
    print("\nInfluence Method Correlation Analysis Results")
    print("=" * 80)

    if len(results) > 0:
        correlation_col = [
            col for col in df.columns if col.startswith("Correlation_with_")
        ][0]
        print(
            df[["Method", "Window", "Delta_Loss", correlation_col]].to_string(
                index=False
            )
        )
    else:
        print("No results computed.")

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
