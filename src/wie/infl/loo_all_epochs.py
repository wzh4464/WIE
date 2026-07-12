import numpy as np
import torch
import gc
import os
from typing import List, Dict, Tuple
from collections import defaultdict

from wie.models.networks import get_network  # type: ignore
from .core import (
    InfluenceCalculator,
    InfluenceCalculatorFactory,
)


@InfluenceCalculatorFactory.register("loo_all_epochs")
class LOOAllEpochsInfluenceCalculator(InfluenceCalculator):
    """
    Computes Leave-One-Out (LOO) valuation for each epoch interval.

    LOO valuation is defined as:
    LOO_valuation(sample_i, epoch_j) = Loss(main_model_epoch_j) - Loss(counterfactual_model_i_epoch_j)

    Where:
    - main_model: Model trained on all training data
    - counterfactual_model_i: Model trained without the i-th training sample
    - Loss: Loss on validation set

    Positive values indicate removing the sample worsens performance (sample is valuable)
    Negative values indicate removing the sample improves performance (sample may be harmful)
    """

    def __init__(self, infl_type: str, **kwargs):
        super().__init__(infl_type, **kwargs)

        # Check if counterfactual models exist
        self.records_dir = os.path.join(self.dn, "records")
        if not os.path.exists(self.records_dir):
            raise FileNotFoundError(
                f"Records directory not found: {self.records_dir}. "
                "Ensure LOO training was completed."
            )

        # Find available model files
        self.main_files, self.cf_files = self._find_model_files()

        if not self.main_files:
            raise FileNotFoundError(
                f"No main model files found in {self.records_dir}. "
                "Ensure training was completed."
            )

        if not self.cf_files:
            raise FileNotFoundError(
                f"No counterfactual model files found in {self.records_dir}. "
                "Ensure LOO training was enabled and completed."
            )

    def _get_infl_type(self) -> str:
        return "loo_all_epochs"

    def _find_model_files(self) -> Tuple[Dict[int, str], Dict[int, Dict[int, str]]]:
        """Find main model and counterfactual model files"""
        seed_suffix = f"{self.seed:03d}"

        # Relabel runs prefix every checkpoint with e.g. "relabel_010_pct_";
        # try the prefixed name first, then the bare name.
        prefixes = [""]
        rp = getattr(self, "relabel_percentage", None)
        if rp:
            prefixes = [f"relabel_{int(rp):03d}_pct_", ""]

        def _first_existing(names):
            for nm in names:
                fp = os.path.join(self.records_dir, nm)
                if os.path.exists(fp):
                    return fp
            return None

        # Main model files
        main_files = {}
        for epoch in range(self.num_epoch):
            fp = _first_existing(
                [f"{p}epoch_{epoch}_{seed_suffix}.pt" for p in prefixes]
            )
            if fp:
                main_files[epoch] = fp

        # Counterfactual model files
        cf_files = defaultdict(dict)
        for epoch in range(self.num_epoch):
            for sample_idx in range(self.n_tr):
                fp = _first_existing(
                    [f"{p}counterfactual_{sample_idx:04d}_epoch_{epoch}_{seed_suffix}.pt"
                     for p in prefixes]
                )
                if fp:
                    cf_files[sample_idx][epoch] = fp

        self.logger.info(f"Found {len(main_files)} main model epoch files")
        self.logger.info(f"Found {len(cf_files)} samples with counterfactual files")

        return main_files, cf_files

    def _load_model_state_dict(self, file_path: str) -> Dict:
        """Load model state dictionary from file"""
        try:
            checkpoint = torch.load(
                file_path, map_location=self.device, weights_only=False
            )

            if "model_state" in checkpoint:
                return checkpoint["model_state"]
            elif isinstance(checkpoint, dict) and any(
                key.startswith(("weight", "bias")) for key in checkpoint.keys()
            ):
                return checkpoint
            else:
                self.logger.warning(f"Unrecognized file format: {file_path}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to load model state from {file_path}: {e}")
            return None

    def _compute_validation_loss(self, model: torch.nn.Module) -> float:
        """Compute validation loss using the provided model"""
        model.eval()
        total_loss = 0.0
        total_samples = 0
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(self.x_val), batch_size):
                batch_x = self.x_val[i : i + batch_size]
                batch_y = self.y_val[i : i + batch_size]

                logits = model(batch_x)
                loss = self.loss_fn(logits, batch_y)

                total_loss += loss.item() * len(batch_x)
                total_samples += len(batch_x)

        return total_loss / total_samples if total_samples > 0 else float("nan")

    def calculate(self) -> List[np.ndarray]:
        """
        Calculate LOO-based influence for each epoch interval.

        Returns:
            List of numpy arrays, one for each epoch, containing the LOO
            valuation scores for that epoch.
        """
        self.logger.info("Starting LOO All Epochs influence calculation...")

        all_epoch_infl: List[np.ndarray] = []

        # Create model template
        model = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )

        # Find samples with counterfactual models
        samples_with_cf = sorted(list(self.cf_files.keys()))
        available_epochs = sorted(list(self.main_files.keys()))

        self.logger.info(
            f"Processing {len(samples_with_cf)} samples with counterfactual models"
        )
        self.logger.info(f"Available epochs: {available_epochs}")

        for epoch_idx in range(self.num_epoch):
            self.logger.info(f"--- Calculating LOO Influence for Epoch {epoch_idx} ---")
            infl_epoch = np.zeros(self.n_tr, dtype=np.float64)

            if epoch_idx not in self.main_files:
                self.logger.warning(f"Epoch {epoch_idx}: No main model file found")
                all_epoch_infl.append(infl_epoch)
                continue

            try:
                # Load main model for this epoch
                main_state_dict = self._load_model_state_dict(
                    self.main_files[epoch_idx]
                )
                if main_state_dict is None:
                    self.logger.warning(f"Epoch {epoch_idx}: Failed to load main model")
                    all_epoch_infl.append(infl_epoch)
                    continue

                model.load_state_dict(main_state_dict)
                main_loss = self._compute_validation_loss(model)

                if np.isnan(main_loss):
                    self.logger.warning(f"Epoch {epoch_idx}: Main model loss is NaN")
                    all_epoch_infl.append(infl_epoch)
                    continue

                # Compute LOO valuation for samples with counterfactual models
                for sample_idx in samples_with_cf:
                    if epoch_idx in self.cf_files[sample_idx]:
                        try:
                            # Load counterfactual model
                            cf_state_dict = self._load_model_state_dict(
                                self.cf_files[sample_idx][epoch_idx]
                            )
                            if cf_state_dict is not None:
                                model.load_state_dict(cf_state_dict)
                                cf_loss = self._compute_validation_loss(model)

                                # LOO valuation = main_loss - counterfactual_loss
                                # Positive values indicate removing the sample worsens performance
                                loo_val = main_loss - cf_loss
                                infl_epoch[sample_idx] = loo_val

                        except Exception as e:
                            self.logger.error(
                                f"Epoch {epoch_idx} sample {sample_idx}: Error computing LOO: {e}"
                            )
                            continue

                self.logger.info(
                    f"Epoch {epoch_idx} LOO scores: "
                    f"mean={infl_epoch.mean():.6f}, "
                    f"std={infl_epoch.std():.6f}, "
                    f"min={infl_epoch.min():.6f}, "
                    f"max={infl_epoch.max():.6f}, "
                    f"non-zero={np.count_nonzero(infl_epoch)}"
                )

                all_epoch_infl.append(infl_epoch)

            except Exception as e:
                self.logger.error(f"Epoch {epoch_idx}: Error: {e}", exc_info=True)
                all_epoch_infl.append(infl_epoch)

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("LOO All Epochs calculation finished.")
        return all_epoch_infl


__all__ = ["LOOAllEpochsInfluenceCalculator"]
