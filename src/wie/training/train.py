import os
import copy
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import torch
import torch.nn as nn
import pandas as pd
import logging
import json
import csv
import random
from typing import Optional

# Assuming NetworkModule, DataModule, config, arg_parser, device_config,
# relabel_utils, logging_utils are in the same directory or accessible
from wie.models.networks import (
    NetList,
    get_network,
    NETWORK_REGISTRY,
)  # Added imports
from wie.data.modules import DATA_MODULE_REGISTRY  # Added imports
from wie.utils.arg_parser import parse_arguments  # Added import
from wie.utils import (  # bridge imports
    get_device,
    get_real_gpu_index,
    handle_relabeling,
    setup_logging,
)
from wie.utils.paths import resolve_output_dir


file_abspath = os.path.abspath(__file__)
# Use project root as the base directory for consistent path handling with
# `pixi run python -m`. This file lives at src/wie/training/train.py, so the
# repository root is four directories up.
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(file_abspath)))
)
current_dir = project_root


class TrainManager:
    def __init__(
        self,
        target=None,
        model=None,
        seed=None,
        save_dir=None,  # Expecting a relative path string like "results"
        csv_path=None,
        n_tr=None,
        n_val=None,
        n_test=None,
        num_epoch=None,
        batch_size=None,
        lr=None,
        decay=None,
        compute_counterfactual=False,
        init_model_path=None,
        save_checkpoints=False,
        save_recording=True,  # Add save_recording parameter
        steps_only=False,  # Add steps_only parameter
        epochs_only=False,  # Add epochs_only parameter
        relabel_csv=None,
        relabel_prefix="",
        relabel_percentage=None,
        device=None,
        logger=None,
        alpha=None,  # Add alpha parameter
        label_smoothing=None,  # Add label smoothing for BCE targets
        # === DVE options (default OFF; safe on weak hardware) ===
        dve_enable: bool = False,
        dve_proj_dim: int = 128,
        dve_proj_type: str = "gaussian",  # or "achlioptas"
        dve_granularity: str = "step",  # "step" or "epoch"
        dve_last_layer_only: bool = True,  # minimal viable DVE
        dve_fp16: bool = True,  # save FP16 for shards
        dve_sample_rate: float = 1.0,  # in epoch-mode, subsample training data (0<r<=1)
    ):
        # Set all random seeds
        self.set_all_seeds(seed)

        self.target = target
        self.model = model
        self.seed = seed
        # Store the relative path argument if needed, but primarily use absolute paths
        self.save_dir_arg = save_dir
        self.csv_path = csv_path
        self.n_tr = n_tr
        self.n_val = n_val
        self.n_test = n_test
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.decay = decay
        self.compute_counterfactual = compute_counterfactual
        self.init_model_path = init_model_path
        self.save_checkpoints = save_checkpoints
        self.save_recording = save_recording  # Store save_recording parameter
        self.steps_only = steps_only  # Store steps_only parameter
        self.epochs_only = epochs_only  # Store epochs_only parameter
        self.relabel_csv = relabel_csv
        self.relabel_prefix = relabel_prefix
        self.relabel_percentage = relabel_percentage
        self.device = device
        self.logger = logger or logging.getLogger(__name__)  # Default logger
        self.current_dir = current_dir  # Directory of this train.py script
        self.alpha = alpha  # Initialize alpha attribute
        # Label smoothing epsilon (0..0.5). Can be None to disable.
        self.label_smoothing = label_smoothing

        # === DVE config ===
        self.dve_enable = bool(dve_enable)
        self.dve_proj_dim = int(dve_proj_dim or 128)
        self.dve_proj_type = str(dve_proj_type or "gaussian")
        self.dve_granularity = (
            dve_granularity if dve_granularity in ("step", "epoch") else "step"
        )
        self.dve_last_layer_only = bool(dve_last_layer_only)
        self.dve_fp16 = bool(dve_fp16)
        self.dve_sample_rate = float(dve_sample_rate or 1.0)
        # runtime DVE buffers
        self._dve_proj_last: Optional[torch.Tensor] = None  # [d, p_last]
        self._dve_saved_last_input: Optional[torch.Tensor] = None  # [B, in_features]
        self._dve_last_linear: Optional[nn.Linear] = None
        self._dve_fwd_handle = None

        # --- Path Management ---
        # Use standardized path resolution to ensure all outputs go to ./outputs/
        if save_dir:
            self.base_save_dir = str(resolve_output_dir(save_dir))
        else:
            # Default save directory under outputs/ if none provided
            default_dir = f"training_results_{target}_{model}_{seed}"
            self.base_save_dir = str(resolve_output_dir(default_dir))
            self.logger.warning(
                f"No save_dir provided, defaulting to {self.base_save_dir}"
            )

        self.records_dir = os.path.join(self.base_save_dir, "records")
        # DVE output dirs
        self.dve_root_dir = os.path.join(self.records_dir, "dve")
        self.dve_raw_dir = os.path.join(self.records_dir, "dve_raw")

        # Create directories early and ensure they exist
        os.makedirs(self.base_save_dir, exist_ok=True)
        os.makedirs(self.records_dir, exist_ok=True)
        if self.dve_enable:
            os.makedirs(self.dve_root_dir, exist_ok=True)
            os.makedirs(self.dve_raw_dir, exist_ok=True)

        self.logger.info(f"Results will be saved in: {self.base_save_dir}")
        if self.save_recording:
            if self.epochs_only:
                self.logger.info(
                    f"Epochs-only recording enabled. Checkpoints will be in: {self.records_dir}"
                )
            elif self.steps_only:
                self.logger.info(
                    f"Steps-only recording enabled. Checkpoints will be in: {self.records_dir}"
                )
            else:
                self.logger.info(
                    f"Recordings (steps and epochs) will be saved in: {self.records_dir}"
                )
        else:
            self.logger.info(
                f"Recording disabled by flag. No checkpoints will be saved to: {self.records_dir}"
            )

        # Validate mutually exclusive recording modes
        if self.steps_only and self.epochs_only:
            self.logger.error(
                "Invalid flags: both --steps-only and --epochs-only are set. Choose one."
            )
            raise ValueError("Choose either steps-only or epochs-only, not both.")

        # Initialize path attributes (will be fully defined in train_and_save)
        self.step_loss_file = None

        # --- Other Attributes ---
        self.x_tr = None
        self.y_tr = None
        self.x_val = None
        self.y_val = None
        self.data_sizes = None
        self.training_params = None
        self.input_dim = None
        self.net_func = None
        self.num_steps = None
        self.list_of_counterfactual_models = None
        self.list_of_sgd_models = []

    def set_all_seeds(self, seed):
        """Set all possible random seeds to ensure full reproducibility"""
        # Python random number generator
        random.seed(seed)
        # NumPy random number generator
        np.random.seed(seed)
        # PyTorch random number generator
        torch.manual_seed(seed)
        # CUDA random number generator
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # Set environment variables
        os.environ["PYTHONHASHSEED"] = str(seed)

    # --- Utility methods (formerly top-level functions) ---
    @staticmethod
    def initialize_data_and_params(key, model_type, csv_path, logger=None, seed=0):
        # Ensure logger is available
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.warning("initialize_data_and_params called without a logger.")

        # --- Check if modules are imported ---
        try:
            from wie.data.modules import fetch_data_module
            from wie.training.dataset_config import fetch_training_params
        except ImportError as e:
            logger.error(
                f"Error importing DataModule or config: {e}. Ensure they are in the correct path."
            )
            raise

        module = fetch_data_module(key, data_dir=csv_path, logger=logger, seed=seed)
        config = fetch_training_params(key, model_type)
        training_params = {
            "num_epoch": config.get("num_epoch", 21),
            "batch_size": config.get("batch_size", 60),
            "lr": config.get("lr", 0.01),
            "decay": config.get("decay", True),
        }
        data_sizes = {
            "n_tr": config.get("n_tr", 200),
            "n_val": config.get("n_val", 200),
            "n_test": config.get("n_test", 200),
        }
        module.append_one = False
        return module, data_sizes, training_params

    def get_model(self, model_type, input_dim, device):
        self.logger.debug(
            f"Getting model {model_type} with input dimension {input_dim}"
        )
        # Ensure get_network is available (already imported at top level)
        return get_network(model_type, input_dim, self.logger).to(device)

    def load_data(
        self,
        key,
        model_type,
        seed,
        csv_path,
        custom_n_tr=None,
        custom_n_val=None,
        custom_n_test=None,
        custom_num_epoch=None,
        custom_batch_size=None,
        custom_lr=None,
        custom_decay=None,
        device="cpu",
    ):
        module, data_sizes, training_params = TrainManager.initialize_data_and_params(
            key, model_type, csv_path, logger=self.logger, seed=seed
        )
        # Override defaults with custom values if provided
        if custom_n_tr:
            data_sizes["n_tr"] = custom_n_tr
        if custom_n_val:
            data_sizes["n_val"] = custom_n_val
        if custom_n_test:
            data_sizes["n_test"] = custom_n_test
        if custom_num_epoch:
            training_params["num_epoch"] = custom_num_epoch
        if custom_batch_size:
            training_params["batch_size"] = custom_batch_size
        if custom_lr:
            training_params["lr"] = custom_lr
        if custom_decay is not None:
            training_params["decay"] = custom_decay

        z_tr, z_val, _ = module.fetch(
            data_sizes["n_tr"], data_sizes["n_val"], data_sizes["n_test"], seed
        )
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val

        # Convert to tensors and move to device
        x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
        y_tr = torch.from_numpy(y_tr).to(torch.float32).unsqueeze(1).to(device)
        x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
        y_val = torch.from_numpy(y_val).to(torch.float32).unsqueeze(1).to(device)

        return x_tr, y_tr, x_val, y_val, data_sizes, training_params

    def generate_relabel_indices(self, n_samples, relabel_percentage, seed):
        np.random.seed(seed)
        num_to_relabel = int(n_samples * relabel_percentage / 100)
        return np.random.choice(n_samples, num_to_relabel, replace=False)

    def save_relabel_indices(self, indices, seed):
        """Saves relabel indices to the base save directory."""
        # Directory is already created in __init__
        filename = os.path.join(self.base_save_dir, f"relabel_indices_{seed:03d}.csv")
        try:
            pd.DataFrame({"relabel_indices": indices}).to_csv(filename, index=False)
            self.logger.info(f"Saved relabel indices to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save relabel indices to {filename}: {e}")
        return filename

    def load_relabel_indices(self, filename):
        # Ensure filename is absolute or resolve it relative to base_save_dir if necessary
        if not os.path.isabs(filename):
            filename = os.path.join(self.base_save_dir, filename)
        self.logger.info(f"Loading relabel indices from: {filename}")
        df = pd.read_csv(filename)
        return df["relabel_indices"].values

    def apply_relabeling(self, y_tr, relabel_indices):
        y_tr_clone = y_tr.clone()  # Avoid modifying original tensor if passed around
        original_labels = y_tr_clone[relabel_indices].clone()
        y_tr_clone[relabel_indices] = 1 - y_tr_clone[relabel_indices]
        flipped_labels = y_tr_clone[relabel_indices].clone()
        self.logger.info(f"Relabeled {len(relabel_indices)} samples")
        self.logger.debug(
            f"Sample of label flips: {original_labels[:5].tolist()} -> {flipped_labels[:5].tolist()}"
        )
        return y_tr_clone

    def save_at_initial(
        self,
        model,
        list_of_sgd_models,
        list_of_counterfactual_models,
        n,
        compute_counterfactual,
        filename_suffix="",
    ):
        """
        Save the initial model state and potentially lists of SGD/Counterfactual models.
        All files are saved within self.records_dir.
        """
        # Directories are already created in __init__

        # --- Save initial model ---
        # Simplified path: save directly in records_dir
        init_filename_base = f"init_{self.seed:03d}.pt"
        if self.relabel_percentage is not None:
            init_filename_base = (
                f"relabel_{self.relabel_percentage:03d}_pct_init_{self.seed:03d}.pt"
            )

        # Apply suffix
        if filename_suffix:
            base, ext = os.path.splitext(init_filename_base)
            init_filename_base = f"{base}{filename_suffix}{ext}"

        init_file = os.path.join(self.records_dir, init_filename_base)

        if self.save_recording:
            try:
                torch.save(model.state_dict(), init_file)
                self.logger.info(f"Saved initial model state to {init_file}")
            except Exception as e:
                self.logger.error(f"Failed to save initial model to {init_file}: {e}")
        else:
            self.logger.debug(
                f"Recording disabled. Skipping saving initial model state ({init_file})."
            )

        # --- Save SGD models list (only for n=-1 run) ---
        if (
            self.save_recording
            and (not self.epochs_only)
            and n == -1
            and list_of_sgd_models is not None
        ):  # Check n == -1; skip saving when epochs-only
            sgd_filename_base = f"sgd_models_{self.seed:03d}.pt"
            if self.relabel_percentage is not None:
                sgd_filename_base = f"relabel_{self.relabel_percentage:03d}_pct_sgd_models_{self.seed:03d}.pt"

            # Apply suffix
            if filename_suffix:
                base, ext = os.path.splitext(sgd_filename_base)
                sgd_filename_base = f"{base}{filename_suffix}{ext}"

            sgd_file = os.path.join(self.records_dir, sgd_filename_base)
            try:
                # Handle both file paths (for large models) and model objects (for small models)
                if self.model is None:
                    raise ValueError("Model type not specified.")
                is_large_model = self.model.lower() in ["bert", "vit", "resnet"]

                if (
                    is_large_model
                    and list_of_sgd_models
                    and isinstance(list_of_sgd_models[0], str)
                ):
                    # For large models: save list of file paths
                    torch.save(list_of_sgd_models, sgd_file)
                    self.logger.info(
                        f"Saved list of SGD model file paths to {sgd_file}"
                    )
                else:
                    # For small models: save actual model objects/state_dicts
                    torch.save(list_of_sgd_models, sgd_file)
                    self.logger.info(
                        f"Saved list of SGD models (state dicts/models) to {sgd_file}"
                    )
            except Exception as e:
                self.logger.error(f"Failed to save SGD models list to {sgd_file}: {e}")

        # --- Save counterfactual models list (if applicable) ---
        # Note: This saves the *entire list* potentially multiple times if called outside n=-1.
        # Consider saving only the final list after all n loops are done in train_and_save.
        # However, sticking to the original logic for now:
        if (
            self.save_recording
            and compute_counterfactual
            and list_of_counterfactual_models is not None
        ):
            cf_filename_base = f"counterfactual_models_{self.seed:03d}.pt"
            if self.relabel_percentage is not None:
                cf_filename_base = f"relabel_{self.relabel_percentage:03d}_pct_counterfactual_models_{self.seed:03d}.pt"

            # Apply suffix
            if filename_suffix:
                base, ext = os.path.splitext(cf_filename_base)
                cf_filename_base = f"{base}{filename_suffix}{ext}"

            cf_file = os.path.join(self.records_dir, cf_filename_base)
            try:
                torch.save(
                    list_of_counterfactual_models, cf_file
                )  # Assuming list contains NetList objects or similar
                self.logger.info(f"Saved list of counterfactual models to {cf_file}")
            except Exception as e:
                self.logger.error(
                    f"Failed to save counterfactual models list to {cf_file}: {e}"
                )

    def save_after_epoch(
        self, model, list_of_counterfactual_models, n, epoch, compute_counterfactual
    ):
        """Saves the model state for a specific counterfactual run 'n' after an epoch."""
        if compute_counterfactual and n >= 0:  # Only for counterfactual runs (n != -1)
            if list_of_counterfactual_models is None or n >= len(
                list_of_counterfactual_models
            ):
                self.logger.warning(
                    f"Counterfactual models list is not properly initialized or index {n} is out of bounds."
                )
                return

            if self.net_func is None:
                raise ValueError("Network function not initialized.")
            m = self.net_func()  # Create a new instance
            m.load_state_dict(copy.deepcopy(model.state_dict()))
            m.to("cpu")  # Move to CPU before appending

            # Ensure the target list element is a NetList and append the model
            if isinstance(list_of_counterfactual_models[n], NetList):
                list_of_counterfactual_models[n].models.append(m)
                self.logger.debug(
                    f"Appended model state for counterfactual sample {n}, epoch {epoch + 1}"
                )
            else:
                self.logger.warning(
                    f"Counterfactual list element {n} is not a NetList. Cannot append model."
                )

            # Save epoch checkpoint to disk for counterfactual runs
            if self.save_recording and not self.steps_only and epoch >= 0:
                epoch_filename = (
                    f"counterfactual_{n:04d}_epoch_{epoch}_{self.seed:03d}.pt"
                )
                if self.relabel_prefix:
                    epoch_filename = f"{self.relabel_prefix}{epoch_filename}"

                epoch_file = os.path.join(self.records_dir, epoch_filename)
                epoch_data = {
                    "model_state": copy.deepcopy(model.state_dict()),
                    "epoch": epoch,
                    "counterfactual_sample": n,
                }

                try:
                    torch.save(epoch_data, epoch_file)
                    self.logger.debug(
                        f"Saved counterfactual sample {n} epoch {epoch + 1} checkpoint to {epoch_file}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to save counterfactual sample {n} epoch {epoch + 1} checkpoint to {epoch_file}: {e}"
                    )

    def _save_each_step(self, model, total_step, idx, lr, step_loss=None, epoch=None):
        """Saves model state and info for each SGD step. Uses self attributes for paths etc.
        Now saves full model states for all models including BERT to support influence function calculations.
        This uses more memory and disk space but is necessary for proper WIE/ICML calculations.
        """
        # Respect flags controlling checkpoint recording
        # 1) Global off
        if not getattr(self, "save_recording", True):
            # Still allow step loss CSV below if configured, but skip model checkpointing
            self.logger.debug(
                f"Recording disabled. Skipping step checkpoint for step {total_step + 1}."
            )
        # 2) Epochs-only mode → don't save steps
        elif getattr(self, "epochs_only", False):
            self.logger.debug(
                f"Epochs-only mode. Skipping step checkpoint for step {total_step + 1}."
            )
        else:
            # Memory optimization: For large models like BERT, save to disk instead of keeping in memory
            if self.model is None:
                raise ValueError("Model type not specified.")
            is_large_model = self.model.lower() in ["bert", "vit", "resnet"]

            if is_large_model:
                # Save directly to disk to reduce memory usage
                step_file = os.path.join(
                    self.records_dir, f"step_{total_step + 1:06d}.pt"
                )
                try:
                    torch.save(model.state_dict(), step_file)
                    self.logger.debug(
                        f"Saved step {total_step + 1} model checkpoint to {step_file}"
                    )
                    # Keep a lightweight placeholder or counter
                    self.list_of_sgd_models.append(
                        step_file
                    )  # Store file path instead of model

                    # Force garbage collection for large models
                    import gc

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    self.logger.error(f"Failed to save step {total_step + 1}: {e}")
            else:
                # Keep in-memory list for smaller models
                if self.net_func is None:
                    raise ValueError("Network function not initialized.")
                m = self.net_func()  # Create a new instance
                m.load_state_dict(copy.deepcopy(model.state_dict()))
                m.to("cpu")
                self.list_of_sgd_models.append(m)  # Append the CPU model instance

        # --- Append step loss to CSV ---
        # Path uses self.step_loss_file (defined in train_and_save)
        if (
            self.step_loss_file is not None
            and step_loss is not None
            and epoch is not None
        ):
            try:
                write_header = not os.path.exists(self.step_loss_file)
                with open(self.step_loss_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(["step", "epoch", "loss", "idx"])
                    # Ensure idx is formatted correctly (e.g., comma-separated string)
                    idx_str = (
                        ",".join(map(str, idx))
                        if isinstance(idx, (list, np.ndarray))
                        else str(idx)
                    )
                    writer.writerow([total_step, epoch, step_loss, idx_str])
                    self.logger.debug(
                        f"Appended step loss to {self.step_loss_file}: "
                        f"Step {total_step}, Epoch {epoch}, Loss {step_loss:.4f}"
                    )
            except Exception as e:
                self.logger.error(
                    f"Failed to write step loss to {self.step_loss_file}: {e}"
                )

    def save_global_info(
        self, seed, data_sizes, training_params, alpha, filename_suffix=None
    ):
        """Saves global training parameters to a JSON file in the base save directory."""
        global_info = {
            "seed": seed,
            "target": self.target,  # Added target
            "model": self.model,  # Added model
            "n_tr": data_sizes["n_tr"],
            "n_val": data_sizes["n_val"],
            "n_test": data_sizes["n_test"],
            "num_epoch": training_params["num_epoch"],
            "batch_size": training_params["batch_size"],
            "lr": training_params["lr"],
            "decay": training_params["decay"],
            "training_params": training_params,  # Might be redundant
            "alpha": alpha,
            "relabel_csv": self.relabel_csv,  # Added relabel info
            "relabel_prefix": self.relabel_prefix,
            "relabel_percentage": self.relabel_percentage,
            "compute_counterfactual": self.compute_counterfactual,
            "init_model_path": self.init_model_path,
            "save_checkpoints": self.save_checkpoints,
            "device": str(self.device),
            # DVE spec snapshot
            "dve": {
                "enable": self.dve_enable,
                "proj_dim": self.dve_proj_dim,
                "proj_type": self.dve_proj_type,
                "granularity": self.dve_granularity,
                "last_layer_only": self.dve_last_layer_only,
                "fp16": self.dve_fp16,
                "sample_rate": self.dve_sample_rate,
            },
        }
        json_file_name = os.path.join(
            self.base_save_dir,
            f"global_info_{seed:03d}{f'_{filename_suffix}' if filename_suffix else ''}.json",
        )
        try:
            with open(json_file_name, "w") as f:
                json.dump(global_info, f, indent=4)
            self.logger.info(f"Saved global info to {json_file_name}")
        except Exception as e:
            self.logger.error(f"Failed to save global info to {json_file_name}: {e}")
        return json_file_name

    def initialize_model(self, n, net_func, model_type, seed, init_model_path):
        """Initializes or loads a model."""
        torch.manual_seed(seed)  # Ensure reproducibility for initialization
        model = net_func()  # Calls self.get_model
        model_device = next(model.parameters()).device  # Check device
        self.logger.debug(
            f"Initializing model {model_type} (run n={n}) with seed {seed} on device {model_device}"
        )

        # Load initial state only for the main run (n=-1) if path is provided
        if init_model_path and n == -1:
            # Make path absolute if it's relative
            if not os.path.isabs(init_model_path):
                init_model_path = os.path.join(
                    self.current_dir, init_model_path
                )  # Assume relative to script dir

            if os.path.exists(init_model_path):
                try:
                    # Load the state dict directly
                    state_dict = torch.load(init_model_path, map_location=self.device)
                    # Handle cases where the saved file might contain more than just the state_dict
                    if isinstance(state_dict, dict) and "model_state" in state_dict:
                        state_dict = state_dict[
                            "model_state"
                        ]  # Common pattern from _save_each_step

                    model.load_state_dict(state_dict)
                    self.logger.info(
                        f"Loaded initial model state from {init_model_path} for n=-1 run."
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load initialization model from {init_model_path}: {str(e)}. Using random initialization."
                    )
            else:
                self.logger.warning(
                    f"Initialization model path not found: {init_model_path}. Using random initialization."
                )
        elif n != -1:
            self.logger.debug(
                f"Using random initialization for counterfactual run n={n}."
            )
        else:
            self.logger.debug(
                "No init_model_path provided or n != -1. Using random initialization."
            )

        return model.to(self.device)  # Ensure model is on the correct device

    def setup_optimizer(self, model, model_type, training_params, n):
        is_vit_model = model_type.lower() == "vit"
        is_bert_model = model_type.lower() == "bert"
        unfreeze_epoch = (
            min(3, training_params["num_epoch"] // 3)
            if training_params["num_epoch"] > 0
            else 0
        )

        lr = training_params["lr"]
        momentum = 0.0  # Hardcoded momentum

        if is_bert_model:
            # Use AdamW for BERT finetuning
            lr = (
                float(training_params["lr"])
                if training_params.get("lr") is not None
                else 2e-5
            )
            weight_decay = 0.01 if training_params.get("decay", True) else 0.0
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )
            is_vit_model = False  # ensure only one path is active
            unfreeze_epoch = 0
            self.logger.debug(
                f"BERT optimizer setup (AdamW): LR={lr}, weight_decay={weight_decay}"
            )
        elif is_vit_model:
            # Separate backbone and head parameters for potential differential learning rates
            backbone_params = [
                p
                for name, p in model.named_parameters()
                if "backbone.heads.head" not in name and p.requires_grad
            ]
            head_params = [
                p
                for name, p in model.named_parameters()
                if "backbone.heads.head" in name and p.requires_grad
            ]

            # Initial setup: freeze backbone for n=-1 run
            if n == -1:
                self.logger.info(
                    "ViT model (n=-1): Freezing backbone layers for initial training epochs."
                )
                for name, param in model.named_parameters():
                    if "backbone.heads.head" not in name:
                        param.requires_grad = False
                # Update param lists after freezing
                backbone_params = [
                    p
                    for name, p in model.named_parameters()
                    if "backbone.heads.head" not in name and p.requires_grad
                ]
                head_params = [
                    p
                    for name, p in model.named_parameters()
                    if "backbone.heads.head" in name and p.requires_grad
                ]

            # Optimizer with different LR for head and (potentially frozen) backbone
            optimizer = torch.optim.SGD(
                [
                    {
                        "params": backbone_params,
                        "lr": lr * 0.1,
                    },  # Lower LR for backbone
                    {"params": head_params, "lr": lr},  # Standard LR for head
                ],
                momentum=momentum,
            )
            self.logger.debug(
                f"ViT optimizer setup: Head LR={lr}, Backbone LR={lr * 0.1}"
            )

        else:
            # Standard SGD optimizer for other models
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                momentum=momentum,
            )
            self.logger.debug(f"Standard SGD optimizer setup: LR={lr}")

        return optimizer, is_vit_model, unfreeze_epoch

    def perform_initial_evaluation(self, model, x_val, y_val, loss_fn, n):
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            self.logger.debug(
                f"Performing initial evaluation for n={n} with {x_val.shape[0]} validation samples."
            )
            # Ensure validation data is on the correct device
            x_val_dev = x_val.to(self.device)
            y_val_dev = y_val.to(self.device)
            val_logits = model(x_val_dev)
            val_loss = loss_fn(val_logits, y_val_dev).item()
            test_pred = (
                torch.sigmoid(val_logits) > 0.5
            ).float()  # Apply sigmoid for prediction
            test_acc = (test_pred == y_val_dev).float().mean().item()
            self.logger.info(
                f"Initial Eval (n={n}): Validation Loss: {val_loss:.4f}, Validation Accuracy: {test_acc:.4f}"
            )
        model.train()  # Set model back to training mode
        return val_loss, test_acc

    # ========================= DVE helpers =========================
    def _dve_find_last_linear(self, model: nn.Module) -> nn.Linear:
        last = None
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and ("classifier" in name or "head" in name):
                last = m
        if last is None:
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    last = m
        if last is None:
            raise RuntimeError("DVE: No nn.Linear layer found for last_layer_only.")
        return last

    def _dve_make_projection(self, p_last: int, device: torch.device):
        if self._dve_proj_last is not None and self._dve_proj_last.shape[1] == p_last:
            return
        torch.manual_seed(self.seed if self.seed is not None else 42)
        d = self.dve_proj_dim
        if self.dve_proj_type == "achlioptas":
            s = 3
            R = torch.zeros(d, p_last, device=device)
            mask = torch.rand(d, p_last, device=device) < (1.0 / s)
            signs = (torch.rand(d, p_last, device=device) < 0.5).float() * 2.0 - 1.0
            R[mask] = signs[mask] * ((s / d) ** 0.5)
        else:
            R = torch.randn(d, p_last, device=device) / (d**0.5)
        self._dve_proj_last = R
        try:
            spec = {
                "type": self.dve_proj_type,
                "dim": d,
                "seed": int(self.seed if self.seed is not None else 42),
                "space": "last_linear_weight_grad_vec",
            }
            with open(
                os.path.join(self.dve_root_dir, "projection_spec.json"), "w"
            ) as f:
                json.dump(spec, f, indent=2)
            torch.save(
                R.cpu(), os.path.join(self.dve_root_dir, "projection_last_layer.pt")
            )
            self.logger.info(f"DVE: projection saved to {self.dve_root_dir}")
        except Exception as e:
            self.logger.warning(f"DVE: failed to persist projection spec: {e}")

    def _dve_register_last_input_hook(self, model: nn.Module):
        self._dve_last_linear = self._dve_find_last_linear(model)

        def fwd_hook(mod, inp, out):
            self._dve_saved_last_input = inp[0].detach()

        self._dve_fwd_handle = self._dve_last_linear.register_forward_hook(fwd_hook)
        self.logger.info("DVE: forward hook registered on last Linear.")

    def _dve_remove_hooks(self):
        if hasattr(self, "_dve_fwd_handle") and self._dve_fwd_handle is not None:
            try:
                self._dve_fwd_handle.remove()
            except Exception:
                pass
            self._dve_fwd_handle = None
            self.logger.info("DVE: forward hook removed.")
        self._dve_last_linear = None
        self._dve_saved_last_input = None

    def _dve_compute_u_last(
        self, logits: torch.Tensor, y_input: torch.Tensor
    ) -> torch.Tensor:
        if self._dve_saved_last_input is None:
            raise RuntimeError("DVE: last-layer activation not captured; hook missing?")
        A = self._dve_saved_last_input  # [B, in_features]
        with torch.no_grad():
            if logits.dim() == 2 and logits.size(1) == 1:
                delta = torch.sigmoid(logits) - y_input  # BCEWithLogits
            else:
                probs = torch.softmax(logits, dim=1)
                if y_input.dim() == 2 and y_input.size(1) > 1:
                    y_idx = torch.argmax(y_input.long(), dim=1, keepdim=True)
                else:
                    y_idx = y_input.long().view(-1, 1)
                one_hot = torch.zeros_like(probs)
                one_hot.scatter_(1, y_idx, 1)
                delta = probs - one_hot
        # per-sample outer product for last layer weight gradient, then flatten
        G = torch.einsum("bi,bj->bij", delta, A).reshape(
            delta.size(0), -1
        )  # [B, p_last]
        self._dve_make_projection(G.size(1), device=G.device)
        U = torch.matmul(G, self._dve_proj_last.t())  # [B, d]
        return U

    def _dve_write_shard(
        self, *, epoch: int, step: int, lr: float, idx: list, U: torch.Tensor
    ):
        shard = {
            "epoch": int(epoch),
            "step": int(step),
            "lr": float(lr),
            "idx": list(map(int, idx)),
            "U": (U.half() if self.dve_fp16 else U.float()).cpu(),
        }
        if step >= 0:
            path = os.path.join(self.dve_raw_dir, f"dve_step_{step:06d}.pt")
        else:
            path = os.path.join(
                self.dve_raw_dir, f"dve_epoch_{epoch:03d}_part_{(-step):06d}.pt"
            )
        try:
            torch.save(shard, path)
        except Exception as e:
            self.logger.error(f"DVE: failed to write shard {path}: {e}")

    def _dve_epoch_pass(self, model: nn.Module, epoch: int):
        """Epoch-mode: perform one forward sweep over training set to produce U shards (subsample optional)."""
        self.logger.info(f"DVE: epoch-mode sweep for epoch {epoch + 1} started.")
        model.eval()
        B = int(self.training_params["batch_size"])
        n = len(self.x_tr)
        if self.dve_sample_rate < 1.0:
            k = max(1, int(n * self.dve_sample_rate))
            perm = torch.randperm(n)[:k].cpu().numpy()
        else:
            perm = torch.arange(n).cpu().numpy()
        parts = np.array_split(perm, int(np.ceil(len(perm) / B)))
        if self.dve_last_layer_only:
            self._dve_register_last_input_hook(model)

        part_counter = 0
        for part in parts:
            if len(part) == 0:
                continue
            x = self.x_tr[part].to(self.device)
            y = self.y_tr[part].to(self.device)
            with torch.no_grad():
                logits = model(x)
            U = self._dve_compute_u_last(logits, y)
            self._dve_write_shard(
                epoch=epoch,
                step=-(epoch * 10000 + part_counter + 1),
                lr=0.0,
                idx=part.tolist(),
                U=U,
            )
            part_counter += 1
        self._dve_remove_hooks()
        model.train()
        self.logger.info(
            f"DVE: epoch-mode sweep for epoch {epoch + 1} done. parts={part_counter}"
        )

    # ========================= /DVE helpers =========================

    def train_epoch(
        self,
        model,
        optimizer,
        loss_fn,
        epoch,
        n,
        total_step,
        lr_n,
        is_vit_model,
        unfreeze_epoch,
        skip=None,
    ):
        """Trains the model for one epoch."""
        model.train()  # Ensure model is in training mode
        if skip is None:
            skip = []

        # --- ViT Unfreezing Logic (only for n=-1 run) ---
        if is_vit_model and n == -1 and epoch == unfreeze_epoch:
            self.logger.info(
                f"Epoch {epoch + 1} (n=-1): Unfreezing ViT backbone layers."
            )
            for name, param in model.named_parameters():
                if "backbone.heads.head" not in name:  # Unfreeze non-head params
                    param.requires_grad = True

        epoch_loss = 0.0
        info = []  # Step-level info list (can be large)
        epoch_step_info = []  # For saving epoch checkpoints

        # Determine number of steps per epoch
        if self.x_tr is None:
            raise ValueError("Training data (x_tr) not loaded. Call load_data() first.")
        if self.training_params is None:
            raise ValueError("Training parameters not loaded. Call load_data() first.")
        num_steps_epoch = int(
            np.ceil(len(self.x_tr) / self.training_params["batch_size"])
        )

        # Use a fixed random seed to generate permutation, ensuring it's the same for every run
        np.random.seed(self.seed + epoch)
        permuted_indices = np.random.permutation(len(self.x_tr))

        # Save permutation to file for subsequent verification
        perm_file = os.path.join(self.records_dir, f"permutation_epoch_{epoch:03d}.npy")
        np.save(perm_file, permuted_indices)

        # Handle skipping sample 'n' for counterfactual runs
        if n >= 0:
            permuted_indices = np.setdiff1d(permuted_indices, [n], assume_unique=True)
            if len(permuted_indices) < len(self.x_tr):
                self.logger.debug(
                    f"Skipping sample {n} for training in epoch {epoch + 1}"
                )

        idx_list = np.array_split(permuted_indices, num_steps_epoch)

        for i, idx in enumerate(idx_list):
            if len(idx) == 0:
                continue  # Skip empty batches

            # Get batch data and move to device
            if self.x_tr is None or self.y_tr is None:
                raise ValueError("Training data not loaded. Call load_data() first.")
            x_batch = self.x_tr[idx].to(self.device)
            y_batch = self.y_tr[idx].to(self.device)

            # --- Forward pass ---
            z = model(x_batch)
            # Optional: label smoothing for BCEWithLogitsLoss
            if self.label_smoothing is not None and self.label_smoothing > 0:
                eps = float(self.label_smoothing)
                # Clamp to a safe range [0, 0.4999]
                eps = max(0.0, min(eps, 0.4999))
                y_target = y_batch * (1.0 - eps) + (1.0 - y_batch) * eps
            else:
                y_target = y_batch
            loss = loss_fn(z, y_target)
            batch_loss = loss.item()  # Loss without regularization
            epoch_loss += batch_loss * len(
                idx
            )  # Accumulate total loss weighted by batch size

            # --- Add L2 Regularization ---
            l2_reg = 0.0
            if self.alpha is not None and self.alpha > 0:
                for p in filter(lambda p: p.requires_grad, model.parameters()):
                    l2_reg += (p * p).sum()
                loss += 0.5 * self.alpha * l2_reg

            # --- Backward pass and Optimization ---
            optimizer.zero_grad()
            loss.backward()

            # Gradient scaling (optional, seems specific to original code's logic)
            # original_batch_size = self.training_params["batch_size"]
            # scale_factor = len(idx) / original_batch_size
            # for p in model.parameters():
            #     if p.grad is not None:
            #         p.grad.data *= scale_factor
            # Disabling scaling for now, as standard practice is to average or sum loss,
            # and optimizer handles batch size implicitly via gradients.

            optimizer.step()

            # --- Learning Rate Update (if decay is enabled) ---
            if self.training_params is None:
                raise ValueError(
                    "Training parameters not loaded. Call load_data() first."
                )
            if self.training_params["decay"]:
                # Pass the current optimizer state's LR
                if not optimizer.param_groups:
                    raise ValueError("Optimizer has no parameter groups.")
                current_lr_group0 = optimizer.param_groups[0]["lr"]
                _ = self.update_learning_rate(
                    optimizer,
                    is_vit_model,
                    n,
                    self.training_params,
                    epoch,
                    unfreeze_epoch,
                    total_step,
                    num_steps_epoch,
                    current_lr_group0,
                )

            # --- Record step information ---
            step_info = {
                "idx": idx.tolist(),
                "lr": optimizer.param_groups[0]["lr"],
            }  # Record actual LR used
            info.append(step_info)
            epoch_step_info.append(step_info)

            # --- Save step checkpoint (only for n=-1 run) ---
            if n == -1:
                avg_batch_loss = (
                    batch_loss  # Use loss without regularization for logging
                )
                if not optimizer.param_groups:
                    raise ValueError("Optimizer has no parameter groups.")
                self._save_each_step(
                    model,
                    total_step,
                    idx.tolist(),  # Save indices as list
                    optimizer.param_groups[0]["lr"],  # Log LR from optimizer
                    step_loss=avg_batch_loss,
                    epoch=epoch,
                )

                # --- DVE step-wise recording ---
                if self.dve_enable and self.dve_granularity == "step":
                    try:
                        # Compute DVE projections U for this step
                        U = self._dve_compute_u_last(z, y_batch)
                        # Write DVE shard for this step
                        self._dve_write_shard(
                            epoch=epoch,
                            step=total_step,
                            lr=optimizer.param_groups[0]["lr"],
                            idx=idx.tolist(),
                            U=U,
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"DVE step recording failed at step {total_step}: {e}"
                        )

                if (total_step + 1) % 50 == 0:  # Log every 50 steps
                    if not optimizer.param_groups:
                        raise ValueError("Optimizer has no parameter groups.")
                    if self.training_params is None:
                        raise ValueError(
                            "Training parameters not loaded. Call load_data() first."
                        )
                    self.logger.info(
                        f"Step {total_step + 1}/{self.training_params['num_epoch'] * num_steps_epoch}, Epoch {epoch + 1}/{self.training_params['num_epoch']}, Batch Loss: {avg_batch_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
                    )
                step_info["step_loss"] = avg_batch_loss
                step_info["global_step"] = total_step

            total_step += 1
            del z, loss, x_batch, y_batch  # Free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_epoch_loss = (
            epoch_loss / len(permuted_indices) if len(permuted_indices) > 0 else 0
        )
        return (
            avg_epoch_loss,
            total_step,
            optimizer.param_groups[0]["lr"],
            info,
            epoch_step_info,
        )

    def update_learning_rate(
        self,
        optimizer,
        is_vit_model,
        n,
        training_params,
        epoch,
        unfreeze_epoch,
        total_step,
        steps_per_epoch,
        current_lr,
    ):
        """Updates learning rate based on decay strategy. Returns the *new* base LR."""
        new_lr = current_lr  # Start with current LR

        if not training_params["decay"]:
            return new_lr  # No decay, return current LR

        initial_lr = training_params["lr"]
        num_epochs = training_params["num_epoch"]
        total_steps_expected = num_epochs * steps_per_epoch

        if is_vit_model and n == -1:
            # Cosine decay for ViT backbone fine-tuning (applied per step)
            progress = (
                min(1.0, total_step / total_steps_expected)
                if total_steps_expected > 0
                else 1.0
            )
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))

            # Determine base LR based on whether backbone is frozen
            base_lr_backbone = (
                initial_lr * 0.1 if epoch < unfreeze_epoch else initial_lr * 0.01
            )  # Further reduce after unfreezing
            base_lr_head = initial_lr

            # Apply cosine decay
            lr_backbone = base_lr_backbone * cosine_decay
            lr_head = base_lr_head * cosine_decay

            # Update optimizer param groups
            for i, param_group in enumerate(optimizer.param_groups):
                # Assuming group 0 is backbone, group 1 is head (check setup_optimizer)
                if len(param_group["params"]) > 0:  # Only update if group has params
                    if i == 0:
                        param_group["lr"] = lr_backbone
                    else:
                        param_group["lr"] = lr_head
            new_lr = lr_head  # Return the head LR as the reference 'lr_n'

        else:
            # Original sqrt decay for non-ViT models or counterfactual ViT runs
            if self.seed is None:
                raise ValueError("Seed not initialized.")
            new_lr = initial_lr * np.sqrt(
                (self.seed + 1) / (total_step + self.seed + 1)
            )  # Example sqrt decay, adjust as needed
            # Apply to all parameter groups
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

        return new_lr  # Return the potentially updated learning rate

    def evaluate_epoch(self, model, loss_fn, epoch_loss_avg, n, epoch):
        """Evaluates the model on the validation set after an epoch."""
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Ensure validation data is on device
            if self.x_val is None or self.y_val is None:
                raise ValueError("Validation data not loaded. Call load_data() first.")
            x_val_dev = self.x_val.to(self.device)
            y_val_dev = self.y_val.to(self.device)

            val_logits = model(x_val_dev)
            val_loss = loss_fn(val_logits, y_val_dev).item()
            test_pred = (
                torch.sigmoid(val_logits) > 0.5
            ).float()  # Use sigmoid for accuracy calc
            test_acc = (test_pred == y_val_dev).float().mean().item()
            train_loss = epoch_loss_avg  # Use the calculated average epoch loss

            if self.training_params is None:
                raise ValueError(
                    "Training parameters not loaded. Call load_data() first."
                )
            self.logger.info(
                f"Epoch {epoch + 1}/{self.training_params['num_epoch']} (n={n}) Summary: "
                f"Avg Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {test_acc:.4f}"
            )
        model.train()  # Set model back to training mode
        return val_loss, test_acc, train_loss

    def save_epoch_data(
        self, model, epoch, epoch_step_info, val_loss, test_acc, train_loss, n, seed
    ):
        """Saves epoch summary data (only for n=-1 run)."""
        if n == -1:  # Only save epoch checkpoints for the main training run
            epoch_data = {
                "model_state": copy.deepcopy(model.state_dict()),  # Save state dict
                "epoch": epoch,
                "step_info": epoch_step_info,  # List of step infos for this epoch
                "val_loss": val_loss,
                "test_acc": test_acc,
                "train_loss": train_loss,  # Average train loss for the epoch
            }
            epoch_filename = f"epoch_{epoch}_{seed:03d}.pt"
            if self.relabel_prefix:
                epoch_filename = f"{self.relabel_prefix}{epoch_filename}"

            # Path uses self.records_dir
            epoch_file = os.path.join(self.records_dir, epoch_filename)

            if self.save_recording and not self.steps_only:
                try:
                    torch.save(epoch_data, epoch_file)
                    self.logger.info(
                        f"Saved epoch {epoch + 1} checkpoint to {epoch_file}"
                    )
                    # Add more detailed logs
                    self.logger.debug(
                        f"Epoch file details - Path: {epoch_file}, Prefix: {self.relabel_prefix}, "
                        f"Epoch: {epoch}, Seed: {self.seed:03d}, Val Loss: {val_loss:.4f}, "
                        f"Test Acc: {test_acc:.4f}, Train Loss: {train_loss:.4f}, "
                        f"Steps in epoch: {len(epoch_step_info)}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to save epoch {epoch + 1} checkpoint to {epoch_file}: {e}"
                    )
            else:
                if self.steps_only:
                    self.logger.info(
                        f"Steps-only mode enabled. Skipping epoch {epoch + 1} checkpoint saving."
                    )
                else:
                    self.logger.info(
                        f"Model recording disabled. Skipping epoch {epoch + 1} checkpoint saving."
                    )

    # --- Main orchestration methods ---
    def train_and_save(self):
        """Main method to orchestrate loading, training, and saving."""
        logger = self.logger  # Use instance logger
        # Directories are created in __init__

        # Define output file paths (no more .dat files)
        metrics_csv_file_name = os.path.join(
            self.base_save_dir,
            f"{self.relabel_prefix}training_metrics_{self.seed:03d}.csv",
        )

        # --- Load Data ---
        try:
            (
                self.x_tr,
                self.y_tr,
                self.x_val,
                self.y_val,
                self.data_sizes,
                self.training_params,
            ) = self.load_data(
                self.target,
                self.model,
                self.seed,
                self.csv_path,
                self.n_tr,
                self.n_val,
                self.n_test,
                self.num_epoch,
                self.batch_size,
                self.lr,
                self.decay,
                device=self.device or "cpu",
            )
            self.n_tr = self.data_sizes[
                "n_tr"
            ]  # Update n_tr from actual loaded data size
            logger.info(
                f"Dataset {self.target} loaded: {self.data_sizes['n_tr']} train, {self.data_sizes['n_val']} val samples."
            )
            logger.info(
                f"Training Params: Epochs={self.training_params['num_epoch']}, BatchSize={self.training_params['batch_size']}, LR={self.training_params['lr']}, Decay={self.training_params['decay']}"
            )

        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            return  # Cannot proceed without data

        # --- Apply Relabeling (if specified) ---
        if self.relabel_csv:
            try:
                # Check if the path is absolute, otherwise assume relative to base_save_dir
                relabel_csv_path = self.relabel_csv
                if not os.path.isabs(relabel_csv_path):
                    relabel_csv_path = os.path.join(
                        self.base_save_dir, relabel_csv_path
                    )

                if os.path.exists(relabel_csv_path):
                    relabel_indices = self.load_relabel_indices(relabel_csv_path)
                    self.y_tr = self.apply_relabeling(self.y_tr, relabel_indices)
                    logger.info(f"Applied relabeling from file: {relabel_csv_path}")
                else:
                    logger.warning(
                        f"Relabel CSV file not found: {relabel_csv_path}. Proceeding without explicit relabeling from file."
                    )
            except Exception as e:
                logger.error(
                    f"Error applying relabeling from {self.relabel_csv}: {str(e)}. Proceeding without relabeling.",
                    exc_info=True,
                )
        elif self.relabel_percentage is not None and self.relabel_percentage > 0:
            logger.info(
                f"Relabel CSV not provided, but relabel_percentage={self.relabel_percentage}%. Generating and saving indices."
            )
            relabel_indices = self.generate_relabel_indices(
                self.n_tr, self.relabel_percentage, self.seed
            )
            saved_relabel_path = self.save_relabel_indices(relabel_indices, self.seed)
            self.y_tr = self.apply_relabeling(self.y_tr, relabel_indices)
            logger.info(
                f"Generated, applied, and saved relabeling indices for {self.relabel_percentage}% of data to {saved_relabel_path}"
            )
            self.relabel_csv = os.path.basename(
                saved_relabel_path
            )  # Store the generated filename

        else:
            logger.info("No relabeling specified.")

        # --- Setup Model Specifics (like alpha for logreg) ---
        if self.model == "logreg":
            try:
                # Note: LogisticRegressionCV expects 2D X and 1D y numpy arrays
                model_logreg = LogisticRegressionCV(
                    random_state=self.seed,
                    fit_intercept=False,
                    cv=5,
                    Cs=10,
                    solver="liblinear",
                    penalty="l2",
                )  # Added params
                x_tr_np = self.x_tr.view(self.n_tr, -1).cpu().numpy()
                y_tr_np = self.y_tr.view(self.n_tr).cpu().numpy()
                model_logreg.fit(x_tr_np, y_tr_np)
                # Alpha is related to the inverse of the regularization strength C
                self.alpha = (
                    1.0 / (model_logreg.C_[0] * self.n_tr)
                    if model_logreg.C_[0] > 0
                    else 0.0
                )  # Avoid division by zero
                logger.info(
                    f"Logistic Regression CV selected C={model_logreg.C_[0]}, corresponding alpha={self.alpha:.6f}"
                )
                # Ensure y_tr remains the correct shape for subsequent PyTorch loss functions
                self.y_tr = self.y_tr.float().view(-1, 1).to(self.device)
            except Exception as e:
                logger.error(
                    f"Error during LogisticRegressionCV fitting: {e}. Using default alpha.",
                    exc_info=True,
                )
                self.alpha = 0.001  # Default alpha if logreg fails
                self.logger.warning(f"Using default alpha={self.alpha}")

        else:
            # Default alpha for non-logreg models (can be overridden by config if needed)
            self.alpha = self.training_params.get(
                "alpha", 0.001
            )  # Get alpha from params or default
            logger.info(f"Using alpha (L2 reg strength): {self.alpha}")

        # --- Further Initializations ---
        self.input_dim = self.x_tr.shape[
            1:
        ]  # Get input dimensions (e.g., [C, H, W] or [Features])
        self.net_func = lambda: self.get_model(self.model, self.input_dim, self.device)
        # Calculate num_steps based on actual batch size and n_tr
        self.num_steps = int(np.ceil(self.n_tr / self.training_params["batch_size"]))

        # Initialize list for counterfactual models if needed
        if self.compute_counterfactual:
            self.list_of_counterfactual_models = [NetList([]) for _ in range(self.n_tr)]
            logger.info(f"Initialized list for {self.n_tr} counterfactual model runs.")

        # --- Save Global Info ---
        json_file_name = self.save_global_info(
            self.seed, self.data_sizes, self.training_params, self.alpha
        )
        logger.info(f"Global info saved to {json_file_name}")

        # --- Main Training Loop ---
        # sgd_info = None
        sgd_main_losses = []
        sgd_test_accuracies = []
        sgd_train_losses = []

        # Loop from n=-1 (main run) up to n_tr-1 (counterfactuals)
        run_range = range(-1, self.n_tr if self.compute_counterfactual else 0)

        for n in run_range:
            if n == -1:
                logger.info(f"--- Starting Main Training Run (n={n}) ---")
            else:
                logger.info(
                    f"--- Starting Counterfactual Training Run for sample n={n} ---"
                )

            try:
                # DVE announce
                if self.dve_enable and n == -1:
                    self.logger.info(
                        f"DVE enabled: granularity={self.dve_granularity}, "
                        f"proj_dim={self.dve_proj_dim}, proj_type={self.dve_proj_type}, "
                        f"last_layer_only={self.dve_last_layer_only}, fp16={self.dve_fp16}, "
                        f"sample_rate={self.dve_sample_rate}"
                    )

                model_n, info_n, main_losses_n, test_accuracies_n, train_losses_n = (
                    self.train_single_model(n)
                )

                if n == -1:
                    # Store results from the main run
                    # sgd_info = info_n
                    sgd_main_losses = main_losses_n
                    sgd_test_accuracies = test_accuracies_n
                    sgd_train_losses = train_losses_n
                    logger.info(f"--- Completed Main Training Run (n={n}) ---")
                    # Minimal retention: always save final model state for downstream influence
                    try:
                        final_file = os.path.join(
                            self.records_dir, f"epoch_final_{self.seed:03d}.pt"
                        )
                        torch.save(model_n.state_dict(), final_file)
                        logger.info(
                            f"Saved final model state for main run to {final_file}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to save final model state for main run: {e}"
                        )
                else:
                    logger.info(
                        f"--- Completed Counterfactual Training Run for sample n={n} ---"
                    )

                # Clean up GPU memory after each run if possible
                del model_n
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error during training run n={n}: {e}", exc_info=True)
                if n == -1:
                    logger.critical("Main training run failed. Aborting.")
                    return  # Stop if the main run fails
                else:
                    logger.warning(f"Counterfactual run n={n} failed. Continuing...")

        # --- Verification and Saving ---
        logger.info("--- Post-Training Verification and Saving ---")

        # Verify number of saved SGD models (appended in _save_each_step)
        # Expected: 1 (initial) + num_epochs * num_steps
        expected_sgd_models = 1 + self.training_params["num_epoch"] * self.num_steps
        if len(self.list_of_sgd_models) != expected_sgd_models:
            logger.warning(
                f"Unexpected number of SGD models in list. Expected {expected_sgd_models}, got {len(self.list_of_sgd_models)}"
            )
        else:
            logger.info(
                f"Correct number of SGD models collected in list: {len(self.list_of_sgd_models)}"
            )

        # Verify counterfactual models (appended in save_after_epoch)
        if self.compute_counterfactual and self.list_of_counterfactual_models:
            for i, models in enumerate(self.list_of_counterfactual_models):
                # Expected: 1 (initial state saved implicitly at start) + num_epochs
                expected_cf_models = self.training_params[
                    "num_epoch"
                ]  # Models are saved *after* each epoch
                if not isinstance(models, NetList):
                    logger.warning(
                        f"Counterfactual entry {i} is not a NetList. Skipping verification."
                    )
                    continue
                if len(models.models) != expected_cf_models:
                    logger.warning(
                        f"Unexpected number of counterfactual models for sample {i}. Expected {expected_cf_models} (after each epoch), got {len(models.models)}"
                    )
                else:
                    logger.debug(
                        f"Correct number of counterfactual models saved for sample {i}: {len(models.models)}"
                    )

        # --- Save Training Results ---
        final_data_to_save = {}
        if sgd_main_losses:  # Check if main run completed
            final_data_to_save = {  # noqa: F841
                "main_losses": sgd_main_losses,
                "test_accuracies": sgd_test_accuracies,
                "train_losses": sgd_train_losses,
                # Include 'info' only if needed and manage its size
                # "info": sgd_info, # Can be very large
            }

            # Note: .dat file saving has been removed to focus on epoch checkpoints
            logger.info(
                "Training data will be saved as individual epoch checkpoints only."
            )

        # --- Save Metrics to CSV ---
        if sgd_main_losses:
            try:
                # Ensure all lists have the same length (should be num_epochs + 1)
                num_entries = len(sgd_main_losses)
                metrics_dict = {
                    "epoch": list(range(num_entries)),  # 0 to num_epochs
                    "val_loss": sgd_main_losses,
                    "val_accuracy": sgd_test_accuracies,
                    "train_loss_avg": sgd_train_losses,  # Contains NaN for epoch 0
                }
                metrics_df = pd.DataFrame(metrics_dict)
                metrics_df.to_csv(metrics_csv_file_name, index=False)
                logger.info(f"Training metrics saved to {metrics_csv_file_name}")
            except Exception as e:
                logger.error(f"Failed to save metrics to {metrics_csv_file_name}: {e}")
        else:
            logger.warning("Main training run did not complete. Metrics not saved.")

        logger.info(f"--- Training process finished for seed {self.seed}. ---")

    def train_single_model(self, n):
        """Trains one model instance (n=-1 for main, n>=0 for counterfactual)."""
        # --- Setup Reproducibility ---
        torch.manual_seed(self.seed + n + 1)  # Seed for this specific run
        np.random.seed(self.seed + n + 1)
        # CUDNN settings (optional, can impact performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        # --- Initialize Model, Loss, Optimizer ---
        model = self.initialize_model(
            n, self.net_func, self.model, self.seed, self.init_model_path
        )
        # Ensure model is on the correct device after potential loading
        model = model.to(self.device)

        loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits
        optimizer, is_vit_model, unfreeze_epoch = self.setup_optimizer(
            model, self.model, self.training_params, n
        )

        # Get initial learning rate
        lr_n = optimizer.param_groups[0]["lr"]  # Get LR from optimizer after setup

        # --- Initial Evaluation ---
        val_loss_init, test_acc_init = self.perform_initial_evaluation(
            model, self.x_val, self.y_val, loss_fn, n
        )
        main_losses = [val_loss_init]
        test_accuracies = [test_acc_init]
        train_losses = [np.nan]  # No training loss before epoch 1

        # --- Save Initial State ---
        # Append initial model state to list_of_sgd_models only for the main run (n=-1)
        # Counterfactual initial states are implicitly the same as the main run's initial state
        if n == -1:
            if self.model is None:
                raise ValueError("Model type not specified.")
            is_large_model = self.model.lower() in ["bert", "vit", "resnet"]

            if is_large_model and self.save_recording and (not self.epochs_only):
                # For large models, save initial state to disk only when recording is enabled
                initial_file = os.path.join(self.records_dir, "step_000000.pt")
                try:
                    torch.save(model.state_dict(), initial_file)
                    self.list_of_sgd_models.append(initial_file)  # Store file path
                    self.logger.debug(f"Saved initial model state to {initial_file}")
                except Exception as e:
                    self.logger.error(f"Failed to save initial model: {e}")
            else:
                # For smaller models, keep in memory
                if self.net_func is None:
                    raise ValueError("Network function not initialized.")
                initial_model_copy = self.net_func()
                initial_model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
                initial_model_copy.to("cpu")
                self.list_of_sgd_models.append(initial_model_copy)
                self.logger.debug("Appended initial model state to list_of_sgd_models.")

        # Save initial model file and potentially CF list (if compute_counterfactual)
        # Note: save_at_initial saves the *lists*, not just the initial model state here.
        # It might be better named save_lists_at_initial or similar.
        # If n != -1, list_of_sgd_models is passed as None to avoid saving it repeatedly.
        self.save_at_initial(
            model,
            self.list_of_sgd_models if n == -1 else None,
            self.list_of_counterfactual_models,
            n,
            self.compute_counterfactual,
            filename_suffix="",  # Remove duplicate relabel_prefix
        )
        # Save initial state for counterfactual run 'n' if needed
        if n != -1 and self.compute_counterfactual:
            self.save_after_epoch(
                model,
                self.list_of_counterfactual_models,
                n,
                -1,
                self.compute_counterfactual,
            )  # Save epoch -1 state

        # --- Epoch Loop ---
        info = []  # Collect step info across all epochs for this run
        total_step = 0  # Step counter for this specific run (n)

        if self.dve_enable and n == -1 and self.dve_last_layer_only:
            # optional: ensure the hook is registered early (idempotent)
            try:
                self._dve_register_last_input_hook(model)
            except Exception as e:
                self.logger.warning(f"DVE: failed to register hook early: {e}")

        if self.training_params is None:
            raise ValueError("Training parameters not loaded. Call load_data() first.")
        for epoch in range(self.training_params["num_epoch"]):
            epoch_loss_avg, total_step, lr_n, epoch_info, epoch_step_info = (
                self.train_epoch(
                    model,
                    optimizer,
                    loss_fn,
                    epoch,
                    n,
                    total_step,
                    lr_n,
                    is_vit_model,
                    unfreeze_epoch,
                    skip=None,  # Skipping is handled inside train_epoch based on 'n'
                )
            )
            info.extend(epoch_info)  # Accumulate step info

            val_loss, test_acc, train_loss = self.evaluate_epoch(
                model, loss_fn, epoch_loss_avg, n, epoch
            )

            # Append results
            main_losses.append(val_loss)
            test_accuracies.append(test_acc)
            train_losses.append(train_loss)

            # Save epoch checkpoint (only for n=-1 run)
            self.save_epoch_data(
                model,
                epoch,
                epoch_step_info,
                val_loss,
                test_acc,
                train_loss,
                n,
                self.seed,
            )

            # --- DVE epoch-wise recording ---
            if n == -1 and self.dve_enable and self.dve_granularity == "epoch":
                try:
                    self._dve_epoch_pass(model, epoch)
                except Exception as e:
                    self.logger.warning(
                        f"DVE epoch recording failed at epoch {epoch}: {e}"
                    )

            # Save model state after epoch for counterfactual runs (n != -1)
            if n != -1:
                self.save_after_epoch(
                    model,
                    self.list_of_counterfactual_models,
                    n,
                    epoch,  # Current epoch completed
                    self.compute_counterfactual,
                )

            # Optional: GPU cache clearing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Clean up DVE hooks if enabled
        if self.dve_enable and n == -1:
            self._dve_remove_hooks()

        return model, info, main_losses, test_accuracies, train_losses

    @staticmethod
    def main():
        import traceback  # Keep traceback for detailed error logging

        # --- Argument Parsing ---
        args = parse_arguments()

        # --- Setup Device ---
        # Determine the actual GPU index if specified
        real_gpu = get_real_gpu_index(args.gpu)  # Returns None if gpu < 0 or invalid
        device = get_device(real_gpu)  # Returns 'cpu', 'cuda:X', or 'mps'

        # --- Setup Logging ---
        # Use standardized path resolution for logging directory
        log_dir_abs = str(resolve_output_dir(args.save_dir))
        os.makedirs(log_dir_abs, exist_ok=True)  # Ensure log directory exists

        # Compute log GPU id if needed by formatters (already shown via setup_logging)
        log_level = getattr(logging, args.log_level.upper(), logging.INFO)
        log_file_prefix = f"{args.target}_{args.model}"
        logger = setup_logging(
            log_file_prefix, args.seed, log_dir_abs, gpu=real_gpu or 0, level=log_level
        )

        logger.info("--- Starting Training Script ---")
        logger.info(f"Arguments: {vars(args)}")
        logger.info(f"Using device: {device}")
        logger.info(f"Log level: {args.log_level.upper()}")

        try:
            # --- Handle Relabeling Setup ---
            # Pass the absolute save directory path to the helper function
            relabel_csv, relabel_prefix, relabel_percentage = handle_relabeling(
                args,
                log_dir_abs,
                logger,
                TrainManager.initialize_data_and_params,
                current_dir,
            )
            # Update args namespace if relabel_csv was generated
            if relabel_csv and not args.relabel_csv:
                args.relabel_csv = os.path.basename(
                    relabel_csv
                )  # Store filename if generated
                logger.info(f"Using generated relabel CSV: {args.relabel_csv}")

            # --- Validate Inputs ---
            if args.target not in DATA_MODULE_REGISTRY:
                raise ValueError(
                    f"Invalid target data: {args.target}. Available: {', '.join(DATA_MODULE_REGISTRY.keys())}"
                )
            if args.model not in NETWORK_REGISTRY:
                raise ValueError(
                    f"Invalid model type: {args.model}. Available: {', '.join(NETWORK_REGISTRY.keys())}"
                )

            # --- Optional: Apply dropout via CLI by setting env var for BERT head ---
            if hasattr(args, "dropout") and args.dropout is not None:
                try:
                    p = float(args.dropout)
                    if 0.0 <= p < 1.0:
                        os.environ["HF_TEXT_DROPOUT"] = str(p)
                        logger.info(
                            f"Applied head dropout via env: HF_TEXT_DROPOUT={p}"
                        )
                    else:
                        logger.warning(
                            f"Ignoring invalid --dropout value {p}; must be in [0,1)."
                        )
                except Exception as e:
                    logger.warning(f"Failed to parse --dropout: {e}")

            # Resolve label smoothing (CLI overrides env)
            ls_value = getattr(args, "label_smoothing", None)
            if ls_value is None:
                ls_env = os.getenv("LABEL_SMOOTHING")
                if ls_env is not None:
                    try:
                        ls_value = float(ls_env)
                        logger.info(f"Using label smoothing from env: {ls_value}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse LABEL_SMOOTHING='{ls_env}': {e}. Disabling."
                        )

            # --- Initialize and Run Training Manager ---
            logger.info("Initializing TrainManager...")
            manager = TrainManager(
                target=args.target,
                model=args.model,
                seed=args.seed,
                save_dir=args.save_dir,  # Pass the relative path from args
                csv_path=getattr(
                    args, "csv_path", None
                ),  # Use getattr for optional args
                n_tr=getattr(args, "n_tr", None),
                n_val=getattr(args, "n_val", None),
                n_test=getattr(args, "n_test", None),
                num_epoch=getattr(args, "num_epoch", None),
                batch_size=getattr(args, "batch_size", None),
                lr=getattr(args, "lr", None),
                decay=getattr(args, "decay", None),
                compute_counterfactual=getattr(args, "compute_counterfactual", False),
                init_model_path=getattr(args, "init_model", None),
                save_recording=getattr(args, "save_recording", True),
                steps_only=getattr(args, "steps_only", False),
                epochs_only=getattr(args, "epochs_only", False),
                relabel_csv=args.relabel_csv,  # Use potentially updated value
                relabel_prefix=relabel_prefix,
                relabel_percentage=relabel_percentage,
                device=device,
                logger=logger,
                alpha=getattr(args, "alpha", None),  # Use potentially updated value
                label_smoothing=ls_value,
                # DVE CLI (getattr: backward-compatible if arg_parser未定义)
                dve_enable=getattr(args, "dve_enable", False),
                dve_proj_dim=getattr(args, "dve_proj_dim", 128),
                dve_proj_type=getattr(args, "dve_proj_type", "gaussian"),
                dve_granularity=getattr(args, "dve_granularity", "step"),
                dve_last_layer_only=getattr(args, "dve_last_layer_only", True),
                dve_fp16=getattr(args, "dve_fp16", True),
                dve_sample_rate=getattr(args, "dve_sample_rate", 1.0),
            )

            logger.info("Starting train_and_save process...")
            manager.train_and_save()
            logger.info("--- Training Script Finished Successfully ---")

        except ValueError as e:
            logger.error(f"Configuration Error: {str(e)}", exc_info=True)
        except FileNotFoundError as e:
            logger.error(f"File Not Found Error: {str(e)}", exc_info=True)
        except ImportError as e:
            logger.error(
                f"Import Error: {str(e)}. Check dependencies and paths.", exc_info=True
            )
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during the training process: {str(e)}"
            )
            logger.error(traceback.format_exc())  # Log the full traceback


if __name__ == "__main__":
    TrainManager.main()
