import os
import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Union, Type, Optional
import warnings
import logging
import gc

# Keep dataset fetch via legacy module to avoid moving DataModule now
from wie.data.modules import fetch_data_module  # type: ignore
from wie.models.networks import get_network  # type: ignore

from wie.utils import get_device as _bridge_get_device
from wie.io.paths import get_file_paths_general
from wie.io.records import (
    load_global_info as _bridge_load_global_info,
    load_epoch_data as _bridge_load_epoch_data,
    load_step_data as _bridge_load_step_data,
)

# --- Standard Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
warnings.simplefilter(action="ignore", category=FutureWarning)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SCRIPT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# --- Constants ---
BATCH_SIZE_ICML = 200
LR_ICML = 0.01
MOMENTUM_ICML = 0.9
NUM_EPOCHS_ICML = 100


# Reuse unified device selection from wie.utils
def get_device(gpu: int) -> str:  # noqa: D401
    """Return runtime device string (mps/cuda:idx/cpu)."""
    return _bridge_get_device(gpu)


# Prefer centralized IO helpers from wie.io to avoid duplication
load_global_info = _bridge_load_global_info
load_epoch_data = _bridge_load_epoch_data
load_step_data = _bridge_load_step_data


def get_file_paths(
    key: str,
    model_type: str,
    seed: int,
    infl_type: str = None,
    save_dir: str = None,
    relabel_percentage: float = None,
) -> Tuple[str, str, str]:
    """Thin wrapper delegating to wie.io.paths.

    Keeps legacy API while centralizing path logic under src/.
    """
    return get_file_paths_general(
        SCRIPT_DIR,
        key,
        model_type,
        seed,
        infl_type,
        save_dir,
        relabel_percentage,
    )


def load_data(
    key: str,
    global_info: Dict[str, Any],
    seed: int,
    device: str,
    logger: logging.Logger,
    relabel_percentage: float = None,
    dn: str = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_tr = global_info["n_tr"]
    n_val = global_info["n_val"]
    n_test = global_info["n_test"]

    module = fetch_data_module(
        key,
        data_dir=os.path.join(SCRIPT_DIR, "experiment", "data"),
        logger=logger,
        seed=seed,
    )
    module.append_one = False

    z_tr, z_val, z_test = module.fetch(n_tr, n_val, n_test, seed)
    (x_tr_np, y_tr_np), (x_val_np, y_val_np) = z_tr, z_val

    x_tr = torch.from_numpy(x_tr_np).to(torch.float32).to(device)
    y_tr = torch.from_numpy(y_tr_np).to(torch.float32).unsqueeze(1).to(device)
    x_val = torch.from_numpy(x_val_np).to(torch.float32).to(device)
    y_val = torch.from_numpy(y_val_np).to(torch.float32).unsqueeze(1).to(device)
    # Free NumPy arrays
    del x_tr_np, y_tr_np, x_val_np, y_val_np

    if relabel_percentage and dn:
        relabel_prefix = f"relabel_{int(relabel_percentage):03d}_pct_"
        idx_csv_name = os.path.join(dn, f"{relabel_prefix}indices_{seed:03d}.csv")
        logger.info(
            f"Attempting to relabel {relabel_percentage}% of training data using {idx_csv_name}"
        )
        try:
            relabel_indices_df = pd.read_csv(idx_csv_name)
            relabel_col = None
            possible_cols = ["relabel_indices", "index", "idx"]
            for col in possible_cols:
                if col in relabel_indices_df.columns:
                    relabel_col = col
                    break
            if relabel_col is None:
                raise ValueError(
                    f"Cannot find relabeled indices column in {idx_csv_name} (tried: {possible_cols})"
                )
            relabel_indices = relabel_indices_df[relabel_col].values
            if len(relabel_indices) > 0:
                max_idx = relabel_indices.max()
                if max_idx >= n_tr:
                    logger.error(
                        f"Relabeled index {max_idx} is out of bounds for training data size {n_tr}. Check {idx_csv_name}"
                    )
                    raise IndexError("Relabeled index out of bounds")
                y_tr[relabel_indices] = 1 - y_tr[relabel_indices]
                logger.info(f"Successfully relabeled {len(relabel_indices)} samples.")
            else:
                logger.warning(
                    f"Relabeled indices file {idx_csv_name} was empty or contained no indices."
                )
        except FileNotFoundError:
            logger.error(
                f"Relabeled indices file not found: {idx_csv_name}. Training data NOT relabeled."
            )
        except Exception as e:
            logger.error(
                f"Error reading or applying relabeled indices from {idx_csv_name}: {e}. Training data NOT relabeled."
            )
    return x_tr, y_tr, x_val, y_val


def get_input_dim(x: torch.Tensor, model_type: str) -> Union[int, Tuple[int, ...]]:
    # Flat models (tabular / MLP) consume a single feature dimension.
    if model_type in ["logreg", "dnn"]:
        if x.dim() > 2:
            return x.shape[1:].numel()
        else:
            return x.shape[1]
    # All other (image) models -- cnn, resnet*, tinyvit*, vit, mobilenet* --
    # need the (C, H, W) tuple. Previously only "cnn" got this treatment, so
    # ViT/ResNet influence scoring failed with "input_dim must be (C, H, W)".
    if x.dim() == 4:
        return x.shape[1:]
    elif x.dim() == 3:
        img_size = x.shape[1]
        return (1, img_size, img_size)
    elif x.dim() == 2:
        num_features = x.shape[1]
        img_size = int(np.sqrt(num_features))
        if img_size * img_size != num_features:
            raise ValueError(
                f"Cannot infer image dimensions for {model_type} from flattened "
                f"input of size {num_features}"
            )
        return (1, img_size, img_size)
    else:
        raise ValueError(
            f"Unsupported input dimension for {model_type}: {x.dim()}"
        )


def compute_gradient(
    x: torch.Tensor, y: torch.Tensor, model: torch.nn.Module, loss_fn: torch.nn.Module
) -> List[torch.Tensor]:
    model.eval()
    model.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    grads = [
        (p.grad.data.clone() if p.grad is not None else torch.zeros_like(p))
        for p in model.parameters()
    ]
    model.zero_grad()
    return grads


def load_model_state_from_fallback(
    fn_fallback: str, target_state: str, device: str, logger: logging.Logger
) -> Dict:
    logger.warning(
        f"Attempting to load {target_state} model state from fallback file: {fn_fallback}"
    )
    try:
        res = torch.load(fn_fallback, map_location=device, weights_only=False)
        models_attr = res.get("models")
        is_netlist_like = hasattr(models_attr, "models") and isinstance(
            getattr(models_attr, "models", None), list
        )
        if is_netlist_like:
            models_list = models_attr.models
            if models_list:
                if target_state == "init":
                    state_dict = models_list[0].state_dict()
                    logger.info(
                        f"Loaded initial model state from fallback {fn_fallback} (models list index 0)"
                    )
                    return state_dict
                elif target_state == "final":
                    state_dict = models_list[-1].state_dict()
                    logger.info(
                        f"Loaded final model state from fallback {fn_fallback} (models list index -1)"
                    )
                    return state_dict
                else:
                    logger.error(
                        f"Unknown target_state '{target_state}' for fallback loading from models list."
                    )
                    raise ValueError("Invalid target_state for fallback model loading")
        logger.error(
            f"Fallback file {fn_fallback} does not contain a suitable models list for target '{target_state}'."
        )
        raise ValueError("Invalid fallback content for model loading")
    except FileNotFoundError:
        logger.error(f"Fallback file not found: {fn_fallback}")
        raise


def load_initial_model(
    dn: str, seed: int, device: str, logger: logging.Logger
) -> Dict[str, torch.Tensor]:
    try:
        init_pt = os.path.join(dn, "records", f"init_{seed:03d}.pt")
        return torch.load(init_pt, map_location=device)
    except FileNotFoundError:
        logger.warning(
            f"init file not found under {dn}/records. Trying fallback .dat file."
        )
        _, fn_fallback_dat, _ = get_file_paths("_", "_", seed, save_dir=dn)
        return load_model_state_from_fallback(fn_fallback_dat, "init", device, logger)


def load_final_model(
    dn: str, seed: int, device: str, logger: logging.Logger
) -> Dict[str, torch.Tensor]:
    try:
        final_pt = os.path.join(dn, "records", f"epoch_final_{seed:03d}.pt")
        return torch.load(final_pt, map_location=device)
    except FileNotFoundError:
        logger.warning(
            f"final epoch file not found under {dn}/records. Trying fallback .dat file."
        )
        _, fn_fallback_dat, _ = get_file_paths("_", "_", seed, save_dir=dn)
        return load_model_state_from_fallback(fn_fallback_dat, "final", device, logger)


def save_results(
    infl_data: Union[np.ndarray, List[np.ndarray]],
    dn: str,
    seed: int,
    infl_type: str,
    logger: logging.Logger,
    relabel_percentage: float | None = None,
):
    import json

    relabel_prefix = (
        f"relabel_{int(relabel_percentage):03d}_pct_"
        if relabel_percentage is not None
        else ""
    )
    # Save numpy arrays as CSV/JSON for easier reading
    if isinstance(infl_data, list):
        # wie_all_epochs style: per-epoch arrays stacked by columns
        logger.info(f"[DEBUG] Processing list of {len(infl_data)} epoch arrays")

        # Debug: Log shapes of all arrays before processing
        for i, arr in enumerate(infl_data):
            arr_np = np.asarray(arr)
            logger.info(
                f"[DEBUG] Epoch {i} array shape: {arr_np.shape}, dtype: {arr_np.dtype}"
            )

        # Get the first array to determine the length
        first_arr_flat = np.asarray(infl_data[0]).flatten()
        logger.info(f"[DEBUG] First array flattened shape: {first_arr_flat.shape}")
        df = pd.DataFrame({"sample_idx": np.arange(len(first_arr_flat))})

        for i, arr in enumerate(infl_data):
            # Ensure array is 1D for pandas compatibility
            arr_np = np.asarray(arr)
            arr_flat = arr_np.flatten()
            logger.info(
                f"[DEBUG] Epoch {i}: original shape {arr_np.shape} -> flattened shape {arr_flat.shape}"
            )

            # Validate array length consistency
            if len(arr_flat) != len(first_arr_flat):
                logger.warning(
                    f"[DEBUG] Length mismatch at epoch {i}: expected {len(first_arr_flat)}, got {len(arr_flat)}"
                )

            df[f"influence_epoch_{i}"] = arr_flat

        out_csv = os.path.join(dn, f"infl_{infl_type}_{relabel_prefix}{seed:03d}.csv")
        logger.info(f"[DEBUG] Saving DataFrame with shape {df.shape} to {out_csv}")
        df.to_csv(out_csv, index=False)
        logger.info(f"Saved influence CSV: {out_csv}")
    else:
        # single array
        infl_data_np = np.asarray(infl_data)
        logger.info(
            f"[DEBUG] Processing single array with shape: {infl_data_np.shape}, dtype: {infl_data_np.dtype}"
        )

        infl_data_flat = infl_data_np.flatten()
        logger.info(f"[DEBUG] Flattened to shape: {infl_data_flat.shape}")

        df = pd.DataFrame(
            {"sample_idx": np.arange(len(infl_data_flat)), "influence": infl_data_flat}
        )
        out_csv = os.path.join(dn, f"infl_{infl_type}_{relabel_prefix}{seed:03d}.csv")
        logger.info(f"[DEBUG] Saving DataFrame with shape {df.shape} to {out_csv}")
        df.to_csv(out_csv, index=False)
        logger.info(f"Saved influence CSV: {out_csv}")

    # Also save metadata JSON
    meta = {
        "seed": seed,
        "type": infl_type,
        "relabel_percentage": relabel_percentage,
    }
    out_json = os.path.join(dn, f"infl_{infl_type}_{relabel_prefix}{seed:03d}.json")
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved influence metadata JSON: {out_json}")


def compute_adaptive_lambda(
    margins: np.ndarray, target_percentile: float = 90.0
) -> float:
    thr = np.percentile(np.abs(margins), target_percentile)
    return float(thr)


def compute_hvp_with_finite_diff(
    loss_fn,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    # Simple finite-difference Hessian-vector product (illustrative)
    with torch.no_grad():
        for p, vp in zip(model.parameters(), v):
            p.add_(epsilon * vp)
        loss_pos = loss_fn(model(x), y)
        for p, vp in zip(model.parameters(), v):
            p.add_(-2 * epsilon * vp)
        loss_neg = loss_fn(model(x), y)
        for p, vp in zip(model.parameters(), v):
            p.add_(epsilon * vp)
    hvp = (loss_pos - loss_neg) / (2 * epsilon)
    return hvp


def infl_diff_helper(
    seg_matrix: np.ndarray,
    lo: int,
    hi: int,
) -> np.ndarray:
    """Window-level leave-one-out ground truth from the per-epoch full-LOO trajectory.

    ``seg_matrix[e, i] = l_cf_i[e] - l_base[e]`` (the ``segment_true_full`` data):
    the cumulative full-LOO validation-loss effect of removing training sample
    ``i``, evaluated at epoch ``e``. The ground-truth influence over the window
    with epoch endpoints ``(lo, hi)`` is

        Q[i] = seg[hi, i] - (seg[lo, i] if lo >= 0 else 0),

    which is exactly the paper's Test-Loss window quantity
    ``[l_test(theta_{-i}[t2]) - l_test(theta_{-i}[t1])] -
      [l_test(theta[t2]) - l_test(theta[t1])]`` (Def. 2 / Sec. "Test Loss
    Influence"): differencing the full-LOO trajectory across the window's
    endpoints localizes the LOO effect to that window. ``lo = -1`` means the
    window starts before any training, where the LOO deviation is exactly zero,
    so ``seg[lo]`` is taken to be the zero vector.
    """
    num_points = int(seg_matrix.shape[0])
    if not (0 <= hi < num_points):
        raise ValueError(f"window end epoch index {hi} out of range [0, {num_points})")
    if lo >= hi:
        raise ValueError(
            f"empty LOO window: lo={lo} >= hi={hi} (need a positive-length window)"
        )
    if lo < -1 or lo >= num_points:
        raise ValueError(
            f"window start epoch index {lo} out of range [-1, {num_points})"
        )
    hi_v = np.asarray(seg_matrix[hi], dtype=np.float64)
    lo_v = (
        np.asarray(seg_matrix[lo], dtype=np.float64) if lo >= 0 else np.zeros_like(hi_v)
    )
    return hi_v - lo_v


class InfluenceCalculator(ABC):
    def __init__(
        self,
        infl_type: str,
        key: str,
        model_type: str,
        seed: int,
        gpu: int,
        save_dir: str | None = None,
        relabel_percentage: float | None = None,
        use_tensorboard: bool = False,
        length: int = 3,
        use_effective_lr: bool = False,
        query_type: str = "loss",
        query_index: int = 0,
        query_class: int | None = None,
        query_coord: int = 0,
        **_: Any,
    ) -> None:
        self.infl_type = infl_type
        self.key = key
        self.model_type = model_type
        self.seed = seed
        self.gpu = gpu
        self.relabel_percentage = relabel_percentage
        self.use_tensorboard = use_tensorboard
        self.length = length
        # Appendix-E checkpoint effective-lr heuristic (WIE window variants only).
        self.use_effective_lr = bool(use_effective_lr)
        # Extended query toolkit (WIE window variants only): loss (default),
        # prediction (a class logit), or saliency (an input-gradient component).
        self.query_type = query_type
        self.query_index = int(query_index)
        self.query_class = query_class
        self.query_coord = int(query_coord)
        # paths
        self.dn, self.fn_fallback, _ = get_file_paths(
            key, model_type, seed, infl_type, save_dir, relabel_percentage
        )
        # device
        self.device = get_device(gpu)
        # logging
        self.logger = logging.getLogger(self.__class__.__name__)
        # tb
        self.tb_writer = None
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard.writer import SummaryWriter

                self.tb_writer = SummaryWriter(log_dir=os.path.join(self.dn, "logs"))
            except Exception:
                self.logger.warning("TensorBoard not available; proceeding without it.")
        # common
        self.global_info = load_global_info(
            self.dn,
            self.seed,
            self.fn_fallback,
            self.device,
            self.logger,
            self.relabel_percentage,
        )
        self.x_tr, self.y_tr, self.x_val, self.y_val = load_data(
            self.key,
            self.global_info,
            self.seed,
            self.device,
            self.logger,
            self.relabel_percentage,
            self.dn,
        )
        self.input_dim = get_input_dim(self.x_tr, self.model_type)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        # Additional attributes for WieAllEpochs and other calculators
        self.n_tr = self.global_info["n_tr"]
        self.n_val = self.global_info["n_val"]
        self.n_test = self.global_info["n_test"]
        self.num_epoch = int(self.global_info.get("num_epoch", 0))
        self.alpha = self.global_info.get("alpha", 0.0)

        # Calculate steps_per_epoch if not provided
        self.steps_per_epoch = int(self.global_info.get("steps_per_epoch", 0))
        if self.steps_per_epoch <= 0:
            batch_size = self.global_info.get("batch_size", 1)
            if batch_size > 0:
                self.steps_per_epoch = max(
                    1, (self.n_tr + batch_size - 1) // batch_size
                )
                self.logger.debug(
                    f"Calculated steps_per_epoch: {self.steps_per_epoch} (n_tr={self.n_tr}, batch_size={batch_size})"
                )
            else:
                self.steps_per_epoch = 1
                self.logger.warning("batch_size <= 0; defaulting steps_per_epoch to 1")

        self.total_steps = int(self.global_info.get("total_steps", 0))
        if self.total_steps <= 0:
            self.total_steps = self.num_epoch * self.steps_per_epoch
            self.logger.debug(
                f"Calculated total_steps: {self.total_steps} (num_epoch={self.num_epoch}, steps_per_epoch={self.steps_per_epoch})"
            )

    @abstractmethod
    def _get_infl_type(self) -> str:
        pass

    # --- Default per-epoch delta calculation framework ---
    def _scores_for_model(self, model: torch.nn.Module) -> np.ndarray:
        """Return a single global influence vector I for the given model.

        Subclasses implement this to compute their method-specific influence
        scores for a provided model (treated as the final model).
        """
        raise NotImplementedError

    def _build_model_from_state(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> torch.nn.Module:
        input_dim = get_input_dim(self.x_tr, self.model_type)
        model = get_network(self.model_type, input_dim, logger=self.logger).to(
            self.device
        )
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                self.logger.warning(
                    f"Non-strict load_state (epoch model): missing={len(missing)}, unexpected={len(unexpected)}"
                )
        except Exception as e:
            self.logger.warning(
                f"Failed to load provided model state for epoch. Using initialized model. Error: {e}"
            )
        model.eval()
        return model

    def _scores_final_model(self) -> np.ndarray:
        """Fallback when per-epoch files are unavailable: use final model state."""
        try:
            state = load_final_model(self.dn, self.seed, self.device, self.logger)
            model = self._build_model_from_state(state)
            return self._scores_for_model(model)
        except Exception as e:
            self.logger.error(f"Failed to compute scores with final model: {e}")
            raise

    def calculate(self) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Default: produce per-epoch incremental scores for single-global methods.

        For epochs e=0..E-1:
          - Load epoch checkpoint e as the final model for that horizon
          - Compute I(e) using _scores_for_model(model_e)
          - Emit ΔI(e) = I(e) − I(e−1), with I(−1)=0

        If epoch checkpoints are not available, fall back to computing a single
        global vector from the final model.
        """
        num_epochs = int(self.global_info.get("num_epoch", 0))
        if num_epochs <= 0:
            self.logger.warning(
                "num_epoch <= 0; computing single global vector using final model."
            )
            return self._scores_final_model()

        prev_I: Optional[np.ndarray] = None
        delta_list: List[np.ndarray] = []

        for e in range(num_epochs):
            # Expose current epoch to subclasses that need horizon awareness
            try:
                self._current_epoch = e  # type: ignore[attr-defined]
                self._total_epochs = num_epochs  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                epoch_data = load_epoch_data(
                    self.dn, e, self.seed, self.relabel_percentage, self.logger
                )
                state_dict = epoch_data.get("model_state", None)
                if state_dict is None:
                    raise ValueError(f"Epoch file missing 'model_state' for epoch {e}.")
            except Exception as ex:
                self.logger.warning(
                    f"Failed to load epoch {e} checkpoint: {ex}. Falling back to final model."
                )
                # Fallback to single global vector and stop per-epoch accumulation
                return self._scores_final_model()

            model_e = self._build_model_from_state(state_dict)
            I_e = self._scores_for_model(model_e)

            if prev_I is None:
                delta = I_e.astype(np.float32)
            else:
                delta = (I_e - prev_I).astype(np.float32)
            delta_list.append(delta)
            prev_I = I_e

            del model_e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return delta_list

    def _output_infl_type(self) -> str:
        """infl_type used for the OUTPUT file/metadata identity.

        Defaults to ``self.infl_type``. Only calculators that actually honor a
        non-loss functional query override this (see
        ``_WieWindowInfluenceCalculator``), so a query flag accidentally passed to
        a calculator that ignores it never mislabels the output.
        """
        return self.infl_type

    def _save(self, infl_data):
        save_results(
            infl_data,
            self.dn,
            self.seed,
            self._output_infl_type(),
            self.logger,
            self.relabel_percentage,
        )
        if self.tb_writer:
            self.tb_writer.close()

    def run(self):
        self.logger.info(f"[{self.__class__.__name__}] Starting calculation...")
        self.logger.info(
            f"[DEBUG] Calculator config: n_tr={self.n_tr}, num_epoch={self.num_epoch}, device={self.device}"
        )
        try:
            result = self.calculate()
            self.logger.info(
                f"[{self.__class__.__name__}] Calculation finished. Saving results..."
            )

            # Debug logging for result validation
            self.logger.info(f"[DEBUG] Result type: {type(result)}")
            if isinstance(result, list):
                self.logger.info(f"[DEBUG] Result list length: {len(result)}")
                for i, item in enumerate(result[:3]):  # Log first 3 items
                    if hasattr(item, "shape"):
                        self.logger.info(
                            f"[DEBUG] Result[{i}] shape: {item.shape}, dtype: {getattr(item, 'dtype', 'unknown')}"
                        )
            elif hasattr(result, "shape"):
                self.logger.info(
                    f"[DEBUG] Result shape: {result.shape}, dtype: {getattr(result, 'dtype', 'unknown')}"
                )

            self._save(result)
            self.logger.info(
                f"[{self.__class__.__name__}] Results saved for type '{self.infl_type}'."
            )
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(
                f"[{self.__class__.__name__}] Calculation failed: {e}", exc_info=True
            )
            import traceback

            self.logger.error(f"[DEBUG] Full traceback:\n{traceback.format_exc()}")
            if self.tb_writer:
                self.tb_writer.close()
            raise


class InfluenceCalculatorFactory:
    _calculators: Dict[str, Type[InfluenceCalculator]] = {}

    # Legacy infl_type aliases: the team method was renamed tim_* -> wie_*.
    # Pre-existing configs/commands/output metadata that still say 'tim_*'
    # resolve to the current wie_* calculators so nothing breaks. New runs
    # write under the resolved (wie_*) name.
    # Only aliases whose canonical target is actually registered are kept.
    # wie_first/wie_middle are now implemented and registered, so their legacy
    # tim_first/tim_middle aliases are restored.
    INFL_TYPE_ALIASES: Dict[str, str] = {
        "tim_all_epochs": "wie_all_epochs",
        "tim_last": "wie_last",
        "tim_first": "wie_first",
        "tim_middle": "wie_middle",
    }

    @classmethod
    def _resolve(cls, infl_type: str) -> str:
        """Map a (possibly legacy tim_*) infl_type to its canonical key."""
        return cls.INFL_TYPE_ALIASES.get(infl_type, infl_type)

    @classmethod
    def register(cls, infl_type: str):
        def decorator(calculator_class: Type[InfluenceCalculator]):
            if not issubclass(calculator_class, InfluenceCalculator):
                raise TypeError(
                    f"{calculator_class.__name__} must inherit from InfluenceCalculator"
                )
            if infl_type in cls._calculators:
                logging.warning(
                    f"Overwriting registration for influence type '{infl_type}'"
                )
            cls._calculators[infl_type] = calculator_class
            logging.getLogger(__name__).debug(
                f"Registered influence calculator: {infl_type} -> {calculator_class.__name__}"
            )
            return calculator_class

        return decorator

    @classmethod
    def create(cls, infl_type: str, **kwargs) -> "InfluenceCalculator":
        resolved = cls._resolve(infl_type)
        calculator_class = cls._calculators.get(resolved)
        if not calculator_class:
            raise ValueError(
                f"Unknown influence type: '{infl_type}'. Available types: {list(cls._calculators.keys())}"
            )
        # Use the resolved (canonical) type so new outputs are written under the
        # current wie_* name even when invoked with a legacy tim_* alias.
        return calculator_class(infl_type=resolved, **kwargs)


class BaseDifferenceCalculator(InfluenceCalculator):
    """Window-level LOO ground truth by differencing the full-LOO trajectory.

    Concrete ``true_first``/``true_middle``/``true_last`` calculators supply the
    epoch window; this base loads the ``segment_true_full`` per-epoch full-LOO
    val-loss matrix and returns ``seg[hi] - seg[lo]`` (see :func:`infl_diff_helper`),
    the exact leave-one-out oracle the WIE window estimators are validated
    against (paper Table 1 RQ1-Local / Table 2 First/Mid/Last-vs-LOO).
    """

    @abstractmethod
    def get_source_prefix(self) -> str:
        """Prefix of the source per-epoch data (``"segment_true"``)."""

    @abstractmethod
    def get_window_epoch_bounds(self, num_epoch: int) -> Tuple[int, int]:
        """Return ``(lo, hi)`` epoch-boundary indices for this window.

        ``num_epoch`` is the trained epoch count (from training metadata, equal
        to the number of per-epoch entries in the full-LOO trajectory after the
        alignment check in :meth:`calculate`). ``hi`` is the window-end epoch
        index in ``[0, num_epoch)``; ``lo`` is the window-start boundary in
        ``[-1, hi)``, where ``lo = -1`` denotes "before any training" (zero
        deviation). Implementations must NOT clamp a non-positive ``length`` up
        to 1 -- ``calculate`` rejects ``length < 1`` before calling this.
        """

    def _segment_full_csv_path(self, source_prefix: str) -> str:
        relabel_prefix = (
            f"relabel_{int(self.relabel_percentage):03d}_pct_"
            if self.relabel_percentage is not None
            else ""
        )
        return os.path.join(
            self.dn, f"infl_{source_prefix}_full_{relabel_prefix}{self.seed:03d}.csv"
        )

    def _load_segment_full_matrix(self, source_prefix: str) -> np.ndarray:
        """Load ``segment_true_full`` CSV into a ``(num_epochs, n_tr)`` matrix.

        The CSV (written by :func:`save_results` for the per-epoch list) has a
        ``sample_idx`` column plus one ``influence_epoch_{e}`` column per epoch,
        each holding ``l_cf_i[e] - l_base[e]`` for every training sample ``i``.
        """
        path = self._segment_full_csv_path(source_prefix)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required source file {path} not found for '{self.infl_type}'. "
                f"Run the '{source_prefix}_full' calculator first (it needs the "
                "per-epoch counterfactual models from training with "
                "--compute_counterfactual)."
            )
        df = pd.read_csv(path)
        epoch_cols = [c for c in df.columns if c.startswith("influence_epoch_")]
        if not epoch_cols:
            raise ValueError(
                f"Source file {path} has no 'influence_epoch_*' columns; cannot "
                f"compute '{self.infl_type}' window difference."
            )
        epoch_cols.sort(key=lambda c: int(c.rsplit("_", 1)[1]))
        seg = np.stack([df[c].to_numpy(dtype=np.float64) for c in epoch_cols], axis=0)
        return seg

    def calculate(self) -> np.ndarray:
        # Reject non-positive window lengths (mirrors the wie_* window guard):
        # clamping to a 1-epoch oracle would silently answer a different, invalid
        # request. get_window_epoch_bounds must NOT clamp length up.
        if int(self.length) < 1:
            raise ValueError(
                f"{self.infl_type}: window length must be >= 1 epoch, got "
                f"{self.length}; refusing to emit a length-1 oracle for an "
                "invalid request."
            )
        source_prefix = self.get_source_prefix()
        seg = self._load_segment_full_matrix(source_prefix)
        num_points = int(seg.shape[0])
        # Define the window from the TRAINED epoch count (metadata), not merely
        # the number of rows the source CSV happens to have. Otherwise a
        # partial/early-stopped run (or a trajectory that includes an extra
        # initial-model row) would shift true_last/true_middle relative to the
        # wie_* window they must match. Require exact alignment (one row per
        # trained epoch) and fail loud on a mismatch rather than guessing.
        num_epoch = int(getattr(self, "num_epoch", 0) or 0)
        if num_epoch <= 0:
            num_epoch = num_points  # metadata unavailable: trust the source rows
        if num_epoch != num_points:
            raise ValueError(
                f"{self.infl_type}: '{source_prefix}_full' has {num_points} epoch "
                f"rows but training metadata records {num_epoch} epochs; cannot "
                "align the window to the trained trajectory. Regenerate "
                f"'{source_prefix}_full' for the full run before computing the "
                "window oracle."
            )
        lo, hi = self.get_window_epoch_bounds(num_epoch)
        self.logger.info(
            f"{self.infl_type}: window LOO ground truth over epoch endpoints "
            f"(lo={lo}, hi={hi}) of {num_epoch} trained epochs."
        )
        infl = infl_diff_helper(seg, lo, hi)
        save_results(
            infl,
            self.dn,
            self.seed,
            self.infl_type,
            self.logger,
            self.relabel_percentage,
        )
        return infl
