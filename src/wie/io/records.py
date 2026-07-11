"""
Record and metadata IO helpers for experiment runs.

This module centralizes logic to load global info and step/epoch
records. It mirrors the behavior used by the legacy experiment
implementation so callers can rely on consistent formats.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import torch

from . import make_relabel_prefix, resolve_epoch_file, resolve_step_file


def _safe_torch_load(
    path: str, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Safely load a torch file on CPU-only setups.

    If CUDA is unavailable, force map_location to CPU to avoid deserialization
    errors for tensors that were saved with CUDA device info.
    """
    try:
        if torch.cuda.is_available():
            return torch.load(path, weights_only=False)
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except Exception as e:
        if logger:
            logger.error(f"Failed to load torch file {path}: {e}")
        raise


def load_global_info(
    dn: str,
    seed: int,
    fn_fallback_dat: str,
    device: str,
    logger: logging.Logger,
    relabel_percentage: Optional[float] = None,
) -> Dict[str, Any]:
    """Load global training information for a run.

    Tries JSON first; if missing fields, falls back to legacy `.dat` file
    with best-effort reconstruction of required fields.
    """
    json_fn_base = f"global_info_{seed:03d}.json"
    json_fn_relabel = f"{make_relabel_prefix(relabel_percentage)}{json_fn_base}"

    json_paths_to_try = (
        [
            os.path.join(dn, json_fn_relabel),
            os.path.join(dn, json_fn_base),
        ]
        if relabel_percentage is not None
        else [os.path.join(dn, json_fn_base)]
    )

    for json_fn in json_paths_to_try:
        try:
            if os.path.exists(json_fn):
                with open(json_fn, "r") as f:
                    global_info = json.load(f)
                logger.info(
                    f"Successfully loaded global information from JSON: {json_fn}"
                )
                # Fill sensible defaults for missing optional keys
                if "alpha" not in global_info or global_info.get("alpha") is None:
                    global_info["alpha"] = 0.0
                if "lr" not in global_info or global_info.get("lr") is None:
                    global_info["lr"] = 0.0
                if (
                    "decay" not in global_info
                    and global_info.get("weight_decay") is not None
                ):
                    global_info["decay"] = global_info.get("weight_decay")
                return global_info
        except Exception as e:
            logger.warning(
                f"Error reading JSON {json_fn}: {e}. Will try fallback if needed."
            )

    logger.warning("Global info JSON not found. Attempting to use fallback .dat file.")

    # Fallback path (.dat)
    try:
        res = torch.load(fn_fallback_dat, map_location=device, weights_only=False)
        # Compose a dict with the required fields from res
        global_info = {
            "n_tr": res.get("n_tr"),
            "n_val": res.get("n_val"),
            "n_test": res.get("n_test"),
            "num_epoch": res.get("num_epoch"),
            "batch_size": res.get("batch_size"),
            "lr": res.get("lr"),
            "decay": res.get("decay", res.get("weight_decay")),
            "alpha": res.get("alpha"),
        }
        required_keys = [
            "n_tr",
            "n_val",
            "n_test",
            "num_epoch",
            "batch_size",
            "lr",
            "alpha",
        ]
        missing_keys = [k for k in required_keys if global_info.get(k) is None]
        if missing_keys:
            logger.error(
                f"Fallback file {fn_fallback_dat} is missing required keys for global_info: {missing_keys}"
            )
            raise ValueError(
                f"Fallback file {fn_fallback_dat} missing keys: {missing_keys}"
            )
        if global_info.get("decay") is None:
            global_info["decay"] = 0.0
            logger.info("Assuming 'decay'=0.0 as it was not found in fallback .dat.")
        logger.info(
            f"Successfully reconstructed global information from fallback file: {fn_fallback_dat}"
        )
        return global_info
    except FileNotFoundError:
        logger.error(f"Fallback file {fn_fallback_dat} also not found.")
        raise FileNotFoundError(
            f"Could not load global training information from JSON or fallback {fn_fallback_dat}"
        )
    except Exception as e:
        logger.error(f"Error loading or parsing fallback file {fn_fallback_dat}: {e}")
        raise


def load_epoch_data(
    dn: str,
    epoch: int,
    seed: int,
    relabel_percentage: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Load a saved epoch record file, honoring relabel prefix fallbacks."""
    path = resolve_epoch_file(dn, seed, epoch, relabel_percentage)
    if logger:
        logger.debug(f"Attempting to load epoch file: {path}")
    return _safe_torch_load(path, logger)


def load_step_data(
    dn: str,
    step: int,
    seed: int,
    relabel_percentage: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Load a saved step record file, with epoch-format fallback conversion."""
    path = resolve_step_file(dn, seed, step, relabel_percentage)
    if logger:
        logger.debug(f"Attempting to load step file: {path}")

    data = _safe_torch_load(path, logger)

    # Handle epoch file format - extract step info from step_info list
    if "step_info" in data and "idx" not in data:
        step_info = data["step_info"]
        if isinstance(step_info, list) and len(step_info) > 0:
            target_global_step = max(step - 1, 0)
            chosen = None
            for entry in step_info:
                if not isinstance(entry, dict):
                    continue
                if entry.get("global_step") == target_global_step:
                    chosen = entry
                    break

            if chosen is None:
                chosen = step_info[-1]
                if logger:
                    logger.debug(
                        "Step %s: target global_step=%s not found; using fallback with global_step=%s",
                        step,
                        target_global_step,
                        chosen.get("global_step"),
                    )

            result = {
                "idx": chosen.get("idx", []),
                "lr": chosen.get("lr", 0.0),
                "model_state": data.get("model_state", {}),
                "step_loss": chosen.get("step_loss", 0.0),
                "global_step": chosen.get("global_step", step),
            }
            if logger:
                logger.debug(
                    "Converted epoch file format to step format for step %s via fallback",
                    step,
                )
            return result

    return data
