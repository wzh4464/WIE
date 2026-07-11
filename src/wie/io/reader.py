import json
import math
import os
from .naming import make_relabel_prefix


def _infer_steps_per_epoch(dn: str, seed: int) -> int:
    """Best-effort retrieval of steps_per_epoch for fallback step resolution."""

    info_path = os.path.join(dn, f"global_info_{seed:03d}.json")
    try:
        with open(info_path, "r") as f:
            info = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return 1

    steps_per_epoch = info.get("steps_per_epoch")
    if isinstance(steps_per_epoch, int) and steps_per_epoch > 0:
        return steps_per_epoch

    n_tr = info.get("n_tr")
    batch_size = info.get("batch_size")
    if isinstance(batch_size, int) and batch_size > 0:
        pass
    else:
        training_params = info.get("training_params") or {}
        batch_size = training_params.get("batch_size")

    if isinstance(n_tr, int) and isinstance(batch_size, int) and batch_size > 0:
        return max(1, math.ceil(n_tr / batch_size))

    return 1


def resolve_epoch_file(
    dn: str, seed: int, epoch: int, relabel_percentage: float | int | None = None
):
    """Return the epoch file path, applying the same fallback policy as infl.py.
    Does not load the file; only resolves the path that exists.
    """
    records_dir = os.path.join(dn, "records")
    prefix = make_relabel_prefix(relabel_percentage)
    cand = os.path.join(records_dir, f"{prefix}epoch_{epoch}_{seed:03d}.pt")
    if os.path.exists(cand):
        return cand
    # fallback without relabel prefix
    cand2 = os.path.join(records_dir, f"epoch_{epoch}_{seed:03d}.pt")
    if os.path.exists(cand2):
        return cand2
    raise FileNotFoundError(cand)


def resolve_step_file(
    dn: str, seed: int, step: int, relabel_percentage: float | int | None = None
):
    records_dir = os.path.join(dn, "records")
    prefix = make_relabel_prefix(relabel_percentage)
    cand = os.path.join(records_dir, f"{prefix}step_{step}_{seed:03d}.pt")
    if os.path.exists(cand):
        return cand
    # fallback without relabel prefix
    cand2 = os.path.join(records_dir, f"step_{step}_{seed:03d}.pt")
    if os.path.exists(cand2):
        return cand2

    # fallback to epoch files when step files are missing
    # This handles cases where training saves epoch-level checkpoints instead of step-level
    try:
        steps_per_epoch = _infer_steps_per_epoch(dn, seed)
        if steps_per_epoch <= 0:
            steps_per_epoch = 1
        # Step indices used by influence calculators are 1-based.
        target_step = max(step - 1, 0)
        epoch_index = target_step // steps_per_epoch

        epoch_candidates = [
            os.path.join(records_dir, f"{prefix}epoch_{epoch_index}_{seed:03d}.pt"),
            os.path.join(records_dir, f"epoch_{epoch_index}_{seed:03d}.pt"),
        ]
        for candidate in epoch_candidates:
            if os.path.exists(candidate):
                return candidate
    except Exception:
        pass

    raise FileNotFoundError(cand)
