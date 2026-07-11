"""
Path helpers for experiment artifacts.

Provides a generalized get_file_paths implementation that callers can use
by supplying the script_dir where experiment artifacts should live.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from .naming import make_base_dir, make_relabel_prefix


def get_file_paths_general(
    script_dir: str,
    key: str,
    model_type: str,
    seed: int,
    infl_type: Optional[str] = None,
    save_dir: Optional[str] = None,
    relabel_percentage: Optional[float] = None,
) -> Tuple[str, str, str]:
    """Generalized file path resolver for experiment outputs.

    Returns a tuple of (base_dir, main_results_path, influence_results_path).
    - base_dir contains all artifacts for a given (key, model_type[, save_dir]).
    - main_results_path defaults to a `.dat` file with optional relabel prefix.
    - influence_results_path points to the `.dat` file for specified infl_type.
    """
    base_dir = make_base_dir(script_dir, save_dir, key, model_type)
    relabel_prefix = make_relabel_prefix(relabel_percentage)
    seed_suffix = f"{seed:03d}.dat"

    # Fallback/Main results file (.dat)
    sgd_prefix = "sgd_" if relabel_percentage is None else ""
    main_path = os.path.join(base_dir, f"{relabel_prefix}{sgd_prefix}{seed_suffix}")

    # Influence output file path (.dat)
    infl_path = ""
    if infl_type:
        if infl_type == "lie_full":
            infl_path = os.path.join(
                base_dir, f"infl_lie_full_{relabel_prefix}{seed_suffix}"
            )
        elif infl_type == "segment_true_full":
            infl_path = os.path.join(
                base_dir, f"infl_segment_true_full_{relabel_prefix}{seed_suffix}"
            )
        else:
            infl_path = os.path.join(
                base_dir, f"infl_{infl_type}_{relabel_prefix}{seed_suffix}"
            )

    return base_dir, main_path, infl_path
