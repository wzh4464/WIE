"""
IO helpers for experiment pipelines under src/.

This package centralizes naming, path resolution, and record readers.
Legacy callers using `experiment.io` can continue to work via a thin shim.
"""

from .naming import (  # noqa: F401
    make_relabel_prefix,
    make_base_dir,
    INFL_TYPE_LEGACY_ALIASES,
    infl_type_read_aliases,
    infl_csv_candidates,
    resolve_infl_csv,
    infl_glob_patterns,
    glob_infl_csvs,
)
from .reader import resolve_epoch_file, resolve_step_file  # noqa: F401
from .records import (  # noqa: F401
    load_global_info,
    load_epoch_data,
    load_step_data,
)
from .paths import get_file_paths_general  # noqa: F401

__all__ = [
    "make_relabel_prefix",
    "make_base_dir",
    "INFL_TYPE_LEGACY_ALIASES",
    "infl_type_read_aliases",
    "infl_csv_candidates",
    "resolve_infl_csv",
    "infl_glob_patterns",
    "glob_infl_csvs",
    "resolve_epoch_file",
    "resolve_step_file",
    "load_global_info",
    "load_epoch_data",
    "load_step_data",
    "get_file_paths_general",
]
