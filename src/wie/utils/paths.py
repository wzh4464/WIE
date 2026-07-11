"""Path standardization utilities for the project.

This module provides functions to standardize paths across the project:
- All output paths are relative to ./outputs/
- All data paths are relative to ./data/
"""

from pathlib import Path
from typing import Union, Optional


def get_project_root() -> Path:
    """Get the project root directory (sgd-influence)."""
    current_file = Path(__file__).resolve()
    # Navigate from src/experiment/utils/ to project root
    return current_file.parent.parent.parent.parent


def resolve_output_dir(save_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve output directory to be under ./outputs/.

    Args:
        save_dir: Directory path to resolve. If None, returns ./outputs/

    Returns:
        Resolved path under ./outputs/
    """
    project_root = get_project_root()
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if save_dir is None:
        return outputs_dir

    # Convert to Path object
    save_dir = Path(save_dir)

    # If already under outputs/, return as is
    if save_dir.is_absolute():
        try:
            save_dir.relative_to(outputs_dir)
            return save_dir
        except ValueError:
            # Not under outputs/, use basename
            save_dir = save_dir.name

    # Resolve relative to outputs/
    return outputs_dir / save_dir


def resolve_data_dir(data_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve data directory to be under ./data/.

    Args:
        data_dir: Directory path to resolve. If None, returns ./data/

    Returns:
        Resolved path under ./data/
    """
    project_root = get_project_root()
    base_data_dir = project_root / "data"
    base_data_dir.mkdir(parents=True, exist_ok=True)

    if data_dir is None:
        return base_data_dir

    # Convert to Path object
    data_dir = Path(data_dir)

    # If already under data/, return as is
    if data_dir.is_absolute():
        try:
            data_dir.relative_to(base_data_dir)
            return data_dir
        except ValueError:
            # Not under data/, use basename
            data_dir = data_dir.name

    # Resolve relative to data/
    return base_data_dir / data_dir
