import glob as _glob
import os

from wie.utils.paths import get_project_root


# The team method was renamed tim_* -> wie_*. Historical outputs on disk keep
# their legacy tim_* filenames/dirs; new runs write wie_*. These helpers let
# readers (analysis, plotting, cleansing, run-index) resolve BOTH names without
# renaming anything under outputs/, result/, or data/.
# Only canonical keys that are actually registered are aliased. wie_first/
# wie_middle are now implemented and registered, so their legacy tim_* filename
# aliases are included.
INFL_TYPE_LEGACY_ALIASES = {
    "wie_all_epochs": "tim_all_epochs",
    "wie_last": "tim_last",
    "wie_first": "tim_first",
    "wie_middle": "tim_middle",
}

# Inverse map (legacy tim_* -> canonical wie_*). A caller may pass EITHER the
# canonical or the legacy key; this lets us normalize a legacy key to its
# canonical name so reads always try the wie_* filename first.
INFL_TYPE_CANONICAL_ALIASES = {
    legacy: canonical for canonical, legacy in INFL_TYPE_LEGACY_ALIASES.items()
}


def infl_type_read_aliases(infl_type: str) -> list[str]:
    """Return candidate infl_type tokens to try when READING outputs.

    Bidirectional: whether given a canonical (wie_*) OR a legacy (tim_*) key,
    normalize to the canonical name first and then append the legacy alias, so
    new wie_* outputs win but pre-existing tim_* files remain discoverable.

    Examples:
        infl_type_read_aliases("wie_all_epochs") -> ["wie_all_epochs", "tim_all_epochs"]
        infl_type_read_aliases("tim_all_epochs") -> ["wie_all_epochs", "tim_all_epochs"]
    """
    # Normalize legacy tim_* input to its canonical wie_* name first.
    canonical = INFL_TYPE_CANONICAL_ALIASES.get(infl_type, infl_type)
    aliases = [canonical]
    legacy = INFL_TYPE_LEGACY_ALIASES.get(canonical)
    if legacy is not None and legacy not in aliases:
        aliases.append(legacy)
    return aliases


def infl_csv_candidates(
    dir_name: str,
    infl_type: str,
    seed: int,
    relabel_percentage: float | int | None = None,
) -> list[str]:
    """Return candidate ``infl_{type}_{...}.csv`` paths (canonical then legacy)."""
    prefix = make_relabel_prefix(relabel_percentage)
    return [
        os.path.join(dir_name, f"infl_{t}_{prefix}{seed:03d}.csv")
        for t in infl_type_read_aliases(infl_type)
    ]


def resolve_infl_csv(
    dir_name: str,
    infl_type: str,
    seed: int,
    relabel_percentage: float | int | None = None,
) -> str:
    """Return the first existing infl CSV (canonical wie_*, else legacy tim_*).

    Falls back to the canonical path if neither exists (so callers get a sane
    default to report/write against).
    """
    candidates = infl_csv_candidates(dir_name, infl_type, seed, relabel_percentage)
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def infl_glob_patterns(infl_type: str) -> list[str]:
    """Return ``infl_{type}_*.csv`` glob patterns for both canonical and legacy."""
    return [f"infl_{t}_*.csv" for t in infl_type_read_aliases(infl_type)]


def glob_infl_csvs(dir_name: str, infl_type: str) -> list[str]:
    """Glob a directory for infl CSVs matching the canonical or legacy prefix."""
    found: list[str] = []
    for pattern in infl_glob_patterns(infl_type):
        found.extend(_glob.glob(os.path.join(dir_name, pattern)))
    return sorted(set(found))


def make_relabel_prefix(relabel_percentage: float | int | None) -> str:
    if relabel_percentage is None:
        return ""
    try:
        val = int(relabel_percentage)
        return f"relabel_{val:03d}_pct_"
    except Exception:
        return ""


def make_base_dir(
    script_dir: str, save_dir: str | None, key: str, model_type: str
) -> str:
    """Resolve base output directory.

    Policy:
    - Absolute save_dir: use as-is.
    - Relative save_dir: interpret relative to REPO_ROOT/outputs/.
    - No save_dir: default to REPO_ROOT/outputs/{key}_{model_type}.

    REPO_ROOT is derived from the installed package location via
    wie.utils.paths.get_project_root() (matching resolve_output_dir/
    resolve_data_dir), not inferred from script_dir.
    """
    # script_dir retained for backward compat; the outputs root is now sourced
    # from the centralized project-root helper instead of the script location.
    outputs_root = os.path.join(str(get_project_root()), "outputs")

    if save_dir:
        if os.path.isabs(save_dir):
            base_dir = save_dir
        else:
            base_dir = os.path.join(outputs_root, save_dir)
    else:
        base_dir = os.path.join(outputs_root, f"{key}_{model_type}")

    os.makedirs(base_dir, exist_ok=True)
    return base_dir
