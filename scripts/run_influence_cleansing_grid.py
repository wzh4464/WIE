"""run_influence_cleansing_grid.py
=====================================
Grid search script for influence-based data cleansing experiments across multiple GPUs.

Features
--------
* Builds a Cartesian product over influence methods, keep ratios, and relabel percentages.
* Distributes runs evenly across the supplied GPU ids and executes them sequentially per GPU.
* Supports both training + influence + cleansing pipeline and cleansing-only mode.
* Automatically analyzes precision performance across all methods after completion.
* Captures stdout/stderr for each run into the experiment save_dir log.
* Parameter lists may be provided via command-line options or a JSON "grid file".

Usage example::

    python scripts/run_influence_cleansing_grid.py \
        --dry-run \
        --output-root outputs \
        --save-dir-prefix sentiment_cleansing \
        --grid-file configs/cleansing_grid.json

Create a JSON grid description with keys "methods", "keep_ratios", and "relabel_percentages"::

    {
      "methods": ["wie_all_epochs", "icml_all_epochs", "lava_all_epochs"],
      "keep_ratios": [70, 80, 90],
      "relabel_percentages": [5, 10, 15, 20]
    }
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import os
import sys
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

# Make the src/ layout importable when this script is run by path
# (python scripts/foo.py) without the editable install (pixi run install).
try:
    import wie  # noqa: F401
except ModuleNotFoundError:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from wie.utils.paths import resolve_output_dir
from wie.io.naming import infl_type_read_aliases, infl_csv_candidates

# Default parameter grids
DEFAULT_METHODS = ["wie_all_epochs", "icml_all_epochs", "lava_all_epochs", "dve_all_epochs"]
DEFAULT_KEEP_RATIOS = [70, 80, 90, 95]
DEFAULT_RELABEL_PERCENTAGES = [5, 10, 15, 20]
DEFAULT_SEEDS = [0, 1, 2]


@dataclass
class CleansingTask:
    """Single cleansing experiment metadata for a scheduled run."""

    command: List[str]
    log_path: Path
    method: str
    keep_ratio: int
    relabel_pct: int
    seed: int
    save_dir: str


def parse_args() -> argparse.Namespace:
    """Configure and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate and optionally run a grid of influence cleansing experiments, "
            "distributing runs across the requested GPUs."
        )
    )

    # Grid parameters
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Influence methods to sweep. "
            f"Default: {', '.join(DEFAULT_METHODS)}."
        ),
    )
    parser.add_argument(
        "--keep-ratios",
        nargs="+",
        type=int,
        dest="keep_ratios",
        default=None,
        help=(
            "Keep ratio percentages to sweep. "
            f"Default: {', '.join(map(str, DEFAULT_KEEP_RATIOS))}."
        ),
    )
    parser.add_argument(
        "--relabel-percentages",
        nargs="+",
        type=int,
        dest="relabel_percentages",
        default=None,
        help=(
            "Relabel percentages to sweep. "
            f"Default: {', '.join(map(str, DEFAULT_RELABEL_PERCENTAGES))}."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Random seeds to sweep. "
            f"Default: {', '.join(map(str, DEFAULT_SEEDS))}."
        ),
    )

    # Grid file
    parser.add_argument(
        "--grid-file",
        type=Path,
        help=(
            "Optional JSON file containing keys 'methods', 'keep_ratios', 'relabel_percentages', 'seeds'. "
            "Values from the command line take precedence."
        ),
    )

    # System configuration
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["0", "1", "2"],
        help="GPU ids used for scheduling (each GPU runs its queue sequentially).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional directory prepended to each run's save_dir. NOTE: paths are "
            "resolved under the centralized outputs/ directory (resolve_output_dir "
            "convention), so e.g. --output-root result writes to -- and the "
            "follow-up analysis reads from -- outputs/result/... . Truly escaping "
            "outputs/ would require changing resolve_output_dir (out of scope). If "
            "omitted, runs land directly under outputs/."
        ),
    )
    parser.add_argument(
        "--save-dir-prefix",
        default="influence_cleansing",
        help="Prefix used when building each run's save_dir (parameters are appended).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated commands without executing them.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help=(
            "Optional cap on the number of GPU worker threads. Defaults to len(--gpus). "
            "Ignored when --dry-run is set."
        ),
    )

    # Experiment configuration
    parser.add_argument(
        "--target",
        default="sentiment",
        help="Target dataset for experiments.",
    )
    parser.add_argument(
        "--model",
        default="bert",
        help="Model identifier for experiments.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str.upper,  # normalize case: exp_influence_cleansing only accepts UPPER
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for experiments (case-insensitive; normalized to UPPER).",
    )
    # Training hyperparameter pass-throughs. When set, these are forwarded to the
    # training subprocess (wie.training.train in grouped mode, or
    # epoch_wise_keep_ratio.py in the non-grouped path). lr is also forwarded to
    # the cleansing retraining step. Left unset -> the subprocess defaults apply.
    parser.add_argument(
        "--num-epoch",
        dest="num_epoch",
        type=int,
        default=None,
        help="Number of training epochs (forwarded to the training subprocess).",
    )
    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        default=None,
        help="Learning rate (forwarded to training and cleansing subprocesses).",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help="Batch size (forwarded to the training subprocess).",
    )

    # Pipeline control
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training step and use existing training data.",
    )
    parser.add_argument(
        "--existing-train-dir",
        type=str,
        help="Path to existing training data directory (required when --skip-training is used).",
    )
    parser.add_argument(
        "--compute-precision",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Compute precision/recall/F1 statistics (default: True).",
    )
    parser.add_argument(
        "--compute-retraining-loss",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Perform actual retraining and compute losses (default: True).",
    )
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="Run precision analysis after all experiments complete.",
    )
    parser.add_argument(
        "--epochs-only",
        action="store_true",
        help="Only run experiments for all epochs (skips single-epoch if applicable).",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip cleanup of model checkpoint files (.pt) after each experiment completion.",
    )
    parser.add_argument(
        "--share-training",
        action="store_true",
        default=True,
        help="Share training data across methods (train once per relabel/seed). Default: True.",
    )
    parser.add_argument(
        "--no-share-training",
        action="store_true",
        help="Disable training sharing (run full pipeline for each method separately).",
    )

    return parser.parse_args()


def load_grid_from_file(path: Path) -> Dict[str, List]:
    """Load grid specification from disk."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse grid file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Grid file {path} must contain a JSON object.")

    grid: Dict[str, List] = {}

    # Load methods
    if "methods" in data:
        if not isinstance(data["methods"], list):
            raise ValueError("Grid entry 'methods' must be a list.")
        grid["methods"] = [str(item) for item in data["methods"]]

    # Load keep_ratios
    if "keep_ratios" in data:
        if not isinstance(data["keep_ratios"], list):
            raise ValueError("Grid entry 'keep_ratios' must be a list.")
        grid["keep_ratios"] = [int(item) for item in data["keep_ratios"]]

    # Load relabel_percentages
    if "relabel_percentages" in data:
        if not isinstance(data["relabel_percentages"], list):
            raise ValueError("Grid entry 'relabel_percentages' must be a list.")
        grid["relabel_percentages"] = [int(item) for item in data["relabel_percentages"]]

    # Load seeds
    if "seeds" in data:
        if not isinstance(data["seeds"], list):
            raise ValueError("Grid entry 'seeds' must be a list.")
        grid["seeds"] = [int(item) for item in data["seeds"]]

    # Load epochs_only flag
    if "epochs_only" in data:
        if not isinstance(data["epochs_only"], bool):
            raise ValueError("Grid entry 'epochs_only' must be a boolean.")
        grid["epochs_only"] = data["epochs_only"]

    # Load no_loo flag
    if "no_loo" in data:
        if not isinstance(data["no_loo"], bool):
            raise ValueError("Grid entry 'no_loo' must be a boolean.")
        grid["no_loo"] = data["no_loo"]

    return grid


def merge_grid_args(args: argparse.Namespace) -> Tuple[List[str], List[int], List[int], List[int], bool, bool]:
    """Combine CLI values with the optional grid file."""
    file_grid: Dict[str, List] = {}
    if args.grid_file:
        if not args.grid_file.exists():
            raise FileNotFoundError(f"Grid file {args.grid_file} does not exist.")
        file_grid = load_grid_from_file(args.grid_file)

    methods = args.methods or file_grid.get("methods") or DEFAULT_METHODS
    keep_ratios = args.keep_ratios or file_grid.get("keep_ratios") or DEFAULT_KEEP_RATIOS
    relabel_percentages = args.relabel_percentages or file_grid.get("relabel_percentages") or DEFAULT_RELABEL_PERCENTAGES
    seeds = args.seeds or file_grid.get("seeds") or DEFAULT_SEEDS
    epochs_only = args.epochs_only or file_grid.get("epochs_only", False)  # CLI flag or grid file
    no_loo = file_grid.get("no_loo", False)  # Default to False if not specified

    return (
        ensure_string_values("methods", methods),
        ensure_int_values("keep_ratios", keep_ratios),
        ensure_int_values("relabel_percentages", relabel_percentages),
        ensure_int_values("seeds", seeds),
        epochs_only,
        no_loo,
    )


def ensure_string_values(name: str, values: Sequence) -> List[str]:
    """Validate that the provided parameter list is non-empty and convert to strings."""
    cleaned = [str(v).strip() for v in values if str(v).strip() != ""]
    if not cleaned:
        raise ValueError(f"No values provided for {name}.")
    return cleaned


def ensure_int_values(name: str, values: Sequence) -> List[int]:
    """Validate that the provided parameter list is non-empty and convert to ints."""
    try:
        cleaned = [int(v) for v in values if v is not None]
        if not cleaned:
            raise ValueError(f"No values provided for {name}.")
        return cleaned
    except (ValueError, TypeError) as e:
        raise ValueError(f"All values for {name} must be integers: {e}")


# Influence methods that require leave-one-out / counterfactual retraining.
# ONLY methods that actually READ counterfactual (leave-one-out) models need it:
# loo_all_epochs, the ground-truth true*/segment_true_full methods. Everything
# else (incl. icml_all_epochs which reads step_*.pt, and lava_all_epochs which
# reads normal epoch_*.pt checkpoints) trains with --no-loo, which is orders of
# magnitude faster. The true*/loo/segment_true_full clauses in
# method_needs_counterfactual cover the rest, so this set is just loo_all_epochs.
COUNTERFACTUAL_METHODS = {"loo_all_epochs"}


def method_needs_counterfactual(method: str) -> bool:
    """Return True if ``method`` needs counterfactual/LOO retraining."""
    return (
        method in COUNTERFACTUAL_METHODS
        or method == "loo"
        or method.startswith("true")
        or method == "segment_true_full"
    )


def build_save_dir(
    prefix: str,
    target: str,
    model: str,
    method: str,
    keep_ratio: int,
    relabel_pct: int,
    seed: int,
    output_root: Path | None
) -> str:
    """Construct the resolved save directory for a given parameter combination.

    Honors ``--output-root`` (prefixed when provided) and returns the fully
    resolved directory under the centralized ``outputs/`` tree, so every stage
    epoch_wise_keep_ratio.py launches agrees on one path. The run name includes
    ``{target}_{model}`` (matching grouped mode) so different datasets/models
    never share a dir.
    """
    method_short = method.replace("_all_epochs", "").replace("_", "")
    tokens = (
        f"{target}",
        f"{model}",
        f"{method_short}",
        f"keep{keep_ratio}",
        f"relabel{relabel_pct:02d}",
        f"seed{seed:02d}",
    )
    run_name = f"{prefix}_{'_'.join(tokens)}"
    # Resolve the --output-root FIRST, then append the run name (same as grouped
    # mode), so every stage epoch_wise_keep_ratio.py launches agrees on ONE
    # directory -- and it matches how the analyzer resolves its base dir -- even
    # for an absolute --output-root outside the repo.
    return os.path.join(resolved_output_root(output_root), run_name)


def build_command(
    method: str,
    keep_ratio: int,
    relabel_pct: int,
    seed: int,
    gpu: str,
    save_dir: str,
    args: argparse.Namespace,
    epochs_only: bool = False,
    no_loo: bool = False,
) -> List[str]:
    """Assemble the epoch_wise_keep_ratio.py command for a single experiment."""
    command = [
        "python",
        "scripts/epoch_wise_keep_ratio.py",
        "--target", args.target,
        "--model", args.model,
        "--save_dir", save_dir,
        "--relabel", str(relabel_pct),
        "--seed", str(seed),
        "--keep_ratio", str(keep_ratio),
        "--type", method,
        "--gpu", gpu,
        "--log_level", args.log_level,
        "--compute_precision", args.compute_precision,
        "--compute_retraining_loss", args.compute_retraining_loss,
    ]

    # Add skip training options if specified
    if args.skip_training:
        command.append("--skip_train")
        if args.existing_train_dir:
            command.extend(["--existing_train_dir", args.existing_train_dir])

    # Add epochs_only flag if specified
    if epochs_only:
        command.append("--epochs_only")

    # Add no_loo flag if specified
    if no_loo:
        command.append("--no-loo")

    # Forward training hyperparameters when provided (epoch_wise_keep_ratio.py
    # accepts --num_epoch/--lr/--batch_size and threads them to its subprocesses).
    if args.num_epoch is not None:
        command.extend(["--num_epoch", str(args.num_epoch)])
    if args.lr is not None:
        command.extend(["--lr", str(args.lr)])
    if args.batch_size is not None:
        command.extend(["--batch_size", str(args.batch_size)])

    # Forward no-cleanup flag (epoch_wise_keep_ratio.py accepts --no-cleanup)
    if args.no_cleanup:
        command.append("--no-cleanup")

    return command


def generate_task_map(
    gpus: Sequence[str],
    methods: Sequence[str],
    keep_ratios: Sequence[int],
    relabel_percentages: Sequence[int],
    seeds: Sequence[int],
    output_root: Path | None,
    prefix: str,
    args: argparse.Namespace,
    epochs_only: bool = False,
    no_loo: bool = False,
) -> Dict[str, List[CleansingTask]]:
    """Return a GPU-to-command mapping with evenly distributed runs."""
    tasks: Dict[str, List[CleansingTask]] = {gpu: [] for gpu in gpus}

    # Generate all combinations
    combos = list(itertools.product(methods, keep_ratios, relabel_percentages, seeds))

    print(f"Generated {len(combos)} experiment combinations:")
    print(f"  Methods: {len(methods)} ({', '.join(methods)})")
    print(f"  Keep ratios: {len(keep_ratios)} ({', '.join(map(str, keep_ratios))})")
    print(f"  Relabel percentages: {len(relabel_percentages)} ({', '.join(map(str, relabel_percentages))})")
    print(f"  Seeds: {len(seeds)} ({', '.join(map(str, seeds))})")
    print(f"  Total experiments: {len(combos)}")
    print(f"  GPUs: {len(gpus)} ({', '.join(gpus)})")
    print()

    for idx, (method, keep_ratio, relabel_pct, seed) in enumerate(combos):
        gpu = gpus[idx % len(gpus)]
        save_dir = build_save_dir(
            prefix, args.target, args.model, method, keep_ratio, relabel_pct, seed, output_root
        )
        command = build_command(method, keep_ratio, relabel_pct, seed, gpu, save_dir, args, epochs_only, no_loo)
        log_path = Path(save_dir) / "cleansing_experiment.log"

        task = CleansingTask(
            command=command,
            log_path=log_path,
            method=method,
            keep_ratio=keep_ratio,
            relabel_pct=relabel_pct,
            seed=seed,
            save_dir=save_dir
        )
        tasks[gpu].append(task)

    return tasks


def run_command_queue(gpu: str, tasks: Sequence[CleansingTask]) -> None:
    """Execute a sequence of commands on a single GPU sequentially."""
    print(f"[GPU {gpu}] Starting {len(tasks)} experiments")

    for i, task in enumerate(tasks, 1):
        command_str = " ".join(task.command)
        print(f"[GPU {gpu}] Experiment {i}/{len(tasks)}: {task.method}, keep_ratio={task.keep_ratio}, relabel={task.relabel_pct}, seed={task.seed}")
        print(f"[GPU {gpu}] Running: {command_str}")
        print(f"[GPU {gpu}] Log: {task.log_path}")

        # Create log directory
        task.log_path.parent.mkdir(parents=True, exist_ok=True)

        with task.log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"# GPU: {gpu}\n")
            log_file.write(f"# Method: {task.method}\n")
            log_file.write(f"# Keep Ratio: {task.keep_ratio}%\n")
            log_file.write(f"# Relabel Percentage: {task.relabel_pct}%\n")
            log_file.write(f"# Seed: {task.seed}\n")
            log_file.write(f"# Save Dir: {task.save_dir}\n")
            log_file.write(f"# Command: {command_str}\n\n")
            log_file.flush()

            try:
                subprocess.run(
                    task.command, check=True, stdout=log_file, stderr=subprocess.STDOUT
                )
                print(f"[GPU {gpu}] ✅ Experiment {i}/{len(tasks)} completed successfully")
            except subprocess.CalledProcessError as e:
                print(f"[GPU {gpu}] ❌ Experiment {i}/{len(tasks)} failed; see {task.log_path}")
                log_file.write(f"\n\nEXPERIMENT FAILED WITH EXIT CODE: {e.returncode}\n")
                raise

    print(f"[GPU {gpu}] ✅ All {len(tasks)} experiments completed")


def cleanup_pt_files(save_dir: str) -> None:
    """Reclaim disk by deleting per-step trajectory checkpoints (``step_*.pt``).

    Only ``step_*.pt`` is removed. Those per-step snapshots are consumed solely
    while computing influence (wie/icml/lie/nohess) and read by nothing after —
    cleansing retrains from ``init_*.pt``, valuation/analysis read CSVs plus
    ``epoch_final_*``/``counterfactual_*``. Everything else in ``records/``
    (init/epoch/epoch_final/counterfactual and ``dve_raw/``) is preserved
    because it is still referenced by cleansing, valuation, or DVE. Runs after
    influence in grouped mode, so the snapshots are already spent.
    """
    resolved_dir = str(resolve_output_dir(save_dir))
    if not os.path.exists(resolved_dir):
        return

    deleted_count = 0
    freed_space = 0
    for root, dirs, files in os.walk(resolved_dir):
        for f in files:
            # step_NNNNNN.pt only; excludes dve_step_*/counterfactual_*/epoch_*/init_*
            if f.startswith("step_") and f.endswith(".pt"):
                filepath = os.path.join(root, f)
                try:
                    freed_space += os.path.getsize(filepath)
                    os.remove(filepath)
                    deleted_count += 1
                except Exception:
                    pass

    if deleted_count > 0:
        print(f"  🗑️  Removed {deleted_count} step_*.pt trajectory checkpoints, freed {freed_space / 1024 / 1024:.1f} MB")


def build_grouped_train_cmd(
    base_dir: str,
    relabel_pct: int,
    seed: int,
    gpu: str,
    methods: Sequence[str],
    args: argparse.Namespace,
    epochs_only: bool,
    no_loo: bool,
) -> List[str]:
    """Assemble the shared ``wie.training.train`` command for grouped mode.

    Adds ``--no-loo`` when the grid explicitly requested it, or when none of the
    selected methods need counterfactual/LOO retraining -- mirroring the
    per-method logic of the non-grouped path (epoch_wise_keep_ratio.py) so
    WIE/ICML-baseline-only grids don't pay for leave-one-out training.
    """
    train_cmd = [
        "python", "-m", "wie.training.train",
        "--target", args.target,
        "--model", args.model,
        "--save_dir", base_dir,
        "--relabel", str(relabel_pct),
        "--seed", str(seed),
        "--gpu", gpu,
        "--log_level", args.log_level,
    ]
    if epochs_only:
        train_cmd.append("--epochs-only")
    if no_loo or not any(method_needs_counterfactual(m) for m in methods):
        train_cmd.append("--no-loo")
    # Forward training hyperparameters when provided (wie.training.train accepts
    # --num_epoch/--lr/--batch_size).
    if args.num_epoch is not None:
        train_cmd.extend(["--num_epoch", str(args.num_epoch)])
    if args.lr is not None:
        train_cmd.extend(["--lr", str(args.lr)])
    if args.batch_size is not None:
        train_cmd.extend(["--batch_size", str(args.batch_size)])
    # Enable DVE recording if any DVE method is present (both `dve` and
    # `dve_all_epochs` read the raw DVE shards + projection written at train time).
    if "dve_all_epochs" in methods or "dve" in methods:
        train_cmd.append("--dve-enable")
    return train_cmd


def copy_existing_train_data(existing_train_dir: str, target_dir: str) -> None:
    """Copy existing training artifacts into ``target_dir``.

    Mirrors the ``--skip-training`` behavior of the non-grouped path
    (epoch_wise_keep_ratio.py._copy_training_data): instead of retraining, reuse
    an already-trained directory's contents.
    """
    if not os.path.exists(existing_train_dir):
        raise FileNotFoundError(
            f"Existing training directory not found: {existing_train_dir}"
        )
    os.makedirs(target_dir, exist_ok=True)
    for item in os.listdir(existing_train_dir):
        src_path = os.path.join(existing_train_dir, item)
        dst_path = os.path.join(target_dir, item)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)


def resolved_output_root(output_root: Path | None) -> str:
    """Resolve JUST the --output-root (or the default outputs/ root).

    Resolve the ROOT alone -- NOT the combined ``root/<run>`` -- so it matches how
    the analyzer resolves its ``--base_dir``. resolve_output_dir collapses an
    absolute path outside the repo to its basename under ``outputs/``: applied to
    the root that gives ``<repo>/outputs/result``; applied to the combined path it
    would give ``<repo>/outputs/<run>`` (dropping ``result``), diverging from the
    analyzer. So we always resolve the root, then append the run name separately.
    """
    return str(resolve_output_dir(output_root))


def resolve_grouped_dir(
    prefix: str,
    target: str,
    model: str,
    relabel_pct: int,
    seed: int,
    output_root: Path | None,
) -> str:
    """Return the ONE resolved directory every grouped stage must agree on.

    The run name includes ``{target}_{model}`` so different datasets/models never
    share a grouped dir (which would let a reuse overwrite global_info while
    leaving stale overlap CSVs the analyzer would then misattribute). The root is
    resolved first, then the run name appended, so training (which resolves the
    save_dir), influence/cleanse (which consume it as-is) AND the analyzer (which
    resolves the root) all point under the same ``<resolved-root>/<run>``.
    """
    run_name = f"{prefix}_{target}_{model}_relabel{relabel_pct:02d}_seed{seed:02d}"
    return os.path.join(resolved_output_root(output_root), run_name)


def _csv_has_data_row(path: str) -> bool:
    """True if the CSV has at least one non-blank line beyond the header row."""
    try:
        with open(path, "r") as f:
            non_blank = 0
            for line in f:
                if line.strip():
                    non_blank += 1
                    if non_blank >= 2:  # header + >=1 data row
                        return True
    except OSError:
        return False
    return False


def _expected_cleanse_files(
    grouped_dir: str,
    method: str,
    keep_ratio: int,
    seed: int,
    relabel_pct: int,
    args: argparse.Namespace,
) -> List[Tuple[str, List[str]]]:
    """Return the cleanse output files the current compute flags should produce.

    Each entry is ``(canonical_basename, [alias_paths])`` -- exp_influence_cleansing
    canonicalizes tim_* -> wie_*, so any read-alias form of ``method`` counts.
    Only files the flags actually write are required (overlap skipped at
    relabel=0 or when compute_precision is off; performance skipped when
    compute_retraining_loss is off).
    """
    types = infl_type_read_aliases(method)
    expected: List[Tuple[str, List[str]]] = []

    if str(args.compute_retraining_loss).strip().lower() == "true":
        names = [
            f"cleansed_{t}_{keep_ratio:03d}_pct_performance_{seed:03d}.csv"
            for t in types
        ]
        expected.append((names[0], [os.path.join(grouped_dir, n) for n in names]))

    if (
        relabel_pct is not None
        and int(relabel_pct) > 0
        and str(args.compute_precision).strip().lower() == "true"
    ):
        names = [
            f"relabel_overlap_{t}_{keep_ratio:03d}_pct_{seed:03d}.csv" for t in types
        ]
        expected.append((names[0], [os.path.join(grouped_dir, n) for n in names]))

    return expected


def _bad_cleanse_outputs(expected: List[Tuple[str, List[str]]]) -> List[str]:
    """Return problems (missing / empty) with the expected cleanse files.

    Freshness comes from DELETE-BEFORE-LAUNCH, not wall-clock: the caller removes
    the expected files just before launching, so a leftover file that the run
    didn't rewrite is simply GONE (flagged missing), and a real success recreates
    it. A file counts as success only if it exists AND has >=1 data row (not a
    header-only/partial CSV -- exp_influence_cleansing catches per-seed errors and
    still exits 0, and writes the overlap CSV header BEFORE the epoch loop). We do
    NOT compare mtime to a run_start timestamp: that false-fails on filesystems
    with coarse timestamp granularity or when the storage clock lags the client.
    """
    problems: List[str] = []
    for canonical_name, paths in expected:
        existing = next((p for p in paths if os.path.exists(p)), None)
        if existing is None:
            problems.append(f"{canonical_name} missing")
        elif not _csv_has_data_row(existing):
            problems.append(f"{os.path.basename(existing)} has no data rows")
    return problems


def run_grouped_experiment(
    gpu: str,
    relabel_pct: int,
    seed: int,
    methods: Sequence[str],
    keep_ratios: Sequence[int],
    prefix: str,
    args: argparse.Namespace,
    epochs_only: bool,
    no_loo: bool,
) -> None:
    """Run experiment pipeline: train once, compute all influences, then cleanse.

    This approach shares training data across methods to save time and disk space.
    """
    # ONE resolved directory used by EVERY stage (train, influence, cleanse,
    # copy, log dir, CSV checks) so an absolute --output-root can't split the run
    # between outputs/<basename> (training) and the absolute path (influence).
    grouped_dir = resolve_grouped_dir(
        prefix, args.target, args.model, relabel_pct, seed, args.output_root
    )

    print(f"\n{'='*60}")
    print(f"[GPU {gpu}] Starting grouped experiment: relabel={relabel_pct}%, seed={seed}")
    print(f"[GPU {gpu}] Base directory: {grouped_dir}")
    print(f"[GPU {gpu}] Methods: {', '.join(methods)}")
    print(f"[GPU {gpu}] Keep ratios: {', '.join(map(str, keep_ratios))}")
    print(f"{'='*60}")

    log_dir = Path(grouped_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Train model (only once for this relabel/seed combination), or reuse
    # existing training data when --skip-training is requested.
    if args.skip_training:
        print(
            f"\n[GPU {gpu}] Step 1: Skipping training; copying existing data "
            f"from {args.existing_train_dir}..."
        )
        copy_existing_train_data(args.existing_train_dir, grouped_dir)
        print(f"[GPU {gpu}] ✅ Existing training data copied")
    else:
        print(f"\n[GPU {gpu}] Step 1: Training model...")
        train_cmd = build_grouped_train_cmd(
            grouped_dir, relabel_pct, seed, gpu, methods, args, epochs_only, no_loo
        )
        train_log = log_dir / "train.log"
        with train_log.open("w") as f:
            f.write(f"# Training command: {' '.join(train_cmd)}\n\n")
            result = subprocess.run(train_cmd, stdout=f, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                print(f"[GPU {gpu}] ❌ Training failed, see {train_log}")
                raise RuntimeError(f"Training failed for relabel={relabel_pct}, seed={seed}")
        print(f"[GPU {gpu}] ✅ Training completed")

    # Step 2: Compute influence for all methods
    print(f"\n[GPU {gpu}] Step 2: Computing influence for all methods...")
    for method in methods:
        print(f"[GPU {gpu}]   Computing {method}...")
        # Freshness: delete any stale infl CSV (canonical + legacy alias) for this
        # exact (method, relabel, seed) BEFORE recompute, so a leftover file from a
        # prior run can't make the post-run check pass while the current wie.infl
        # silently failed (it exits 0 even on caught errors) -- Step 3 then deletes
        # step_*.pt, leaving the run un-retryable without retraining.
        expected_csvs = infl_csv_candidates(grouped_dir, method, seed, relabel_pct)
        for cand in expected_csvs:
            try:
                os.remove(cand)
            except FileNotFoundError:
                pass

        infl_cmd = [
            "python", "-m", "wie.infl",
            "--target", args.target,
            "--model", args.model,
            "--save_dir", grouped_dir,
            "--relabel", str(relabel_pct),
            "--seed", str(seed),
            "--type", method,
            "--gpu", gpu,
            "--log_level", args.log_level,
        ]

        infl_log = log_dir / f"infl_{method}.log"
        with infl_log.open("w") as f:
            f.write(f"# Influence command: {' '.join(infl_cmd)}\n\n")
            result = subprocess.run(infl_cmd, stdout=f, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                print(f"[GPU {gpu}] ❌ Influence {method} failed, see {infl_log}")
                raise RuntimeError(f"Influence {method} failed")
        # Confirm the EXACT expected CSV for this (canonical-method, relabel, seed)
        # now exists AND has data rows. Because we deleted stale copies above, its
        # presence + content proves a fresh successful write rather than a leftover
        # or a header-only/partial file. Freshness is from delete-before-launch, not
        # a wall-clock mtime comparison (which false-fails on coarse FS/clock skew).
        fresh_csv = next(
            (p for p in expected_csvs if os.path.exists(p) and _csv_has_data_row(p)),
            None,
        )
        if fresh_csv is None:
            print(
                f"[GPU {gpu}] ❌ Influence {method} exited 0 but wrote no fresh "
                f"non-empty {os.path.basename(expected_csvs[0])}, see {infl_log}"
            )
            raise RuntimeError(
                f"Influence {method} produced no fresh output CSV "
                f"(expected {os.path.basename(expected_csvs[0])})"
            )
        print(f"[GPU {gpu}]   ✅ {method} completed")

    # Step 3: Clean up .pt files (no longer needed after influence computation)
    if not args.no_cleanup:
        print(f"\n[GPU {gpu}] Step 3: Cleaning up checkpoint files...")
        cleanup_pt_files(grouped_dir)
    else:
        print(f"\n[GPU {gpu}] Step 3: Skipping checkpoint cleanup (--no-cleanup)")

    # Step 4: Run cleansing for all (method, keep_ratio) combinations. Collect any
    # failures (non-zero exit OR missing expected outputs -- exp_influence_cleansing
    # catches per-seed errors and only warns) and RAISE after the group so a
    # partially-failed group cannot be reported as a success.
    print(f"\n[GPU {gpu}] Step 4: Running cleansing experiments...")
    cleanse_failures: List[str] = []
    for method in methods:
        for keep_ratio in keep_ratios:
            print(f"[GPU {gpu}]   Cleansing {method} keep={keep_ratio}%...")
            cleanse_cmd = [
                "python", "-m", "wie.training.exp_influence_cleansing",
                "--target", args.target,
                "--model", args.model,
                "--save_dir", grouped_dir,
                "--relabel", str(relabel_pct),
                "--seed", str(seed),
                "--type", method,
                "--keep_ratio", str(keep_ratio),
                "--gpu", gpu,
                "--log_level", args.log_level,
                "--compute_precision", args.compute_precision,
                "--compute_retraining_loss", args.compute_retraining_loss,
            ]
            # Forward lr to the cleansing retraining step when provided.
            if args.lr is not None:
                cleanse_cmd.extend(["--lr", str(args.lr)])

            # Freshness discipline (same as the influence check): delete the
            # expected outputs (canonical + legacy alias) BEFORE launching, so a
            # stale/leftover file can't mask a silent failure; then require the
            # files to exist and be non-empty (a leftover the run didn't rewrite
            # is gone -> flagged missing). No wall-clock comparison (see
            # _bad_cleanse_outputs).
            expected = _expected_cleanse_files(
                grouped_dir, method, keep_ratio, seed, relabel_pct, args
            )
            for _canonical_name, paths in expected:
                for p in paths:
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass

            cleanse_log = log_dir / f"cleanse_{method}_keep{keep_ratio}.log"
            with cleanse_log.open("w") as f:
                f.write(f"# Cleansing command: {' '.join(cleanse_cmd)}\n\n")
                result = subprocess.run(cleanse_cmd, stdout=f, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                print(
                    f"[GPU {gpu}] ❌ Cleansing {method} keep={keep_ratio}% exited "
                    f"{result.returncode}, see {cleanse_log}"
                )
                cleanse_failures.append(
                    f"{method} keep={keep_ratio}% (exit {result.returncode})"
                )
                continue
            problems = _bad_cleanse_outputs(expected)
            if problems:
                print(
                    f"[GPU {gpu}] ❌ Cleansing {method} keep={keep_ratio}% exited 0 "
                    f"but {problems}, see {cleanse_log}"
                )
                cleanse_failures.append(
                    f"{method} keep={keep_ratio}% ({'; '.join(problems)})"
                )
            else:
                print(f"[GPU {gpu}]   ✅ {method} keep={keep_ratio}% completed")

    if cleanse_failures:
        raise RuntimeError(
            f"Grouped cleansing failed for relabel={relabel_pct}, seed={seed}: "
            + "; ".join(cleanse_failures)
        )

    print(f"\n[GPU {gpu}] ✅ Grouped experiment completed: relabel={relabel_pct}%, seed={seed}")


def run_precision_analysis(output_root: Path | None, args: argparse.Namespace) -> None:
    """Run precision analysis on all completed experiments."""
    print("\n" + "="*60)
    print("🔍 RUNNING PRECISION ANALYSIS")
    print("="*60)

    # Import analysis tools
    script_dir = Path(__file__).parent
    analysis_script = script_dir / "analyze_precision_performance.py"

    if not analysis_script.exists():
        raise RuntimeError(f"Analysis script not found: {analysis_script}")

    # Determine base directory for analysis. Resolve the --output-root (matching
    # resolve_grouped_dir / build_save_dir, which resolve the root then append the
    # run name), so the analysis searches the same tree the runs landed in.
    base_dir = resolved_output_root(output_root)

    # Run analysis
    analysis_command = [
        "python", str(analysis_script),
        "--base_dir", base_dir,
        "--target", args.target,
        "--model", args.model,
        "--output_dir", f"{base_dir}/precision_analysis_results",
        "--plot"
    ]

    print(f"Running analysis command: {' '.join(analysis_command)}")

    # Propagate failure. analyze_precision_performance.py exits non-zero when it
    # finds/aggregates NOTHING, so a blanket catch-and-continue here would report
    # "analysis complete" after analyzing nothing. Let the CalledProcessError
    # surface (main() does not catch it) so --run-analysis exits non-zero.
    try:
        subprocess.run(analysis_command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Precision analysis failed (exit {e.returncode}); no results were "
            f"produced under {base_dir}."
        ) from e
    print("✅ Precision analysis completed successfully")
    print(f"📊 Results saved to: {base_dir}/precision_analysis_results/")


def main() -> None:
    args = parse_args()

    # Determine if we should share training
    share_training = args.share_training and not args.no_share_training

    # Validate skip training arguments
    if args.skip_training and not args.existing_train_dir:
        raise ValueError("--existing-train-dir is required when using --skip-training")

    # Merge parameters from CLI and grid file
    methods, keep_ratios, relabel_percentages, seeds, epochs_only, no_loo = merge_grid_args(args)

    # Fail fast: every method is cleansed (Step 4 / non-grouped alike), and the
    # cleansing stage only accepts CLEANSING_SUPPORTED_TYPES. Reject unsupported
    # methods (e.g. loo_all_epochs, which wie.infl computes but cleansing cannot
    # consume) UP FRONT with a clear message, instead of failing mid-run at the
    # cleansing subprocess's argparse after training has already happened.
    # Imported lazily so --help / module import don't pull torch via the
    # cleansing module.
    from wie.training.exp_influence_cleansing import CLEANSING_SUPPORTED_TYPES

    supported = set(CLEANSING_SUPPORTED_TYPES)
    unsupported = [m for m in methods if m not in supported]
    if unsupported:
        raise ValueError(
            "These --methods are not supported by the cleansing stage and would "
            f"fail after training: {unsupported}. "
            f"Supported methods: {sorted(supported)}."
        )

    # Print experiment summary
    n_groups = len(relabel_percentages) * len(seeds)
    n_cleansing = len(methods) * len(keep_ratios)

    # Effective concurrency. Grouped mode runs one worker per (relabel, seed)
    # group (capped by the GPU count); the non-grouped path runs one per GPU.
    # Both honor --max-workers.
    if share_training:
        worker_count = min(len(args.gpus), n_groups)
    else:
        worker_count = len(args.gpus)
    if args.max_workers:
        worker_count = max(1, min(worker_count, args.max_workers))

    print(f"\n{'='*60}")
    print("📊 EXPERIMENT CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Target: {args.target}, Model: {args.model}")
    print(f"  Methods: {', '.join(methods)}")
    print(f"  Keep ratios: {', '.join(map(str, keep_ratios))}")
    print(f"  Relabel percentages: {', '.join(map(str, relabel_percentages))}")
    print(f"  Seeds: {', '.join(map(str, seeds))}")
    print(f"  Share training: {share_training}")
    if share_training:
        print(f"  Training runs: {n_groups} (one per relabel/seed)")
        print(f"  Cleansing runs per group: {n_cleansing}")
        print(f"  Total cleansing experiments: {n_groups * n_cleansing}")
    else:
        total = len(methods) * len(keep_ratios) * len(relabel_percentages) * len(seeds)
        print(f"  Total experiments: {total}")
    print(f"  GPUs: {', '.join(args.gpus)}")
    print(f"  Concurrent GPU workers: {worker_count}"
          + (f" (capped by --max-workers={args.max_workers})" if args.max_workers else ""))
    print(f"{'='*60}\n")

    # Handle dry run
    if args.dry_run:
        print("🔍 DRY RUN MODE - Would execute:")
        if share_training:
            for relabel_pct in relabel_percentages:
                for seed in seeds:
                    # Same ONE resolved directory the real run uses for every stage.
                    grouped_dir = resolve_grouped_dir(
                        args.save_dir_prefix, args.target, args.model,
                        relabel_pct, seed, args.output_root,
                    )
                    print(f"\n[Group] relabel={relabel_pct}%, seed={seed}")
                    if args.skip_training:
                        print(
                            f"  1. Skip training: copy from {args.existing_train_dir} "
                            f"to {grouped_dir}"
                        )
                    else:
                        train_cmd = build_grouped_train_cmd(
                            grouped_dir, relabel_pct, seed, args.gpus[0], methods,
                            args, epochs_only, no_loo,
                        )
                        print(f"  1. Train: {' '.join(train_cmd)}")
                    for method in methods:
                        print(
                            f"  2. Influence: python -m wie.infl --type {method} "
                            f"--save_dir {grouped_dir} --log_level {args.log_level} ..."
                        )
                    print(f"  3. Cleanup .pt files")
                    for method in methods:
                        for keep_ratio in keep_ratios:
                            print(f"  4. Cleanse: {method} keep={keep_ratio}%")
        else:
            tasks = generate_task_map(
                args.gpus, methods, keep_ratios, relabel_percentages, seeds,
                args.output_root, args.save_dir_prefix, args, epochs_only, no_loo,
            )
            for gpu, gpu_tasks in tasks.items():
                if gpu_tasks:
                    print(f"\n# GPU {gpu} ({len(gpu_tasks)} experiments)")
                    for task in gpu_tasks:
                        print(f"  {' '.join(task.command)}")
        if args.run_analysis:
            analysis_base = resolved_output_root(args.output_root)
            print(
                f"\n[Analysis] Would run precision analysis with --base_dir "
                f"{analysis_base} (matches where cleansing outputs are resolved)."
            )
        return

    # Execute experiments
    print("🚀 STARTING INFLUENCE CLEANSING GRID EXPERIMENTS")
    print("="*60)

    if share_training:
        # Grouped mode: train once per (relabel, seed), then run all methods.
        # worker_count (computed above) already honors --gpus and --max-workers.
        groups = list(itertools.product(relabel_percentages, seeds))
        print(f"Using {worker_count} GPU workers for {len(groups)} experiment groups")

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = []
            for idx, (relabel_pct, seed) in enumerate(groups):
                gpu = args.gpus[idx % len(args.gpus)]
                future = executor.submit(
                    run_grouped_experiment,
                    gpu, relabel_pct, seed, methods, keep_ratios,
                    args.save_dir_prefix, args, epochs_only, no_loo
                )
                futures.append((future, relabel_pct, seed))

            for future, relabel_pct, seed in futures:
                try:
                    future.result()
                    print(f"🎯 Group completed: relabel={relabel_pct}%, seed={seed}")
                except Exception as e:
                    print(f"❌ Group failed (relabel={relabel_pct}%, seed={seed}): {e}")
                    raise
    else:
        # Original mode: run each combination independently.
        # worker_count (computed above) already honors --gpus and --max-workers.
        tasks = generate_task_map(
            args.gpus, methods, keep_ratios, relabel_percentages, seeds,
            args.output_root, args.save_dir_prefix, args, epochs_only, no_loo,
        )

        print(f"Using {worker_count} GPU workers")

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(run_command_queue, gpu, gpu_tasks)
                for gpu, gpu_tasks in tasks.items()
                if gpu_tasks
            ]

            completed = 0
            total = len(futures)
            for future in futures:
                try:
                    future.result()
                    completed += 1
                    print(f"🎯 GPU worker completed ({completed}/{total})")
                except Exception as e:
                    print(f"❌ GPU worker failed: {e}")
                    raise

    print("\n" + "="*60)
    print("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*60)

    # Run precision analysis if requested
    if args.run_analysis:
        run_precision_analysis(args.output_root, args)
    else:
        print("\n💡 To analyze results, run:")
        base_dir = resolved_output_root(args.output_root)
        print(f"   python scripts/analyze_precision_performance.py --base_dir {base_dir} --target {args.target} --model {args.model} --plot")


if __name__ == "__main__":
    main()