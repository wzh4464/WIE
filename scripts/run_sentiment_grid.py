"""run_sentiment_grid.py
=================================
Utility script for sweeping sentiment training experiments across multiple GPUs.

Features
--------
* Builds a Cartesian product over learning rate, label smoothing, and dropout settings.
* Distributes runs evenly across the supplied GPU ids and executes them sequentially per GPU.
* Supports a dry-run mode for inspecting generated `pixi run` commands without execution.
* Captures stdout/stderr for each run into the experiment save_dir log (runtime errors included).
* Parameter lists may be provided via command-line options or a JSON "grid file".

Usage example::

    pixi run python scripts/run_sentiment_grid.py \
        --dry-run \
        --output-root outputs \
        --save-dir-prefix sentiment_bert \
        --grid-file configs/sentiment_grid.json

Create a JSON grid description with keys ``"lrs"``, ``"label_smoothing"``, and ``"dropouts"``::

    {
      "lrs": ["1e-6", "5e-6"],
      "label_smoothing": ["0", "0.1"],
      "dropouts": ["0.3", "0.4"]
    }

Run ``pixi run python scripts/run_sentiment_grid.py --help`` for detailed options.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

DEFAULT_LRS = ["1e-6", "5e-6", "1e-5", "2e-5"]
DEFAULT_LABEL_SMOOTHING = ["0", "0.1", "0.2"]
DEFAULT_DROPOUTS = ["0.3", "0.4", "0.5"]


@dataclass
class RunTask:
    """Single command execution metadata for a scheduled run."""

    command: List[str]
    log_path: Path


def parse_args() -> argparse.Namespace:
    """Configure and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate and optionally run a grid of sentiment training experiments "
            "with pixi, distributing runs across the requested GPUs."
        )
    )
    parser.add_argument(
        "--lrs",
        nargs="+",
        default=None,
        help=(
            "Learning rate values to sweep (scientific notation accepted). "
            f"Default: {', '.join(DEFAULT_LRS)}."
        ),
    )
    parser.add_argument(
        "--label-smoothing",
        nargs="+",
        dest="label_smoothing",
        default=None,
        help=(
            "Label smoothing values to sweep. "
            f"Default: {', '.join(DEFAULT_LABEL_SMOOTHING)}."
        ),
    )
    parser.add_argument(
        "--dropouts",
        nargs="+",
        default=None,
        help=(f"Dropout rates to sweep. Default: {', '.join(DEFAULT_DROPOUTS)}."),
    )
    parser.add_argument(
        "--grid-file",
        type=Path,
        help=(
            "Optional JSON file containing keys 'lrs', 'label_smoothing', and 'dropouts'. "
            "Values from the command line take precedence."
        ),
    )
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
            "Optional directory prepended to generated save_dir values. If omitted, save_dir "
            "is treated as relative to the working directory."
        ),
    )
    parser.add_argument(
        "--save-dir-prefix",
        default="sentiment_bert",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed forwarded to wie.training.train.",
    )
    parser.add_argument(
        "--relabel",
        type=int,
        default=30,
        help="Relabel iterations forwarded to wie.training.train.",
    )
    parser.add_argument(
        "--n-tr",
        type=int,
        default=16384,
        help="Training subset size forwarded to wie.training.train.",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=2048,
        help="Validation subset size forwarded to wie.training.train.",
    )
    parser.add_argument(
        "--num-epoch",
        type=int,
        default=10,
        help="Number of epochs forwarded to wie.training.train.",
    )
    parser.add_argument(
        "--target",
        default="sentiment",
        help="Target dataset forwarded to wie.training.train.",
    )
    parser.add_argument(
        "--model",
        default="bert",
        help="Model identifier forwarded to wie.training.train.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level forwarded to wie.training.train.",
    )
    parser.add_argument(
        "--epochs-only",
        dest="epochs_only",
        action="store_true",
        help="Include the --epochs-only flag when launching the training script.",
    )
    parser.add_argument(
        "--no-epochs-only",
        dest="epochs_only",
        action="store_false",
        help="Omit the --epochs-only flag when launching the training script.",
    )
    parser.add_argument(
        "--no-loo",
        dest="no_loo",
        action="store_true",
        help="Include the --no-loo flag when launching the training script.",
    )
    parser.add_argument(
        "--allow-loo",
        dest="no_loo",
        action="store_false",
        help="Omit the --no-loo flag when launching the training script.",
    )
    parser.add_argument(
        "--additional-args",
        nargs=argparse.REMAINDER,
        help=(
            "Extra arguments appended verbatim to the generated command. "
            "Separate them from this script's options with "
            "'--additional-args -- <extra args>'."
        ),
    )
    parser.set_defaults(epochs_only=True, no_loo=True)
    return parser.parse_args()


def load_grid_from_file(path: Path) -> Dict[str, List[str]]:
    """Load grid specification from disk."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - guard for invalid files
        raise ValueError(f"Failed to parse grid file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Grid file {path} must contain a JSON object.")
    grid: Dict[str, List[str]] = {}
    for key in ("lrs", "label_smoothing", "dropouts"):
        value = data.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            raise ValueError(f"Grid entry '{key}' must be a list.")
        grid[key] = [str(item) for item in value]
    return grid


def merge_grid_args(args: argparse.Namespace) -> Tuple[List[str], List[str], List[str]]:
    """Combine CLI values with the optional grid file."""
    file_grid: Dict[str, List[str]] = {}
    if args.grid_file:
        if not args.grid_file.exists():
            raise FileNotFoundError(f"Grid file {args.grid_file} does not exist.")
        file_grid = load_grid_from_file(args.grid_file)
    lrs = args.lrs or file_grid.get("lrs") or DEFAULT_LRS
    ls_vals = (
        args.label_smoothing
        or file_grid.get("label_smoothing")
        or DEFAULT_LABEL_SMOOTHING
    )
    dropout_vals = args.dropouts or file_grid.get("dropouts") or DEFAULT_DROPOUTS
    return (
        ensure_values("lrs", lrs),
        ensure_values("label_smoothing", ls_vals),
        ensure_values("dropouts", dropout_vals),
    )


def ensure_values(name: str, values: Sequence[str]) -> List[str]:
    """Validate that the provided hyper-parameter list is non-empty."""
    cleaned = [str(v).strip() for v in values if str(v).strip() != ""]
    if not cleaned:
        raise ValueError(f"No values provided for {name}.")
    return cleaned


def normalize_additional_args(values: Iterable[str] | None) -> List[str] | None:
    """Clean up optional extra CLI arguments."""
    if not values:
        return None
    cleaned = list(values)
    if cleaned and cleaned[0] == "--":
        cleaned = cleaned[1:]
    return cleaned or None


def format_value_token(value: str) -> str:
    """Sanitize hyper-parameter values for directory names."""
    return value.replace(".", "p")


def build_save_dir(
    prefix: str, lr: str, label_smoothing: str, dropout: str, output_root: Path | None
) -> str:
    """Construct the save directory string for a given parameter triplet."""
    tokens = (
        f"lr{format_value_token(lr)}",
        f"ls{format_value_token(label_smoothing)}",
        f"dr{format_value_token(dropout)}",
    )
    relative = f"{prefix}_{'_'.join(tokens)}"
    if output_root:
        return str(output_root / relative)
    return relative


def build_command(
    base_args: Dict[str, object],
    lr: str,
    label_smoothing: str,
    dropout: str,
    gpu: str,
    save_dir: str,
    additional_args: Iterable[str] | None,
) -> List[str]:
    """Assemble the pixi run command for a single experiment."""
    command = [
        "pixi",
        "run",
        "python",
        "-m",
        "wie.training.train",
        "--target",
        base_args["target"],
        "--model",
        base_args["model"],
        "--save_dir",
        save_dir,
        "--relabel",
        str(base_args["relabel"]),
        "--seed",
        str(base_args["seed"]),
        "--gpu",
        gpu,
        "--log_level",
        base_args["log_level"],
        "--n_tr",
        str(base_args["n_tr"]),
        "--n_val",
        str(base_args["n_val"]),
        "--num_epoch",
        str(base_args["num_epoch"]),
        "--lr",
        lr,
        "--dropout",
        dropout,
        "--label_smoothing",
        label_smoothing,
    ]
    if base_args["epochs_only"]:
        command.append("--epochs-only")
    if base_args["no_loo"]:
        command.append("--no-loo")
    if additional_args:
        command.extend(additional_args)
    return command


def generate_task_map(
    gpus: Sequence[str],
    lrs: Sequence[str],
    ls_vals: Sequence[str],
    dropout_vals: Sequence[str],
    base_args: Dict[str, object],
    output_root: Path | None,
    prefix: str,
    additional_args: Iterable[str] | None,
) -> Dict[str, List[RunTask]]:
    """Return a GPU-to-command mapping with evenly distributed runs."""
    tasks: Dict[str, List[RunTask]] = {gpu: [] for gpu in gpus}
    combos = list(itertools.product(lrs, ls_vals, dropout_vals))
    for idx, (lr, ls_val, dropout) in enumerate(combos):
        gpu = gpus[idx % len(gpus)]
        save_dir = build_save_dir(prefix, lr, ls_val, dropout, output_root)
        command = build_command(
            base_args, lr, ls_val, dropout, gpu, save_dir, additional_args
        )
        log_path = Path(save_dir) / "train.log"
        tasks[gpu].append(RunTask(command=command, log_path=log_path))
    return tasks


def run_command_queue(gpu: str, tasks: Sequence[RunTask]) -> None:
    """Execute a sequence of commands on a single GPU sequentially."""
    for task in tasks:
        command_str = " ".join(task.command)
        print(f"[GPU {gpu}] Running: {command_str}")
        print(f"[GPU {gpu}] Log: {task.log_path}")
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        with task.log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"# GPU: {gpu}\n# Command: {command_str}\n\n")
            log_file.flush()
            try:
                subprocess.run(
                    task.command, check=True, stdout=log_file, stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError:
                print(f"[GPU {gpu}] Command failed; see {task.log_path}.")
                raise


def main() -> None:
    args = parse_args()
    lrs, ls_vals, dropout_vals = merge_grid_args(args)
    base_args = {
        "target": args.target,
        "model": args.model,
        "relabel": args.relabel,
        "seed": args.seed,
        "log_level": args.log_level,
        "n_tr": args.n_tr,
        "n_val": args.n_val,
        "num_epoch": args.num_epoch,
        "epochs_only": args.epochs_only,
        "no_loo": args.no_loo,
    }
    additional_args = normalize_additional_args(args.additional_args)
    tasks = generate_task_map(
        args.gpus,
        lrs,
        ls_vals,
        dropout_vals,
        base_args,
        args.output_root,
        args.save_dir_prefix,
        additional_args,
    )
    if args.dry_run:
        for gpu, gpu_tasks in tasks.items():
            print(f"\n# GPU {gpu}")
            for task in gpu_tasks:
                command_str = " ".join(task.command)
                print(f"{command_str}  # log: {task.log_path}")
        return
    worker_count = (
        max(1, min(len(args.gpus), args.max_workers))
        if args.max_workers
        else len(args.gpus)
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(run_command_queue, gpu, gpu_tasks)
            for gpu, gpu_tasks in tasks.items()
            if gpu_tasks
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
