#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import shlex
import shutil

# Make the src/ layout importable when this script is run by path
# (python scripts/foo.py) without the editable install (pixi run install).
try:
    import wie  # noqa: F401
except ModuleNotFoundError:
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from wie.utils.paths import resolve_output_dir

"""
Epoch-wise cleansing orchestration.

Steps:
- Train the model (seed can be specified)
- Compute influence scores (all epochs or configured method)
- Run cleansing based on influence scores

Refactored for clarity: argument parsing and command assembly are split into
helpers to reduce complexity in `main()` while preserving CLI behavior.
"""


def _positive_int(value: str) -> int:
    """argparse type: a strictly positive int (>= 1).

    Rejects ``--length 0`` / negatives, which would collapse a WIE window to an
    empty interval and yield an all-zero score CSV that cleansing treats as
    success.
    """
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}")
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {ivalue}")
    return ivalue


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the epoch-wise keep-ratio pipeline."""
    parser = argparse.ArgumentParser(description="Epoch-wise cleansing experiment")
    # Required/primary settings
    parser.add_argument("--target", type=str, help="Target dataset")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--save_dir", type=str, help="Save directory")
    parser.add_argument("--relabel", type=int, help="Relabel parameter")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, default 0")
    parser.add_argument(
        "--decay", type=str, default="False", help="LR decay (True/False)"
    )
    parser.add_argument(
        "--keep_ratio", type=int, default=90, help="Percent to keep (default 90)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate (TextAttack style)"
    )
    parser.add_argument("--n_tr", type=int, help="Number of training samples")
    parser.add_argument("--n_val", type=int, help="Number of validation samples")
    parser.add_argument("--num_epoch", type=int, default=5, help="Training epochs")
    # Regularization knobs
    parser.add_argument(
        "--dropout", type=float, default=None, help="Head dropout prob for BERT [0,1)"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=None,
        help="Label smoothing epsilon for BCE [0,0.5)",
    )
    # TextAttack-style knobs
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--max_length", type=int, default=128, help="Max sequence length"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=500, help="Warmup steps"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Pretrained model name",
    )
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Grad accumulation steps",
    )
    parser.add_argument(
        "--validation_split", type=float, default=0.1, help="Validation split ratio"
    )
    # Mixed precision
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision")
    # Logging
    parser.add_argument(
        "--use_tensorboard", action="store_true", help="Use TensorBoard logging"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases logging"
    )
    # Trainer strategies
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps", "no"],
        help="Evaluation strategy",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="best",
        choices=["best", "epoch", "steps", "no"],
        help="Save strategy",
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        default=True,
        help="Load best model at end",
    )
    # Compatibility flags
    parser.add_argument(
        "--save_recording", action="store_true", default=True, help="Save recording"
    )
    parser.add_argument("--steps_only", action="store_true", help="Steps only")
    parser.add_argument("--epochs_only", action="store_true", help="Epochs only")
    parser.add_argument(
        "--compute_counterfactual", action="store_true", help="Compute counterfactual"
    )
    parser.add_argument(
        "--no-loo", action="store_true", help="Skip leave-one-out training", dest="no_loo"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip cleanup of model checkpoint files (.pt) after experiment completion",
        dest="no_cleanup",
    )
    parser.add_argument(
        "--init_model", type=str, default=None, help="Initial model path"
    )
    parser.add_argument(
        "--relabel_csv", type=str, default=None, help="Relabel CSV file"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="wie_all_epochs",
        choices=[
            "wie_all_epochs",
            "wie_last",
            "wie_first",
            "wie_middle",
            # Legacy tim_* aliases: the factory maps these to the wie_*
            # calculators (INFL_TYPE_ALIASES), so accept them here too instead
            # of rejecting old commands at argparse time.
            "tim_all_epochs",
            "tim_last",
            "tim_first",
            "tim_middle",
            "dve_all_epochs",
            "loo_all_epochs",
            "lava_all_epochs",
            "icml_all_epochs",
            "sgd",
            "icml",
            "tracin",
            "td_influence",
            "lava",
            "dve",
        ],
        help="Influence method",
    )
    # TD-Influence passthrough
    parser.add_argument(
        "--use_projection",
        action="store_true",
        help="Enable random projection for TD-Influence",
    )
    parser.add_argument(
        "--proj_dim", type=int, default=None, help="Projection dimension"
    )
    parser.add_argument(
        "--proj_type",
        type=str,
        default=None,
        choices=["gaussian", "achlioptas"],
        help="Projection type",
    )
    parser.add_argument(
        "--use_last_layer_only",
        action="store_true",
        help="Last layer only (TD-Influence)",
    )
    # Window length for the WIE window variants (wie_first/wie_middle/wie_last):
    # number of epochs in the window. Matches wie.infl's --length default (3).
    # Must be >= 1: a non-positive length collapses the window to an empty
    # interval (all-zero scores that cleansing would treat as success).
    parser.add_argument(
        "--length",
        type=_positive_int,
        default=3,
        help="Window length in epochs for wie_first/wie_middle/wie_last (>=1, default 3)",
    )
    # System
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would run, then exit",
    )
    # Skip training options
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training step and use existing training data",
    )
    parser.add_argument(
        "--existing_train_dir",
        type=str,
        help="Path to existing training data directory (required when --skip_train is used)",
    )
    # Cleansing functionality control
    parser.add_argument(
        "--compute_precision",
        type=str,
        default="True",
        choices=["True", "False", "true", "false"],
        help="Compute precision/recall/F1 statistics for flip point identification (default: True)",
    )
    parser.add_argument(
        "--compute_retraining_loss",
        type=str,
        default="True",
        choices=["True", "False", "true", "false"],
        help="Perform actual retraining and compute validation/training losses (default: True)",
    )
    return parser


def _resolve_save_dir(save_dir: str) -> str:
    """Place save_dir under project root outputs/ for consistency.

    Uses centralized resolve_output_dir for unified path handling.
    """
    return str(resolve_output_dir(save_dir))


def _build_train_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "wie.training.train",
        "--target",
        args.target,
        "--model",
        args.model,
        "--save_dir",
        args.save_dir,
        "--relabel",
        str(args.relabel),
        "--seed",
        str(args.seed),
        "--gpu",
        str(args.gpu),
        "--log_level",
        args.log_level,
    ]

    # Add optional parameters supported by wie.training.train
    # Recording flags
    if args.type == "td_influence":
        # Default to epochs-only for TD-Influence unless user explicitly overrides
        if args.steps_only:
            cmd.append("--steps-only")
        elif not args.save_recording:
            cmd.append("--no-recording")
        else:
            cmd.append("--epochs-only")
    elif args.type == "dve" or args.type == "dve_all_epochs":
        # DVE requires enabling DVE during training
        cmd.append("--dve-enable")
        # Handle recording options for DVE
        if not args.save_recording:
            cmd.append("--no-recording")
        elif args.steps_only:
            cmd.append("--steps-only")
        elif args.epochs_only:
            cmd.append("--epochs-only")
        else:
            # DVE works best with steps recording for granular influence computation
            cmd.append("--steps-only")
        # Pass DVE projection dimension if specified via --proj_dim
        if hasattr(args, "proj_dim") and getattr(args, "proj_dim") is not None:
            cmd.extend(["--dve-proj-dim", str(getattr(args, "proj_dim"))])
    elif args.type == "loo_all_epochs":
        # LOO requires counterfactual models, enable LOO training
        if not args.compute_counterfactual:
            # Override the default --no-loo behavior for LOO influence type
            pass  # Don't add --no-loo
        # LOO works with epoch-only recording
        if not args.save_recording:
            cmd.append("--no-recording")
        elif not args.steps_only:
            cmd.append("--epochs-only")
    elif args.type == "lava_all_epochs":
        # LAVA requires counterfactual models, enable LOO training
        if not args.compute_counterfactual:
            # Override the default --no-loo behavior for LAVA influence type
            pass  # Don't add --no-loo
        # LAVA works with epoch-only recording
        if not args.save_recording:
            cmd.append("--no-recording")
        elif not args.steps_only:
            cmd.append("--epochs-only")
    elif args.type == "icml_all_epochs":
        # ICML All Epochs doesn't need counterfactual models, works with epoch-only recording
        if not args.save_recording:
            cmd.append("--no-recording")
        elif not args.steps_only:
            cmd.append("--epochs-only")
    else:
        # Other types: keep original behavior
        if not args.save_recording:
            cmd.append("--no-recording")
        if args.steps_only:
            cmd.append("--steps-only")
        elif args.epochs_only:
            cmd.append("--epochs-only")
    # Add --no-loo flag based on explicit parameter or default behavior
    if args.no_loo or (
        not args.compute_counterfactual
        and args.type not in ["loo_all_epochs", "lava_all_epochs", "icml_all_epochs"]
    ):  # Add --no-loo if explicitly requested or if compute_counterfactual is False (except for LOO, LAVA, and ICML influence types)
        cmd.append("--no-loo")
    if args.decay and args.decay.lower() == "true":
        cmd.extend(["--decay", "True"])

    # Add optional string parameters
    if args.init_model is not None:
        cmd.extend(["--init_model", args.init_model])
    if args.relabel_csv is not None:
        cmd.extend(["--relabel_csv", args.relabel_csv])
    if args.n_tr is not None:
        cmd.extend(["--n_tr", str(args.n_tr)])
    if args.n_val is not None:
        cmd.extend(["--n_val", str(args.n_val)])
    if args.num_epoch is not None:
        cmd.extend(["--num_epoch", str(args.num_epoch)])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.lr is not None:
        cmd.extend(["--lr", str(args.lr)])
    return cmd


def _build_infl_cmd(args: argparse.Namespace) -> list[str]:
    """Build command to compute influence scores for all epochs."""
    cmd = [
        sys.executable,
        "-m",
        "wie.infl",
        "--target",
        args.target,
        "--model",
        args.model,
        "--type",
        args.type,
        "--save_dir",
        args.save_dir,
        "--relabel",
        str(args.relabel),
        "--seed",
        str(args.seed),
        "--gpu",
        str(args.gpu),
        "--log_level",
        args.log_level,
    ]

    # TD-Influence passthrough flags
    if args.type == "td_influence":
        if args.use_projection:
            cmd.append("--use_projection")
        if args.proj_dim is not None:
            cmd.extend(["--proj_dim", str(args.proj_dim)])
        if args.proj_type is not None:
            cmd.extend(["--proj_type", args.proj_type])
        if args.use_last_layer_only:
            cmd.append("--use_last_layer_only")

    # Forward the window --length to the WIE window variants. wie.infl accepts
    # --length and maps it to the calculator's `length` kwarg (WieFirst/Middle/
    # Last). Gated to the window types (incl. tim_* aliases) to keep the emitted
    # command tidy; the non-window methods don't use length.
    if args.type in {
        "wie_first",
        "wie_middle",
        "wie_last",
        "tim_first",
        "tim_middle",
        "tim_last",
    }:
        cmd.extend(["--length", str(args.length)])
    return cmd


def _build_cleansing_cmd(args: argparse.Namespace) -> list[str]:
    """Build command for the post-influence cleansing experiment."""
    cmd = [
        sys.executable,
        "-m",
        "wie.training.exp_influence_cleansing",
        "--target",
        args.target,
        "--model",
        args.model,
        "--save_dir",
        args.save_dir,
        "--relabel",
        str(args.relabel),
        "--keep_ratio",
        str(args.keep_ratio),
        "--seed",
        str(args.seed),
        "--decay",
        str(args.decay),
        "--lr",
        str(args.lr),
        "--type",
        args.type,
        "--gpu",
        str(args.gpu),
        "--log_level",
        args.log_level,
        "--compute_precision",
        args.compute_precision,
        "--compute_retraining_loss",
        args.compute_retraining_loss,
    ]
    return cmd

    # exp_influence_cleansing.py doesn't support --n_tr/--n_val parameters
    # The data size information is loaded from the training summary file


def _fmt_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _copy_training_data(existing_train_dir: str, target_save_dir: str) -> None:
    """Copy existing training data to the target save directory."""
    if not os.path.exists(existing_train_dir):
        raise FileNotFoundError(
            f"Existing training directory not found: {existing_train_dir}"
        )

    # Create target directory if it doesn't exist
    os.makedirs(target_save_dir, exist_ok=True)

    print(f"Copying training data from {existing_train_dir} to {target_save_dir}")

    # Copy all files from existing training directory
    for item in os.listdir(existing_train_dir):
        src_path = os.path.join(existing_train_dir, item)
        dst_path = os.path.join(target_save_dir, item)

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  Copied file: {item}")
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            print(f"  Copied directory: {item}")

    print("Training data copy completed.")


def _cleanup_model_checkpoints(save_dir: str) -> None:
    """Reclaim disk by deleting per-step trajectory checkpoints (``step_*.pt``).

    Only ``step_*.pt`` (incl. those under ``records/``) is removed. These
    per-step model snapshots are read **solely** while computing influence
    (wie/icml/lie/nohess) and by nothing downstream: cleansing retrains from
    ``init_*.pt`` into its own records subdir, and loo-valuation/analysis read
    the CSVs plus ``epoch_final_*``/``counterfactual_*``. So once influence is
    computed they are unreferenced, and they are the dominant disk consumer.

    Preserved (still referenced by computation): ``init_*.pt`` (cleansing
    retraining), ``epoch_*``/``epoch_final_*`` (lava, final model, valuation),
    ``counterfactual_*`` (true/loo ground-truth + valuation), ``dve_raw/``
    (DVE), and all ``*.csv``/``*.json``/``*.log``.

    NOTE: re-running influence for this config would require re-training, since
    the trajectory snapshots are gone.
    """
    if not os.path.exists(save_dir):
        print(f"Warning: Save directory {save_dir} does not exist, skipping cleanup")
        return

    print(f"\n🧹 Removing per-step trajectory checkpoints (step_*.pt) in {save_dir}")

    deleted_count = 0
    freed_space = 0

    for root, dirs, files in os.walk(save_dir):
        for filename in files:
            # step_NNNNNN.pt only; excludes dve_step_*/counterfactual_*/epoch_*/init_*
            if filename.startswith("step_") and filename.endswith(".pt"):
                filepath = os.path.join(root, filename)
                try:
                    file_size = os.path.getsize(filepath)
                    os.remove(filepath)
                    deleted_count += 1
                    freed_space += file_size
                except Exception as e:
                    print(f"  Warning: Failed to delete {filepath}: {e}")

    if deleted_count > 0:
        print(f"✅ Removed {deleted_count} step_*.pt checkpoints, freed {freed_space / 1024 / 1024:.2f} MB")
    else:
        print("ℹ️  No step_*.pt checkpoints found to delete")


def run_pipeline(args: argparse.Namespace) -> None:
    """Assemble and run the three-stage pipeline (or print with --dry-run)."""
    # Validate skip_train arguments
    if args.skip_train and not args.existing_train_dir:
        raise ValueError("--existing_train_dir is required when using --skip_train")

    if args.skip_train and not os.path.exists(args.existing_train_dir):
        raise FileNotFoundError(
            f"Existing training directory not found: {args.existing_train_dir}"
        )

    # Reject --steps_only for the reverse-SGD WIE calculators (and wie_all_epochs'
    # step-by-step mode). They consume FULL step records: either the epoch files
    # that embed per-step idx/lr, or seeded step_{step}_{seed}.pt files with
    # model_state/idx/lr. --steps_only writes bare, unseeded step_NNNNNN.pt state
    # dicts AND suppresses epoch checkpoints, leaving nothing load_step_data can
    # read -> the influence stage fails and the CLI swallows it (exit 0, no CSV).
    # Gate up front (covers tim_* aliases too). Full/epochs-only recording is
    # unaffected.
    _WIE_STEP_RECORD_TYPES = {
        "wie_all_epochs",
        "wie_last",
        "wie_first",
        "wie_middle",
        "tim_all_epochs",
        "tim_last",
        "tim_first",
        "tim_middle",
    }
    if args.steps_only and args.type in _WIE_STEP_RECORD_TYPES:
        raise ValueError(
            f"--steps_only is incompatible with --type {args.type}: the WIE "
            "reverse-SGD calculators require full step recording (epoch files "
            "embedding per-step idx/lr, or seeded step records). Re-run with "
            "--epochs_only or the default full recording (omit --steps_only)."
        )

    # Normalize path under project root outputs/
    args.save_dir = _resolve_save_dir(args.save_dir)
    train_cmd = _build_train_cmd(args)
    infl_cmd = _build_infl_cmd(args)
    cleansing_cmd = _build_cleansing_cmd(args)

    if args.dry_run:
        if args.skip_train:
            print(
                f"# Skip training: copy from {args.existing_train_dir} to {args.save_dir}"
            )
        else:
            # Print training command; environment vars shown as comments for clarity
            env_notes = []
            if args.dropout is not None:
                env_notes.append(f"HF_TEXT_DROPOUT={args.dropout}")
            if args.label_smoothing is not None:
                env_notes.append(f"LABEL_SMOOTHING={args.label_smoothing}")
            note = ("  # env: " + ", ".join(env_notes)) if env_notes else ""
            print(_fmt_cmd(train_cmd) + note)
        print(_fmt_cmd(infl_cmd))
        print(_fmt_cmd(cleansing_cmd))
        return

    # Execute pipeline
    if args.skip_train:
        # Copy existing training data instead of training
        _copy_training_data(args.existing_train_dir, args.save_dir)
    else:
        # Prepare environment for training subprocess
        train_env = os.environ.copy()
        if args.dropout is not None:
            train_env["HF_TEXT_DROPOUT"] = str(args.dropout)
        if args.label_smoothing is not None:
            train_env["LABEL_SMOOTHING"] = str(args.label_smoothing)

        subprocess.run(train_cmd, check=True, env=train_env)

    # Always run influence computation and cleansing
    subprocess.run(infl_cmd, check=True)
    subprocess.run(cleansing_cmd, check=True)

    # Clean up model checkpoints to save disk space (unless --no-cleanup is specified)
    if not args.no_cleanup:
        _cleanup_model_checkpoints(args.save_dir)
    else:
        print("\nℹ️  Skipping checkpoint cleanup (--no-cleanup specified)")


def main():
    args = build_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
