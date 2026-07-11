#!/usr/bin/env python3
"""BERT Relabel Detection Experiment Script
==========================================

Runs comprehensive relabel detection experiments on BERT/sentiment dataset
comparing all influence methods (WIE, ICML, LAVA, DVE) across:
- Multiple relabel ratios (poison percentages)
- Multiple keep ratios (cleansing thresholds)
- Multiple seeds (statistical significance)
- All training epochs (temporal analysis)

This script provides BERT-optimized defaults and convenient wrappers around the
existing grid search infrastructure.

Example Usage
-------------
Quick test with default parameters::

    python scripts/run_bert_relabel_experiments.py --save-dir bert_test --gpus 0

Custom parameter ranges::

    python scripts/run_bert_relabel_experiments.py \\
        --save-dir bert_custom \\
        --relabel-ratios 5 10 15 \\
        --keep-ratios 80 90 \\
        --seeds 0 1 2 3 4 \\
        --methods wie_all_epochs icml_all_epochs \\
        --gpus 0 1 2 3

Analysis only (skip experiments)::

    python scripts/run_bert_relabel_experiments.py \\
        --save-dir bert_existing \\
        --analyze-only

Run with JSON config file::

    python scripts/run_bert_relabel_experiments.py \\
        --save-dir bert_experiments \\
        --config-file configs/bert_grid.json \\
        --gpus 0 1 2 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Make the src/ layout importable when this script is run by path
# (python scripts/foo.py) without the editable install (pixi run install).
try:
    import wie  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wie.utils.paths import resolve_output_dir


# Default BERT-optimized parameters (matching the BERT/sentiment defaults in src/wie/training/config.py)
DEFAULT_CONFIG = {
    "target": "sentiment",
    "model": "bert",
    "methods": [
        "wie_all_epochs",
        "icml_all_epochs",
        "lava_all_epochs",
        "dve_all_epochs",
    ],
    "relabel_percentages": [5, 10, 15, 20],  # Poison ratios
    "keep_ratios": [70, 80, 90],  # Keep top X% samples
    "seeds": [0],  # Single seed for faster experiments
    "num_epochs": 5,  # BERT default
    "lr": 2e-5,  # BERT default learning rate
    "batch_size": 16,  # BERT default batch size
    "decay": 1.0,  # No LR decay by default
}


def parse_args() -> argparse.Namespace:
    """Configure and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run BERT relabel detection experiments with all influence methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with default parameters (results in result/bert_test_*)
  python scripts/run_bert_relabel_experiments.py --save-dir bert_test --gpus 0

  # Custom parameter ranges with custom output directory
  python scripts/run_bert_relabel_experiments.py \\
      --save-dir bert_custom \\
      --output-root outputs \\
      --relabel-ratios 5 10 15 \\
      --keep-ratios 80 90 \\
      --seeds 0 1 2 3 4 \\
      --methods wie_all_epochs icml_all_epochs \\
      --gpus 0 1 2 3

  # Analysis only (skip experiments)
  python scripts/run_bert_relabel_experiments.py \\
      --save-dir bert_existing \\
      --output-root result \\
      --analyze-only

  # Use JSON config file
  python scripts/run_bert_relabel_experiments.py \\
      --save-dir bert_experiments \\
      --output-root result \\
      --config-file configs/bert_grid.json \\
      --gpus 0 1 2 3
        """,
    )

    # Basic configuration
    parser.add_argument(
        "--save-dir", required=True, help="Save directory prefix for experiment results (combined with output-root)"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="result",
        help="Root output directory for all experiments (default: result)",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=str,
        help="GPU IDs to use for experiments (required unless --analyze-only)",
    )

    # Parameter overrides (optional - uses DEFAULT_CONFIG if not specified)
    parser.add_argument(
        "--relabel-ratios",
        nargs="+",
        type=int,
        default=None,
        help=f"Relabel percentages to test (default: {DEFAULT_CONFIG['relabel_percentages']})",
    )
    parser.add_argument(
        "--keep-ratios",
        nargs="+",
        type=int,
        default=None,
        help=f"Keep ratio percentages to test (default: {DEFAULT_CONFIG['keep_ratios']})",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=f"Random seeds (default: {DEFAULT_CONFIG['seeds']})",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=f"Influence methods to compare (default: {DEFAULT_CONFIG['methods']})",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help=f"Number of training epochs (default: {DEFAULT_CONFIG['num_epochs']})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=f"Learning rate (default: {DEFAULT_CONFIG['lr']})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Batch size (default: {DEFAULT_CONFIG['batch_size']})",
    )

    # Config file
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Optional JSON config file (CLI args take precedence)",
    )

    # Execution control
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip experiments, only analyze existing results",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true", help="Skip post-experiment analysis"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip resource validation checks"
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )

    # Advanced options
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training phase, use existing trained models",
    )
    parser.add_argument(
        "--existing-train-dir",
        type=str,
        help="Path to existing training data (used with --skip-training)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for experiments",
    )

    return parser.parse_args()


def load_config_file(config_path: Path) -> Dict:
    """Load experiment configuration from JSON file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"✓ Loaded configuration from {config_path}")
        return config
    except Exception as e:
        raise ValueError(f"Failed to load config file {config_path}: {e}")


def get_bert_experiment_config(args: argparse.Namespace) -> Dict:
    """
    Build experiment configuration from args, config file, and defaults.

    Priority: CLI args > config file > DEFAULT_CONFIG
    """
    config = DEFAULT_CONFIG.copy()

    # Load config file if specified
    if args.config_file:
        if not args.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {args.config_file}")
        file_config = load_config_file(args.config_file)
        config.update(file_config)

    # Override with CLI arguments (highest priority)
    if args.relabel_ratios is not None:
        config["relabel_percentages"] = args.relabel_ratios
    if args.keep_ratios is not None:
        config["keep_ratios"] = args.keep_ratios
    if args.seeds is not None:
        config["seeds"] = args.seeds
    if args.methods is not None:
        config["methods"] = args.methods
    if args.num_epochs is not None:
        config["num_epochs"] = args.num_epochs
    if args.lr is not None:
        config["lr"] = args.lr
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    # Add save directory
    config["save_dir"] = args.save_dir

    return config


def validate_resources(config: Dict, args: argparse.Namespace) -> None:
    """
    Validate system resources before running experiments.

    Checks:
    - CUDA GPU availability
    - GPU memory (BERT requires ~6-8GB)
    - Disk space (estimate based on experiment count)
    """
    print("\n" + "=" * 70)
    print("🔍 VALIDATING SYSTEM RESOURCES")
    print("=" * 70)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA GPU not available")
        print("   BERT experiments require a CUDA-capable GPU")
        sys.exit(1)

    print(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s) detected")

    # Check GPU memory
    warnings = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB)")

        if mem_gb < 6:
            warnings.append(
                f"GPU {i} has only {mem_gb:.1f}GB memory - may cause OOM errors"
            )

    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
        print("   Consider reducing batch size or using a GPU with more memory")

    # Estimate disk space requirements
    n_methods = len(config["methods"])
    n_relabel = len(config["relabel_percentages"])
    n_keep = len(config["keep_ratios"])
    n_seeds = len(config["seeds"])
    total_experiments = n_methods * n_relabel * n_keep * n_seeds

    # Rough estimates:
    # - Training: ~200MB per experiment (model checkpoints)
    # - Influence: ~50MB per method
    # - Cleansing: ~10MB per configuration
    estimated_gb = total_experiments * 0.5  # 500MB per complete experiment

    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    stat = os.statvfs(save_dir)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

    print(f"\n💾 Disk Space:")
    print(f"  Available: {free_gb:.1f} GB")
    print(f"  Estimated required: {estimated_gb:.1f} GB")

    if free_gb < estimated_gb:
        print(f"\n❌ ERROR: Insufficient disk space")
        print(f"   Required: {estimated_gb:.1f} GB")
        print(f"   Available: {free_gb:.1f} GB")
        sys.exit(1)

    print("  ✓ Sufficient disk space available")

    print("\n✅ Resource validation passed")


def display_experiment_matrix(config: Dict, args: argparse.Namespace) -> None:
    """
    Display experiment matrix and ask for user confirmation.

    Shows:
    - Parameter dimensions
    - Total experiment count
    - Estimated time and resources
    """
    print("\n" + "=" * 70)
    print("📊 BERT RELABEL DETECTION EXPERIMENT MATRIX")
    print("=" * 70)

    print(f"\nDataset/Model:     {config['target']} (BERT on IMDB)")
    print(f"Training epochs:   {config.get('num_epochs', 5)} epochs per experiment")
    print(f"Learning rate:     {config.get('lr', 2e-5)}")
    print(f"Batch size:        {config.get('batch_size', 16)}")

    print(f"\nInfluence Methods: ({len(config['methods'])})")
    for method in config["methods"]:
        print(f"  - {method}")

    print(
        f"\nRelabel Ratios:    {config['relabel_percentages']} ({len(config['relabel_percentages'])})"
    )
    print(f"Keep Ratios:       {config['keep_ratios']} ({len(config['keep_ratios'])})")
    print(f"Seeds:             {config['seeds']} ({len(config['seeds'])})")

    # Calculate totals
    n_methods = len(config["methods"])
    n_relabel = len(config["relabel_percentages"])
    n_keep = len(config["keep_ratios"])
    n_seeds = len(config["seeds"])
    total = n_methods * n_relabel * n_keep * n_seeds

    print(f"\nExperiment Count:")
    print(
        f"  {n_methods} methods × {n_relabel} relabel × {n_keep} keep × {n_seeds} seeds = {total} total experiments"
    )

    # Estimate time (rough estimates based on BERT training)
    # Training: ~10-15min, Influence: ~5-10min, Cleansing: ~2-5min
    avg_time_per_experiment = 25  # minutes
    total_time_minutes = total * avg_time_per_experiment

    if args.gpus:
        n_gpus = len(args.gpus)
        parallel_time_hours = (total_time_minutes / n_gpus) / 60
        print(f"\nEstimated Time:")
        print(f"  Sequential:   {total_time_minutes / 60:.1f} hours (1 GPU)")
        print(
            f"  Parallel:     {parallel_time_hours:.1f} hours ({n_gpus} GPU{'s' if n_gpus > 1 else ''})"
        )
    else:
        print(f"\nEstimated Time:  {total_time_minutes / 60:.1f} hours")

    print(f"\nOutput Directory:  {config['save_dir']}")

    print("=" * 70)

    # Ask for confirmation unless --yes flag is set
    if not args.yes and not args.dry_run:
        response = input("\n❓ Continue with these experiments? [y/N]: ")
        if response.lower() not in ["y", "yes"]:
            print("❌ Experiment cancelled by user")
            sys.exit(0)


def run_bert_experiments(config: Dict, args: argparse.Namespace) -> None:
    """
    Execute grid search experiments using existing infrastructure.

    Calls scripts/run_influence_cleansing_grid.py with appropriate arguments.
    """
    print("\n" + "=" * 70)
    print("🚀 STARTING BERT RELABEL DETECTION EXPERIMENTS")
    print("=" * 70)

    # Build command for grid search script
    grid_script = Path(__file__).parent / "run_influence_cleansing_grid.py"
    if not grid_script.exists():
        raise FileNotFoundError(f"Grid search script not found: {grid_script}")

    command = [
        sys.executable,  # Use same Python interpreter
        str(grid_script),
        "--output-root",
        args.output_root,
        "--save-dir-prefix",
        config["save_dir"],
        "--target",
        config["target"],
        "--model",
        config["model"],
        "--log-level",
        args.log_level,
    ]

    # Add parameter lists
    command.extend(["--methods"] + config["methods"])
    command.extend(["--keep-ratios"] + [str(k) for k in config["keep_ratios"]])
    command.extend(
        ["--relabel-percentages"] + [str(r) for r in config["relabel_percentages"]]
    )
    command.extend(["--seeds"] + [str(s) for s in config["seeds"]])

    # Forward training hyperparameters so BERT-specific overrides are honored by
    # the training/cleansing subprocesses (previously these were displayed but
    # silently dropped).
    if config.get("num_epochs") is not None:
        command.extend(["--num-epoch", str(config["num_epochs"])])
    if config.get("lr") is not None:
        command.extend(["--lr", str(config["lr"])])
    if config.get("batch_size") is not None:
        command.extend(["--batch-size", str(config["batch_size"])])

    # Add GPUs
    if args.gpus:
        command.extend(["--gpus"] + args.gpus)

    # Add optional flags
    if args.dry_run:
        command.append("--dry-run")

    if args.skip_training:
        command.append("--skip-training")
        if args.existing_train_dir:
            command.extend(["--existing-train-dir", args.existing_train_dir])

    # Always run analysis afterward (unless explicitly skipped)
    if not args.skip_analysis:
        command.append("--run-analysis")

    print(f"\n📝 Executing command:")
    print(f"   {' '.join(command)}")
    print()

    # Execute grid search
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
        )
        print("\n✅ All experiments completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiments failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Experiments interrupted by user")
        sys.exit(130)


def analyze_results(config: Dict, args: argparse.Namespace) -> None:
    """
    Run post-experiment analysis on completed results.

    Calls scripts/analyze_precision_performance.py to aggregate results.
    """
    print("\n" + "=" * 70)
    print("📊 ANALYZING EXPERIMENT RESULTS")
    print("=" * 70)

    analysis_script = Path(__file__).parent / "analyze_precision_performance.py"
    if not analysis_script.exists():
        print(f"⚠️  Analysis script not found: {analysis_script}")
        print("   Skipping automated analysis")
        return

    # Resolve the analysis base dir the SAME way the grid and the analyzer do
    # (resolve_output_dir nests under outputs/), so --analyze-only READS where the
    # runs actually landed (outputs/result/...) and WRITES its reports there too.
    # Align the reports dir name with the grid's run_precision_analysis.
    base_dir = resolve_output_dir(args.output_root) if args.output_root else resolve_output_dir(None)
    output_dir = base_dir / "precision_analysis_results"

    command = [
        sys.executable,
        str(analysis_script),
        "--base_dir",
        str(base_dir),
        "--target",
        config["target"],
        "--model",
        config["model"],
        "--output_dir",
        str(output_dir),
        "--plot",
    ]

    print(f"\n📝 Running analysis:")
    print(f"   {' '.join(command)}")
    print()

    try:
        subprocess.run(command, check=True, text=True)
        print(f"\n✅ Analysis complete!")
        print(f"📈 Results saved to: {output_dir}/")
    except subprocess.CalledProcessError as e:
        # analyze_precision_performance.py exits non-zero when it finds nothing,
        # so PROPAGATE it -- otherwise --analyze-only would exit 0 having analyzed
        # nothing. (A blanket catch-and-continue here masked that.)
        print(f"\n❌ Analysis failed with exit code {e.returncode}", file=sys.stderr)
        print("   No results were analyzed; check that the runs completed.", file=sys.stderr)
        sys.exit(e.returncode)


def print_usage_guide(config: Dict, args: argparse.Namespace) -> None:
    """Print helpful guide for accessing results."""
    print("\n" + "=" * 70)
    print("📚 HOW TO ACCESS YOUR RESULTS")
    print("=" * 70)

    # Build actual output directory
    if args.output_root:
        output_path = Path(args.output_root)
    else:
        output_path = Path(".")

    print(f"""
The experiments have generated results in: {output_path}/

Key files for analysis:

1. **Cleansing Performance** (accuracy across epochs):
   {output_path}/*/cleansed_*_pct_performance_*.csv

   Columns: epoch, val_accuracy, val_loss, train_loss

2. **Poison Detection Metrics** (precision/recall/F1):
   {output_path}/*/relabel_overlap_*.csv

   Columns: epoch, precision, recall, f1_score, num_dropped, num_overlap_dropped

3. **Influence Scores** (per-sample influence):
   {output_path}/*/infl_*_relabel_*_pct_*.csv

   Columns: sample_idx, influence_epoch_0, influence_epoch_1, ...

4. **Aggregated Analysis**:
   {output_path}/analysis_results/  (if analysis was run)

Example: Load results in Python:
```python
import pandas as pd
import glob

# Load all cleansing performance files
files = glob.glob("{output_path}/*/cleansed_*_performance_*.csv")
dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)

# Analyze final epoch accuracy by method
final_epoch = combined[combined['epoch'] == {config.get("num_epochs", 5) - 1}]
final_epoch.groupby('method')['val_accuracy'].agg(['mean', 'std'])
```

For detailed documentation, see: README.md
    """)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate arguments
    if not args.analyze_only and not args.gpus:
        print("❌ ERROR: --gpus is required unless using --analyze-only")
        sys.exit(1)

    if args.skip_training and not args.existing_train_dir:
        print("❌ ERROR: --existing-train-dir is required when using --skip-training")
        sys.exit(1)

    # Build configuration
    try:
        config = get_bert_experiment_config(args)
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)

    # Analysis-only mode
    if args.analyze_only:
        analyze_results(config, args)
        print_usage_guide(config, args)
        return

    # Validate resources (unless skipped)
    if not args.skip_validation and not args.dry_run:
        validate_resources(config, args)

    # Display experiment matrix
    display_experiment_matrix(config, args)

    # Run experiments
    run_bert_experiments(config, args)

    # Print usage guide
    print_usage_guide(config, args)

    print("\n" + "=" * 70)
    print("🎉 BERT EXPERIMENTS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
