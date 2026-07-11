#!/usr/bin/env python3
"""
Analyze precision performance for different influence methods on sentiment-bert experiments.

This script calculates precision performance across different:
- Methods: wie_all_epochs, icml_all_epochs, lava_all_epochs, dve_all_epochs
- Keep ratios: different percentages of samples to keep
- Relabel percentages: different levels of label noise

It finds the best performing configuration for wie_all_epochs.
"""

import os
import re
import sys
import json
import pandas as pd
import numpy as np
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Plotting is OPTIONAL. seaborn (and, in stripped envs, matplotlib) may not be
# installed. The analyzer's CORE -- CSV aggregation, precision analysis, reporting
# and the empty -> sys.exit(1) logic -- must run without them, so import them
# lazily and only skip plot generation when unavailable.
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _HAS_PLOTTING = True
except ImportError:
    plt = None  # type: ignore
    sns = None  # type: ignore
    _HAS_PLOTTING = False

# Make the src/ layout importable when this script is run by path
# (python scripts/foo.py) without the editable install (pixi run install).
try:
    import wie  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wie.utils.paths import resolve_output_dir
from wie.io.naming import INFL_TYPE_CANONICAL_ALIASES

# Path-token -> canonical method for the legacy-filename fallback. Historical
# (pre-rename) result dirs encode the method ONLY in the path token, e.g.
# cleansed_tim_all_epochs_090_pct/relabel_overlap_000.csv, so this is the only
# source of `method` for those rows -- without it they parse as None and get
# dropped. Legacy tim_* tokens are mapped to their canonical wie_* name so the
# reported `method` is uniform. Seeded from wie.io.naming's legacy->canonical
# map (tim_all_epochs, tim_last) plus the historical window variants
# tim_first/tim_middle (whose wie_* targets are no longer registered, but whose
# pre-rename output dirs may still exist on disk).
_PATH_METHOD_TOKENS = {
    "wie_all_epochs": "wie_all_epochs",
    "icml_all_epochs": "icml_all_epochs",
    "lava_all_epochs": "lava_all_epochs",
    "dve_all_epochs": "dve_all_epochs",
    **INFL_TYPE_CANONICAL_ALIASES,  # tim_all_epochs->wie_all_epochs, tim_last->wie_last
    "tim_first": "wie_first",
    "tim_middle": "wie_middle",
}


def _run_dir_matches(overlap_path: str, target: str, model: str) -> bool:
    """Whether the run that produced ``overlap_path`` is for ``target``/``model``.

    Isolation is by RUN METADATA, not path substrings: read the sibling
    ``global_info_*.json`` in the overlap file's run directory (written at train
    time with ``target`` and ``model``) and keep the row only if both match. This
    works for grouped runs, whose dirs (e.g. ``influence_cleansing_relabel05_
    seed00/``) lack the dataset/model tokens in the path but do have global_info,
    AND for non-grouped runs. Only when no usable global_info is present do we
    fall back to the legacy path-substring check.
    """
    run_dir = os.path.dirname(overlap_path)
    for gi in sorted(glob.glob(os.path.join(run_dir, "global_info_*.json"))):
        try:
            with open(gi) as f:
                info = json.load(f)
        except (OSError, ValueError):
            continue
        gi_target, gi_model = info.get("target"), info.get("model")
        if gi_target is not None and gi_model is not None:
            return (
                str(gi_target).lower() == str(target).lower()
                and str(gi_model).lower() == str(model).lower()
            )
    # No usable global_info -> fall back to the legacy path-substring check.
    path_l = overlap_path.lower()
    return target.lower() in path_l and model.lower() in path_l


def find_overlap_files(
    base_dir: str,
    target: str = "sentiment",
    model: str = "bert",
) -> List[str]:
    """Find relabel_overlap_*.csv files under ``base_dir`` for ``target``/``model``.

    Isolation is by run metadata (each run dir's global_info_*.json), so a shared
    ``base_dir`` (e.g. the repo-wide ``outputs/``) does not aggregate CSVs from
    other datasets/models, while grouped runs -- whose dirs lack target/model
    path tokens -- are still kept. See :func:`_run_dir_matches`.
    """
    pattern = os.path.join(base_dir, "**", f"relabel_overlap_*.csv")
    overlap_files = glob.glob(pattern, recursive=True)

    if target and model:
        overlap_files = [
            f for f in overlap_files if _run_dir_matches(f, target, model)
        ]

    return sorted(overlap_files)


def parse_file_path(file_path: str) -> Dict[str, Optional[str]]:
    """
    Parse experiment parameters from file path.

    Expected path structure might contain:
    - method type (wie_all_epochs, icml_all_epochs, etc.)
    - keep ratio
    - relabel percentage
    - seed
    """
    path_parts = Path(file_path).parts
    file_name = Path(file_path).stem

    info = {
        'method': None,
        'keep_ratio': None,
        'relabel_pct': None,
        'seed': None,
        'path': file_path
    }

    # Extract method / keep_ratio / seed from the filename.
    # New format (per-run, collision-free even when a directory is shared):
    #   relabel_overlap_{method}_{keep:03d}_pct_{seed:03d}.csv
    # Legacy format (seed only; method/keep_ratio come from the path below):
    #   relabel_overlap_{seed}.csv
    if file_name.startswith('relabel_overlap_'):
        remainder = file_name[len('relabel_overlap_'):]
        new_fmt = re.match(r'^(?P<method>.+)_(?P<keep>\d{3})_pct_(?P<seed>\d{3})$', remainder)
        if new_fmt:
            info['method'] = new_fmt.group('method')
            info['keep_ratio'] = int(new_fmt.group('keep'))
            info['seed'] = int(new_fmt.group('seed'))
        else:
            try:
                info['seed'] = int(remainder)
            except ValueError:
                pass

    # Look for method indicators in path (fallback for legacy filenames).
    # Legacy tim_* path tokens map to their canonical wie_* name so pre-rename
    # result dirs are not dropped (their method lives only in the path token).
    if info['method'] is None:
        for token, canonical in _PATH_METHOD_TOKENS.items():
            if token in file_path:
                info['method'] = canonical
                break

    # Look for keep ratio and relabel percentage in path
    for part in path_parts:
        # Look for keep ratio patterns (e.g., cleansed_wie_all_epochs_090_pct).
        # Only used as a fallback; the new filename already carries keep_ratio.
        if info['keep_ratio'] is None and 'cleansed_' in part and '_pct' in part:
            try:
                # Extract keep ratio from patterns like "cleansed_wie_all_epochs_090_pct"
                ratio_part = part.split('_')[-2]  # Should be "090"
                info['keep_ratio'] = int(ratio_part)
            except (ValueError, IndexError):
                pass

        # Look for relabel percentage in the directory path. Two layouts:
        #   legacy non-grouped: "relabel_010_pct"  -> relabel_(\d+)_pct
        #   new grouped dir:    "..._relabel05_seed00" -> relabel(\d+)
        # Only set once (first match wins), and prefer the explicit _pct form.
        if info['relabel_pct'] is None:
            legacy_match = re.search(r'relabel_(\d+)_pct', part)
            grouped_match = re.search(r'relabel(\d+)', part)
            if legacy_match:
                info['relabel_pct'] = int(legacy_match.group(1))
            elif grouped_match:
                info['relabel_pct'] = int(grouped_match.group(1))

    return info


def load_overlap_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load and validate overlap CSV data."""
    try:
        df = pd.read_csv(file_path)

        # Validate required columns
        required_cols = ['precision', 'recall', 'f1_score', 'epoch']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing required columns in {file_path}")
            return None

        # Filter out invalid values
        df = df.replace([np.inf, -np.inf], np.nan)

        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def calculate_summary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate summary metrics from overlap data."""
    if df.empty:
        return {'precision_mean': 0, 'precision_max': 0, 'precision_final': 0,
                'recall_mean': 0, 'recall_max': 0, 'recall_final': 0,
                'f1_mean': 0, 'f1_max': 0, 'f1_final': 0}

    metrics = {}

    # Precision metrics
    metrics['precision_mean'] = df['precision'].mean()
    metrics['precision_max'] = df['precision'].max()
    metrics['precision_final'] = df['precision'].iloc[-1] if len(df) > 0 else 0

    # Recall metrics
    metrics['recall_mean'] = df['recall'].mean()
    metrics['recall_max'] = df['recall'].max()
    metrics['recall_final'] = df['recall'].iloc[-1] if len(df) > 0 else 0

    # F1 metrics
    metrics['f1_mean'] = df['f1_score'].mean()
    metrics['f1_max'] = df['f1_score'].max()
    metrics['f1_final'] = df['f1_score'].iloc[-1] if len(df) > 0 else 0

    return metrics


def analyze_precision_performance(
    base_dir: str,
    target: str = "sentiment",
    model: str = "bert",
) -> pd.DataFrame:
    """Analyze precision performance across all experiments."""

    # Find all overlap files (isolated by run metadata to the target/model)
    overlap_files = find_overlap_files(base_dir, target, model)
    print(f"Found {len(overlap_files)} overlap files")

    results = []

    for file_path in overlap_files:
        # Parse experiment info
        exp_info = parse_file_path(file_path)

        # Load data
        df = load_overlap_data(file_path)
        if df is None:
            continue

        # Calculate metrics
        metrics = calculate_summary_metrics(df)

        # Combine info
        result = {**exp_info, **metrics}
        results.append(result)

    if not results:
        print("No valid results found!")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Clean up and filter results
    results_df = results_df.dropna(subset=['method', 'keep_ratio', 'relabel_pct'])

    return results_df


def find_best_tim_performance(results_df: pd.DataFrame, metric: str = 'precision_max') -> Tuple[Dict, pd.DataFrame]:
    """Find the best WIE performance configuration."""

    # Filter for WIE results
    tim_results = results_df[results_df['method'] == 'wie_all_epochs'].copy()

    if tim_results.empty:
        print("No WIE results found!")
        return {}, pd.DataFrame()

    # Group by configuration and calculate mean across seeds
    if 'seed' in tim_results.columns:
        tim_summary = tim_results.groupby(['keep_ratio', 'relabel_pct']).agg({
            metric: ['mean', 'std', 'count'],
            'precision_mean': 'mean',
            'recall_mean': 'mean',
            'f1_mean': 'mean'
        }).reset_index()
        tim_summary.columns = ['keep_ratio', 'relabel_pct', f'{metric}_mean', f'{metric}_std',
                              'seed_count', 'precision_mean', 'recall_mean', 'f1_mean']
    else:
        tim_summary = tim_results[['keep_ratio', 'relabel_pct', metric, 'precision_mean', 'recall_mean', 'f1_mean']].copy()

    # Find best configuration
    best_idx = tim_summary[f'{metric}_mean' if 'seed' in tim_results.columns else metric].idxmax()
    best_config = tim_summary.iloc[best_idx].to_dict()

    return best_config, tim_summary


def generate_comparison_report(results_df: pd.DataFrame, output_dir: str = None) -> str:
    """Generate a comprehensive comparison report."""

    report_lines = []
    report_lines.append("# Precision Performance Analysis Report")
    report_lines.append("=" * 50)
    report_lines.append("")

    # Overall statistics
    methods = results_df['method'].unique()
    report_lines.append(f"Methods analyzed: {', '.join(methods)}")
    report_lines.append(f"Total configurations: {len(results_df)}")
    report_lines.append("")

    # Method comparison
    report_lines.append("## Method Comparison (Mean Precision)")
    report_lines.append("-" * 30)

    for method in methods:
        method_data = results_df[results_df['method'] == method]
        if not method_data.empty:
            mean_precision = method_data['precision_mean'].mean()
            max_precision = method_data['precision_max'].max()
            report_lines.append(f"{method:20s}: Mean={mean_precision:.4f}, Max={max_precision:.4f}")

    report_lines.append("")

    # Best WIE configuration
    best_config, tim_summary = find_best_tim_performance(results_df, 'precision_max')

    if best_config:
        report_lines.append("## Best WIE Configuration")
        report_lines.append("-" * 25)
        report_lines.append(f"Keep Ratio: {best_config.get('keep_ratio', 'N/A')}%")
        report_lines.append(f"Relabel Percentage: {best_config.get('relabel_pct', 'N/A')}%")
        report_lines.append(f"Best Precision: {best_config.get('precision_max_mean', best_config.get('precision_max', 'N/A')):.4f}")
        if 'precision_max_std' in best_config:
            report_lines.append(f"Precision Std: ±{best_config['precision_max_std']:.4f}")
        report_lines.append("")

    # Detailed results table
    if not results_df.empty:
        report_lines.append("## Detailed Results")
        report_lines.append("-" * 20)

        # Create summary table
        summary_cols = ['method', 'keep_ratio', 'relabel_pct', 'precision_mean', 'precision_max', 'f1_mean']
        summary_df = results_df[summary_cols].round(4)

        report_lines.append(summary_df.to_string(index=False))
        report_lines.append("")

    report_text = "\n".join(report_lines)

    # Save report if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "precision_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {report_path}")

    return report_text


def create_visualizations(results_df: pd.DataFrame, output_dir: str = None):
    """Create visualizations of the results."""

    if not _HAS_PLOTTING:
        print("Plotting libraries unavailable; skipping visualizations.")
        return

    if results_df.empty:
        print("No data to visualize")
        return

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Method comparison boxplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Precision Performance Analysis', fontsize=16)

    # Precision comparison
    if len(results_df['method'].unique()) > 1:
        sns.boxplot(data=results_df, x='method', y='precision_max', ax=axes[0, 0])
        axes[0, 0].set_title('Max Precision by Method')
        axes[0, 0].tick_params(axis='x', rotation=45)

    # Keep ratio vs precision for WIE
    tim_data = results_df[results_df['method'] == 'wie_all_epochs']
    if not tim_data.empty:
        sns.scatterplot(data=tim_data, x='keep_ratio', y='precision_max',
                       hue='relabel_pct', ax=axes[0, 1])
        axes[0, 1].set_title('WIE: Keep Ratio vs Max Precision')

    # F1 score comparison
    if len(results_df['method'].unique()) > 1:
        sns.boxplot(data=results_df, x='method', y='f1_max', ax=axes[1, 0])
        axes[1, 0].set_title('Max F1 Score by Method')
        axes[1, 0].tick_params(axis='x', rotation=45)

    # Heatmap for WIE performance
    if not tim_data.empty and len(tim_data) > 1:
        # Create pivot table for heatmap
        heatmap_data = tim_data.pivot_table(
            values='precision_max',
            index='relabel_pct',
            columns='keep_ratio',
            aggfunc='mean'
        )
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', ax=axes[1, 1])
            axes[1, 1].set_title('WIE: Precision Heatmap')

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "precision_analysis_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze precision performance across influence methods")
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing experiment results")
    parser.add_argument("--target", type=str, default="sentiment",
                       help="Target dataset name (default: sentiment)")
    parser.add_argument("--model", type=str, default="bert",
                       help="Model name (default: bert)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for reports and plots")
    parser.add_argument("--metric", type=str, default="precision_max",
                       choices=["precision_max", "precision_mean", "f1_max", "f1_mean"],
                       help="Metric to use for finding best WIE performance")
    parser.add_argument("--plot", action="store_true", help="Generate visualizations")

    args = parser.parse_args()

    # Resolve --base_dir the SAME way runs' save_dirs are resolved, so both the
    # grid's run_precision_analysis and the BERT wrapper's --analyze-only search
    # where runs actually landed. resolve_output_dir nests relative paths under
    # outputs/ (e.g. "result" -> <repo>/outputs/result) and is idempotent on an
    # absolute path already under outputs/ (the grid passes an already-resolved
    # base_dir), so there is no outputs/outputs double-nesting.
    args.base_dir = str(resolve_output_dir(args.base_dir))

    print(f"Analyzing experiments in: {args.base_dir}")
    print(f"Target: {args.target}, Model: {args.model}")
    print("=" * 50)

    # Analyze performance
    results_df = analyze_precision_performance(args.base_dir, args.target, args.model)

    if results_df.empty:
        # Treat "no valid results" as a FAILURE (non-zero exit) so automated
        # grid/wrapper runs surface it instead of printing "analysis complete"
        # after analyzing nothing.
        print(
            f"ERROR: No valid results found under {args.base_dir}. "
            "Nothing was analyzed (check that the runs completed and wrote "
            "relabel_overlap_*.csv files under this directory).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded {len(results_df)} experiment results")
    print()

    # Generate report
    report = generate_comparison_report(results_df, args.output_dir)
    print(report)

    # Save detailed results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "detailed_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Detailed results saved to: {results_path}")

    # Create visualizations (optional; core analysis above already ran)
    if args.plot:
        if not _HAS_PLOTTING:
            print(
                "Skipping plot generation: matplotlib/seaborn not available "
                "(install seaborn to enable --plot). Core analysis is unaffected.",
                file=sys.stderr,
            )
        else:
            try:
                create_visualizations(results_df, args.output_dir)
            except Exception as e:
                print(f"Error creating visualizations: {e}")

    # Find and highlight best WIE performance
    best_config, tim_summary = find_best_tim_performance(results_df, args.metric)
    if best_config:
        print("\n" + "=" * 50)
        print("🏆 BEST WIE CONFIGURATION")
        print("=" * 50)
        print(f"Keep Ratio: {best_config.get('keep_ratio', 'N/A')}%")
        print(f"Relabel Percentage: {best_config.get('relabel_pct', 'N/A')}%")
        metric_key = f"{args.metric}_mean" if f"{args.metric}_mean" in best_config else args.metric
        print(f"Best {args.metric}: {best_config.get(metric_key, 'N/A'):.4f}")


if __name__ == "__main__":
    main()