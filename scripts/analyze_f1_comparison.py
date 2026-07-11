#!/usr/bin/env python3
"""
F1 Score Comparison Analysis for Influence Methods

This script specifically analyzes F1 scores across different influence methods,
keep ratios, and relabel percentages to find the best performing configurations.
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import the existing analysis functions
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

try:
    from analyze_precision_performance import analyze_precision_performance, find_best_tim_performance
except ImportError:
    print("Warning: Could not import analyze_precision_performance. Some functions may be limited.")


def analyze_f1_performance(results_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze F1 score performance specifically."""
    if results_df.empty:
        return pd.DataFrame()

    # Focus on F1 metrics
    f1_cols = ['method', 'keep_ratio', 'relabel_pct', 'seed', 'f1_mean', 'f1_max', 'f1_final']
    f1_df = results_df[f1_cols].copy()

    # Calculate summary statistics by configuration (method, keep_ratio, relabel_pct)
    f1_summary = f1_df.groupby(['method', 'keep_ratio', 'relabel_pct']).agg({
        'f1_mean': ['mean', 'std', 'count'],
        'f1_max': ['mean', 'std', 'count'],
        'f1_final': ['mean', 'std', 'count']
    }).reset_index()

    # Flatten column names
    f1_summary.columns = [
        'method', 'keep_ratio', 'relabel_pct',
        'f1_mean_avg', 'f1_mean_std', 'f1_mean_count',
        'f1_max_avg', 'f1_max_std', 'f1_max_count',
        'f1_final_avg', 'f1_final_std', 'f1_final_count'
    ]

    return f1_summary


def find_best_configurations(f1_summary: pd.DataFrame, top_k: int = 5) -> dict:
    """Find top k configurations for each metric."""

    results = {}

    metrics = ['f1_mean_avg', 'f1_max_avg', 'f1_final_avg']
    metric_names = ['F1 Mean', 'F1 Max', 'F1 Final']

    for metric, name in zip(metrics, metric_names):
        if metric in f1_summary.columns:
            top_configs = f1_summary.nlargest(top_k, metric)
            results[name] = top_configs[['method', 'keep_ratio', 'relabel_pct', metric, metric.replace('_avg', '_std')]]

    return results


def create_f1_comparison_plots(results_df: pd.DataFrame, output_dir: str = None):
    """Create comprehensive F1 score comparison plots."""

    if results_df.empty:
        print("No data available for plotting")
        return

    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('F1 Score Performance Comparison Across Methods', fontsize=16)

    # 1. Method comparison boxplot for F1 max
    if len(results_df['method'].unique()) > 1:
        sns.boxplot(data=results_df, x='method', y='f1_max', ax=axes[0, 0])
        axes[0, 0].set_title('F1 Max Score by Method')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylabel('F1 Max Score')

    # 2. Keep ratio vs F1 score by method
    sns.boxplot(data=results_df, x='keep_ratio', y='f1_max', hue='method', ax=axes[0, 1])
    axes[0, 1].set_title('F1 Max Score by Keep Ratio and Method')
    axes[0, 1].set_ylabel('F1 Max Score')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 3. Relabel percentage vs F1 score
    sns.boxplot(data=results_df, x='relabel_pct', y='f1_max', hue='method', ax=axes[1, 0])
    axes[1, 0].set_title('F1 Max Score by Relabel Percentage and Method')
    axes[1, 0].set_ylabel('F1 Max Score')
    axes[1, 0].set_xlabel('Relabel Percentage')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 4. Heatmap for WIE method
    tim_data = results_df[results_df['method'] == 'wie_all_epochs'] if 'wie_all_epochs' in results_df['method'].values else results_df[results_df['method'] == results_df['method'].iloc[0]]

    if not tim_data.empty and len(tim_data) > 1:
        # Create pivot table for heatmap
        heatmap_data = tim_data.pivot_table(
            values='f1_max',
            index='relabel_pct',
            columns='keep_ratio',
            aggfunc='mean'
        )

        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1, 1])
            method_name = tim_data['method'].iloc[0]
            axes[1, 1].set_title(f'{method_name}: F1 Max Heatmap')
            axes[1, 1].set_ylabel('Relabel Percentage')
            axes[1, 1].set_xlabel('Keep Ratio')

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "f1_comparison_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"F1 comparison plots saved to: {plot_path}")

    plt.show()


def create_method_comparison_table(f1_summary: pd.DataFrame) -> pd.DataFrame:
    """Create a comprehensive method comparison table."""

    if f1_summary.empty:
        return pd.DataFrame()

    # Calculate overall method performance
    method_summary = f1_summary.groupby('method').agg({
        'f1_mean_avg': ['mean', 'max'],
        'f1_max_avg': ['mean', 'max'],
        'f1_final_avg': ['mean', 'max']
    }).reset_index()

    # Flatten column names
    method_summary.columns = [
        'method',
        'f1_mean_overall_avg', 'f1_mean_overall_max',
        'f1_max_overall_avg', 'f1_max_overall_max',
        'f1_final_overall_avg', 'f1_final_overall_max'
    ]

    # Sort by overall F1 max average
    method_summary = method_summary.sort_values('f1_max_overall_avg', ascending=False)

    return method_summary


def generate_f1_report(results_df: pd.DataFrame, output_dir: str = None) -> str:
    """Generate comprehensive F1 score analysis report."""

    report_lines = []
    report_lines.append("# F1 Score Performance Analysis Report")
    report_lines.append("=" * 50)
    report_lines.append()

    if results_df.empty:
        report_lines.append("No data available for analysis.")
        return "\n".join(report_lines)

    # Basic statistics
    methods = sorted(results_df['method'].unique())
    keep_ratios = sorted(results_df['keep_ratio'].unique())
    relabel_pcts = sorted(results_df['relabel_pct'].unique())

    report_lines.append(f"**Dataset Coverage:**")
    report_lines.append(f"- Methods: {len(methods)} ({', '.join(methods)})")
    report_lines.append(f"- Keep Ratios: {len(keep_ratios)} ({', '.join(map(str, keep_ratios))})")
    report_lines.append(f"- Relabel Percentages: {len(relabel_pcts)} ({', '.join(map(str, relabel_pcts))})")
    report_lines.append(f"- Total Configurations: {len(results_df)}")
    report_lines.append()

    # Analyze F1 performance
    f1_summary = analyze_f1_performance(results_df)

    if not f1_summary.empty:
        # Method comparison
        method_comparison = create_method_comparison_table(f1_summary)

        report_lines.append("## Overall Method Performance (F1 Scores)")
        report_lines.append("-" * 40)

        if not method_comparison.empty:
            for idx, row in method_comparison.iterrows():
                method = row['method']
                avg_f1 = row['f1_max_overall_avg']
                max_f1 = row['f1_max_overall_max']
                report_lines.append(f"**{method}:**")
                report_lines.append(f"  - Average F1 Max: {avg_f1:.4f}")
                report_lines.append(f"  - Best F1 Max: {max_f1:.4f}")
                report_lines.append()

        # Best configurations
        best_configs = find_best_configurations(f1_summary, top_k=5)

        for metric_name, top_configs in best_configs.items():
            report_lines.append(f"## Top 5 Configurations by {metric_name}")
            report_lines.append("-" * (25 + len(metric_name)))

            if not top_configs.empty:
                for idx, row in top_configs.iterrows():
                    method = row['method']
                    keep_ratio = row['keep_ratio']
                    relabel_pct = row['relabel_pct']
                    score = row.iloc[3]  # The metric value
                    std = row.iloc[4] if len(row) > 4 else None

                    report_lines.append(f"{idx+1}. **{method}** (Keep: {keep_ratio}%, Relabel: {relabel_pct}%)")
                    report_lines.append(f"   Score: {score:.4f}" + (f" (±{std:.4f})" if std is not None and not pd.isna(std) else ""))

                report_lines.append()

        # Best WIE configuration specifically
        tim_results = results_df[results_df['method'] == 'wie_all_epochs']
        if not tim_results.empty:
            best_tim_f1 = tim_results.loc[tim_results['f1_max'].idxmax()]
            report_lines.append("## 🏆 Best WIE Configuration (F1 Max)")
            report_lines.append("-" * 35)
            report_lines.append(f"- **Method:** {best_tim_f1['method']}")
            report_lines.append(f"- **Keep Ratio:** {best_tim_f1['keep_ratio']}%")
            report_lines.append(f"- **Relabel Percentage:** {best_tim_f1['relabel_pct']}%")
            report_lines.append(f"- **Seed:** {best_tim_f1['seed']}")
            report_lines.append(f"- **F1 Max Score:** {best_tim_f1['f1_max']:.4f}")
            report_lines.append(f"- **F1 Mean Score:** {best_tim_f1['f1_mean']:.4f}")
            report_lines.append(f"- **F1 Final Score:** {best_tim_f1['f1_final']:.4f}")
            report_lines.append()

    report_text = "\n".join(report_lines)

    # Save report
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "f1_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"F1 analysis report saved to: {report_path}")

        # Save detailed F1 summary
        if not f1_summary.empty:
            summary_path = os.path.join(output_dir, "f1_summary_table.csv")
            f1_summary.to_csv(summary_path, index=False)
            print(f"F1 summary table saved to: {summary_path}")

    return report_text


def main():
    parser = argparse.ArgumentParser(description="Analyze F1 score performance across influence methods")
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing experiment results")
    parser.add_argument("--target", type=str, default="sentiment",
                       help="Target dataset name (default: sentiment)")
    parser.add_argument("--model", type=str, default="bert",
                       help="Model name (default: bert)")
    parser.add_argument("--output_dir", type=str, default="f1_analysis_results",
                       help="Output directory for reports and plots")
    parser.add_argument("--plot", action="store_true", help="Generate F1 comparison plots")

    args = parser.parse_args()

    print(f"🔍 F1 Score Analysis")
    print(f"Analyzing experiments in: {args.base_dir}")
    print(f"Target: {args.target}, Model: {args.model}")
    print("=" * 50)

    # Load data using existing analysis function
    try:
        results_df = analyze_precision_performance(args.base_dir, args.target, args.model)
    except NameError:
        print("❌ Could not load analyze_precision_performance. Please run from scripts directory.")
        return

    if results_df.empty:
        print("❌ No valid results found!")
        return

    print(f"📊 Loaded {len(results_df)} experiment results")
    print()

    # Generate F1-focused analysis
    report = generate_f1_report(results_df, args.output_dir)
    print(report)

    # Create plots if requested
    if args.plot:
        try:
            create_f1_comparison_plots(results_df, args.output_dir)
        except ImportError:
            print("❌ Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            print(f"❌ Error creating plots: {e}")

    # Summary
    print("\n" + "="*50)
    print("📋 ANALYSIS SUMMARY")
    print("="*50)

    methods = results_df['method'].unique()
    best_f1_overall = results_df.loc[results_df['f1_max'].idxmax()]

    print(f"🏆 **Best Overall F1 Score:** {best_f1_overall['f1_max']:.4f}")
    print(f"   Method: {best_f1_overall['method']}")
    print(f"   Keep Ratio: {best_f1_overall['keep_ratio']}%")
    print(f"   Relabel Percentage: {best_f1_overall['relabel_pct']}%")
    print()

    print(f"📊 **Method Rankings by Average F1 Max:**")
    method_rankings = results_df.groupby('method')['f1_max'].mean().sort_values(ascending=False)
    for rank, (method, avg_f1) in enumerate(method_rankings.items(), 1):
        print(f"   {rank}. {method}: {avg_f1:.4f}")

    print(f"\n📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()