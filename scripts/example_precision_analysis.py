#!/usr/bin/env python3
"""
Example script showing how to use the precision analysis tools.
This creates some test data and demonstrates the analysis workflow.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_test_data(base_dir: str = "test_outputs"):
    """Create sample overlap CSV files for testing the analysis."""

    # Test configuration
    methods = ['wie_all_epochs', 'icml_all_epochs', 'lava_all_epochs', 'dve_all_epochs']
    keep_ratios = [70, 80, 90, 95]
    relabel_pcts = [5, 10, 15, 20]
    seeds = [0, 1, 2]

    print(f"Creating test data in: {base_dir}")

    for method in methods:
        for keep_ratio in keep_ratios:
            for relabel_pct in relabel_pcts:
                for seed in seeds:
                    # Create directory structure
                    exp_dir = os.path.join(
                        base_dir,
                        "sentiment_bert_experiments",
                        f"relabel_{relabel_pct:03d}_pct",
                        f"cleansed_{method}_{keep_ratio:03d}_pct"
                    )
                    os.makedirs(exp_dir, exist_ok=True)

                    # Generate realistic-looking data
                    np.random.seed(seed + hash(method + str(keep_ratio) + str(relabel_pct)) % 1000)

                    # Create performance data that makes sense
                    # Higher keep ratios and lower relabel percentages should generally perform better
                    base_precision = 0.3 + (keep_ratio - 70) * 0.01 - (relabel_pct - 5) * 0.02
                    base_precision = max(0.1, min(0.9, base_precision))  # Clamp between 0.1 and 0.9

                    # Add some method-specific adjustments
                    method_bonus = {
                        'wie_all_epochs': 0.05,
                        'icml_all_epochs': 0.02,
                        'lava_all_epochs': 0.03,
                        'dve_all_epochs': 0.01
                    }
                    base_precision += method_bonus.get(method, 0)

                    # Generate epoch-wise data (5 epochs)
                    epochs = 5
                    data = []

                    for epoch in range(epochs):
                        # Precision generally improves over epochs
                        precision = base_precision + epoch * 0.02 + np.random.normal(0, 0.05)
                        precision = max(0.05, min(0.95, precision))  # Clamp

                        # Recall is usually related to precision but with some variation
                        recall = precision * (0.8 + np.random.normal(0, 0.1))
                        recall = max(0.05, min(0.95, recall))

                        # F1 score is harmonic mean
                        if precision + recall > 0:
                            f1_score = 2 * (precision * recall) / (precision + recall)
                        else:
                            f1_score = 0

                        # Create row data
                        row = {
                            'epoch': epoch,
                            'num_dropped': int(60000 * (100 - keep_ratio) / 100),  # Assuming ~60k samples
                            'num_kept': int(60000 * keep_ratio / 100),
                            'num_relabelled': int(60000 * relabel_pct / 100),
                            'num_overlap_dropped': int(60000 * (100 - keep_ratio) / 100 * precision),
                            'num_overlap_kept': int(60000 * relabel_pct / 100 * (1 - recall)),
                            'overlap_ratio_dropped': precision,
                            'overlap_ratio_kept': 1 - recall,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1_score
                        }
                        data.append(row)

                    # Save CSV file
                    df = pd.DataFrame(data)
                    csv_path = os.path.join(exp_dir, f"relabel_overlap_{seed:03d}.csv")
                    df.to_csv(csv_path, index=False)

    print(f"✅ Created test data for {len(methods)} methods, {len(keep_ratios)} keep ratios, {len(relabel_pcts)} relabel percentages, {len(seeds)} seeds")
    print(f"Total files: {len(methods) * len(keep_ratios) * len(relabel_pcts) * len(seeds)}")


def run_analysis_example():
    """Run the analysis on the test data."""

    # Create test data first
    create_test_data()

    # Import the analysis functions
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)

    try:
        from analyze_precision_performance import analyze_precision_performance, find_best_tim_performance, generate_comparison_report

        print("\n" + "="*50)
        print("Running Analysis Example")
        print("="*50)

        # Run analysis
        results_df = analyze_precision_performance("test_outputs", "sentiment", "bert")

        if results_df.empty:
            print("❌ No results found!")
            return

        print(f"📊 Analyzed {len(results_df)} experiment configurations")
        print(f"Methods found: {sorted(results_df['method'].unique())}")
        print(f"Keep ratios: {sorted(results_df['keep_ratio'].unique())}")
        print(f"Relabel percentages: {sorted(results_df['relabel_pct'].unique())}")

        # Generate report
        report = generate_comparison_report(results_df, "analysis_example_output")

        print("\n" + "="*50)
        print("ANALYSIS REPORT")
        print("="*50)
        print(report)

        # Find best WIE configuration
        best_config, tim_summary = find_best_tim_performance(results_df)

        if best_config:
            print("\n" + "🏆 BEST WIE CONFIGURATION 🏆")
            print("-" * 30)
            for key, value in best_config.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")

    except ImportError as e:
        print(f"❌ Could not import analysis functions: {e}")
        print("Make sure analyze_precision_performance.py is in the same directory")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Example precision analysis workflow")
    parser.add_argument("--create-data", action="store_true",
                       help="Only create test data, don't run analysis")
    parser.add_argument("--run-analysis", action="store_true",
                       help="Only run analysis (assumes test data exists)")
    parser.add_argument("--base-dir", type=str, default="test_outputs",
                       help="Base directory for test data")

    args = parser.parse_args()

    if args.create_data:
        create_test_data(args.base_dir)
    elif args.run_analysis:
        # Just run analysis part
        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(script_dir)

        from analyze_precision_performance import analyze_precision_performance, generate_comparison_report
        results_df = analyze_precision_performance(args.base_dir, "sentiment", "bert")
        report = generate_comparison_report(results_df, "analysis_example_output")
        print(report)
    else:
        # Full workflow
        run_analysis_example()