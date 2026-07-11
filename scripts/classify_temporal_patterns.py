#!/usr/bin/env python
"""Classify RQ2 temporal influence patterns from a wie_all_epochs CSV.

Reads a per-epoch window-level influence CSV (the ``wie_all_epochs`` output,
columns ``influence_epoch_*``), classifies each training sample's trajectory
into {Stable, Early Influencer, Late Bloomer, Highly Fluctuating}
(:mod:`wie.analysis.temporal_patterns`), and writes:

  - ``<out>/temporal_pattern_labels.csv``    -- per-sample slope/pvalue/flips/label
  - ``<out>/temporal_pattern_distribution.csv`` -- Table-5-style distribution
  - ``<out>/temporal_patterns.png`` (with --plot) -- Figure-3-style mean curves

Example:
    python scripts/classify_temporal_patterns.py \
        --input outputs/mnist_dnn/infl_wie_all_epochs_000.csv \
        --output-dir outputs/mnist_dnn/patterns --plot
"""

import argparse
import os
import sys

import pandas as pd

# Ensure `wie` is importable when run from a source checkout without install.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from wie.analysis.temporal_patterns import (  # noqa: E402
    PATTERN_LABELS,
    classify_patterns,
    pattern_distribution,
    load_influence_matrix,
    mean_trajectories,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input", required=True, help="Path to a wie_all_epochs influence CSV."
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: alongside --input).",
    )
    p.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="OLS slope p-value cutoff for a significant trend.",
    )
    p.add_argument(
        "--flip-ratio-threshold",
        type=float,
        default=0.5,
        help="Adjacent-epoch sign-flip fraction for Highly Fluctuating.",
    )
    p.add_argument(
        "--slope-eps",
        type=float,
        default=0.0,
        help="Minimum |slope| (standardized) for a trend.",
    )
    p.add_argument(
        "--no-standardize",
        action="store_true",
        help="Classify raw influence (skip per-epoch z-scoring).",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Also save a Figure-3-style mean-trajectory plot.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.input):
        print(f"ERROR: input CSV not found: {args.input}", file=sys.stderr)
        return 1
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    M = load_influence_matrix(args.input)
    n, E = M.shape
    print(f"Loaded influence matrix: {n} samples x {E} epochs from {args.input}")

    labels, stats = classify_patterns(
        M,
        p_threshold=args.p_threshold,
        flip_ratio_threshold=args.flip_ratio_threshold,
        slope_eps=args.slope_eps,
        standardize=not args.no_standardize,
    )

    labels_csv = os.path.join(out_dir, "temporal_pattern_labels.csv")
    stats.to_csv(labels_csv, index=False)
    print(f"Wrote per-sample labels: {labels_csv}")

    dist = pattern_distribution(labels, as_percent=True)
    counts = pattern_distribution(labels, as_percent=False)
    dist_df = pd.DataFrame(
        {
            "pattern": PATTERN_LABELS,
            "count": [counts[k] for k in PATTERN_LABELS],
            "percent": [round(dist[k], 2) for k in PATTERN_LABELS],
        }
    )
    dist_csv = os.path.join(out_dir, "temporal_pattern_distribution.csv")
    dist_df.to_csv(dist_csv, index=False)
    print(f"Wrote distribution: {dist_csv}")
    print("\nTemporal pattern distribution:")
    for k in PATTERN_LABELS:
        print(f"  {k:<20s} {counts[k]:>7d}  ({dist[k]:5.1f}%)")

    if args.plot:
        traj = mean_trajectories(M, labels, standardize=not args.no_standardize)
        if traj is None:
            print("No samples to plot.", file=sys.stderr)
        else:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(7, 4.5))
                for lab in traj.columns:
                    ax.plot(traj.index, traj[lab], marker="o", label=lab)
                ax.axhline(0.0, color="gray", lw=0.8, ls="--")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Standardized window influence")
                ax.set_title("Temporal influence patterns (mean trajectory per class)")
                ax.legend()
                fig.tight_layout()
                png = os.path.join(out_dir, "temporal_patterns.png")
                fig.savefig(png, dpi=150)
                plt.close(fig)
                print(f"Wrote plot: {png}")
            except Exception as e:  # pragma: no cover - plotting is optional
                print(f"WARNING: could not render plot: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
