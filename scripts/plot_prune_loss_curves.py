#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# Plot style consistent with other scripts
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 22
plt.rcParams["axes.titlesize"] = 26
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["xtick.labelsize"] = 22
plt.rcParams["ytick.labelsize"] = 22
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["figure.titlesize"] = 28


def find_first_existing(paths) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def search_recursive(start_dir: str, filename_substr: str) -> Optional[str]:
    if not os.path.isdir(start_dir):
        return None
    for r, _d, files in os.walk(start_dir):
        for f in files:
            if filename_substr in f:
                return os.path.join(r, f)
    return None


def load_pre_metrics(base_dir: str, relabel: int, seed: int) -> Optional[pd.DataFrame]:
    prefix = f"relabel_{int(relabel):03d}_pct_"
    candidates = [
        # Common patterns across experiments
        os.path.join(base_dir, f"{prefix}metrics_{seed:03d}.csv"),
        os.path.join(base_dir, f"metrics_{seed:03d}.csv"),
        # Some runs name it 'training_metrics'
        os.path.join(base_dir, f"{prefix}training_metrics_{seed:03d}.csv"),
        os.path.join(base_dir, f"training_metrics_{seed:03d}.csv"),
    ]
    path = find_first_existing(candidates)
    if path is None:
        # try recursive search
        path = search_recursive(
            base_dir, f"training_metrics_{seed:03d}.csv"
        ) or search_recursive(base_dir, f"metrics_{seed:03d}.csv")
    if path is None:
        return None
    df = pd.read_csv(path)
    # Expect columns: epoch, val_loss, val_accuracy, train_loss_avg
    return df


def load_post_metrics(
    base_dir: str, infl_type: str, keep_ratio: int, seed: int
) -> Optional[pd.DataFrame]:
    suffix = f"cleansed_{infl_type}_{int(keep_ratio):03d}_pct"
    candidates = [
        os.path.join(base_dir, f"{suffix}_performance_{seed:03d}.csv"),
        os.path.join(base_dir, f"tim_cleansing_performance_{seed:03d}.csv"),  # legacy
    ]
    path = find_first_existing(candidates)
    if path is None:
        # try recursive search
        path = search_recursive(base_dir, f"{suffix}_performance_{seed:03d}.csv")
        if path is None:
            path = search_recursive(
                base_dir, f"tim_cleansing_performance_{seed:03d}.csv"
            )
    if path is None:
        return None
    df = pd.read_csv(path)
    # Expect columns: epoch, test_accuracy, val_loss, train_loss
    return df


def plot_losses(pre_df: pd.DataFrame, post_df: pd.DataFrame, title: str, out_path: str):
    """Plot pre- vs post-prune loss curves with consistent styling."""
    plt.figure(figsize=(10, 7))
    pre_tr = _prepare_pre_df(pre_df)
    post_tr = _prepare_post_df(post_df)
    _plot_series(pre_tr, post_tr)
    _finalize_plot(title, out_path)


def _prepare_pre_df(pre_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if pre_df is None or pre_df.empty:
        return None
    df = pre_df.copy()
    df["epoch"] = df["epoch"].astype(int)
    return df[df["epoch"] >= 1]


def _prepare_post_df(post_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if post_df is None or post_df.empty:
        return None
    df = post_df.copy()
    df["epoch"] = df["epoch"].astype(int) + 1
    return df


def _plot_series(
    pre_tr: Optional[pd.DataFrame], post_tr: Optional[pd.DataFrame]
) -> None:
    if pre_tr is not None:
        if "train_loss_avg" in pre_tr.columns:
            plt.plot(
                pre_tr["epoch"],
                pre_tr["train_loss_avg"],
                label="Pre-prune Train Loss",
                color="#377eb8",
                linewidth=2.5,
            )
        if "val_loss" in pre_tr.columns:
            plt.plot(
                pre_tr["epoch"],
                pre_tr["val_loss"],
                label="Pre-prune Val Loss",
                color="#377eb8",
                linestyle="--",
                linewidth=2.5,
            )
    if post_tr is not None:
        if "train_loss" in post_tr.columns:
            plt.plot(
                post_tr["epoch"],
                post_tr["train_loss"],
                label="Post-prune Train Loss",
                color="#d95f02",
                linewidth=2.5,
            )
        if "val_loss" in post_tr.columns:
            plt.plot(
                post_tr["epoch"],
                post_tr["val_loss"],
                label="Post-prune Val Loss",
                color="#d95f02",
                linestyle="--",
                linewidth=2.5,
            )


def _finalize_plot(title: str, out_path: str) -> None:
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot pre- vs post-prune loss curves")
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing metrics/performance CSVs (e.g., experiment/sentiment_bert_wie)",
    )
    parser.add_argument(
        "--relabel",
        type=int,
        required=True,
        help="Relabel percent used in training (e.g., 30)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--infl_type",
        type=str,
        default="wie_all_epochs",
        choices=[
            "wie_all_epochs",
            "sgd",
            "nohess",
            "wie_last",
            "true",
            "lie",
            "icml",
            "tracin",
            "td_influence",
        ],
        help="Influence type used for pruning",
    )
    parser.add_argument(
        "--keep_ratio",
        type=int,
        required=True,
        help="Keep ratio used during pruning (e.g., 90)",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Optional explicit output PNG path"
    )

    args = parser.parse_args()

    base_dir = args.dir
    relabel = args.relabel
    seed = args.seed
    infl_type = args.infl_type
    keep_ratio = args.keep_ratio

    pre_df = load_pre_metrics(base_dir, relabel, seed)
    if pre_df is None:
        print(
            f"Warning: Could not find pre-prune metrics in {base_dir} (looked for relabel_{relabel:03d}_pct_metrics_{seed:03d}.csv)."
        )
    post_df = load_post_metrics(base_dir, infl_type, keep_ratio, seed)
    if post_df is None:
        print(
            f"Warning: Could not find post-prune performance in {base_dir} (looked for cleansed_{infl_type}_{keep_ratio:03d}_pct_performance_{seed:03d}.csv)."
        )

    if pre_df is None and post_df is None:
        print("Error: No data found to plot.")
        return

    title = f"Loss Curves — relabel {relabel}%, keep {keep_ratio}% (seed {seed})"
    out_path = args.out
    if out_path is None:
        out_path = os.path.join(
            base_dir,
            f"loss_curves_prune_vs_full_relabel_{relabel:03d}_keep_{keep_ratio:03d}_seed_{seed:03d}.png",
        )

    plot_losses(pre_df, post_df, title, out_path)


if __name__ == "__main__":
    main()
