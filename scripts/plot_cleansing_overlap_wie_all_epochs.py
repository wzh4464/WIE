#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, List, Optional, Tuple

# Global plotting style
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 28
plt.rcParams["axes.titlesize"] = 28
plt.rcParams["axes.labelsize"] = 40
plt.rcParams["xtick.labelsize"] = 26
plt.rcParams["ytick.labelsize"] = 26
plt.rcParams["legend.fontsize"] = 28

# 颜色列表（学术风格调色板）
# 颜色列表（用户指定顺序）
COLOR_LIST = [
    "#d95f02",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
]
COLOR_PALETTE = COLOR_LIST


def try_paths(base_dir: str, candidates: Iterable[str]) -> Optional[str]:
    """Return the first existing path among candidates joined to base_dir.

    If a candidate is absolute, use it as-is.
    """
    for p in candidates:
        full = p if os.path.isabs(p) else os.path.join(base_dir, p)
        if os.path.exists(full):
            return full
    return None


def find_file_recursive(
    name_hints: Iterable[str], start_dirs: Iterable[str]
) -> Optional[str]:
    """Search recursively for a CSV file whose name matches any of the hints.

    - name_hints: substrings that should all appear in the filename, e.g.,
      ["relabel", "indices", ".csv"]. Any one of the hint sets is sufficient.
    - start_dirs: directories to start searching from.
    Returns the first matching path found, else None.
    """
    # Normalize to list of lists (OR over outer, AND within inner)
    combo_hints: List[List[str]] = []
    for h in name_hints:
        if isinstance(h, (list, tuple)):
            combo_hints.append(list(h))
        else:
            combo_hints.append([str(h)])

    def matches(fname: str) -> bool:
        fname_low = fname.lower()
        for and_terms in combo_hints:
            if all(term.lower() in fname_low for term in and_terms):
                return True
        return False

    for root_dir in start_dirs:
        if not root_dir or not os.path.isdir(root_dir):
            continue
        for r, _dirs, files in os.walk(root_dir):
            for f in files:
                if matches(f):
                    return os.path.join(r, f)
    return None


def load_relabeled_indices(
    save_dir: str, relabel: int, seed: int, explicit_path: Optional[str] = None
) -> Tuple[set, str]:
    """Load relabeled indices as a Python set and return (set, path).

    Tries a variety of filename patterns and falls back to a recursive search.
    """
    if explicit_path:
        path = explicit_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Relabeled indices file not found: {path}")
    else:
        # Try common naming patterns (zero-padded and non-padded)
        candidates = [
            f"relabel_{int(relabel):03d}_pct_indices_{seed:03d}.csv",  # our generator
            f"relabel_{int(relabel)}_pct_indices_{seed:03d}.csv",
            f"relabeled_indices_{int(relabel):03d}_pct_{seed:03d}.csv",
            f"relabeled_indices_{int(relabel)}_pct_{seed:03d}.csv",
            f"relabeled_indices_{seed:03d}.csv",
            f"relabel_indices_{seed:03d}.csv",
            f"relabeled_indices_{int(relabel)}_pct.csv",
            f"relabeled_indices_{int(relabel):03d}_pct.csv",
            os.path.join(
                "..", f"relabeled_indices_{int(relabel):03d}_pct_{seed:03d}.csv"
            ),
        ]
        path = try_paths(save_dir, candidates)

        # Fallback: recursive search across likely roots
        if path is None:
            likely_roots = [
                save_dir,
                os.getcwd(),
                os.path.dirname(save_dir),
            ]
            # Also add any directory whose basename matches save_dir token
            save_dir_name = os.path.basename(os.path.normpath(save_dir))
            for r in [os.getcwd(), os.path.dirname(os.getcwd())]:
                cand = os.path.join(r, save_dir_name)
                if os.path.isdir(cand):
                    likely_roots.append(cand)

            path = find_file_recursive(
                [
                    ["relabel", "indices", ".csv"],
                    ["relabeled_indices", ".csv"],
                ],
                start_dirs=likely_roots,
            )

        if path is None:
            tried = candidates
            raise FileNotFoundError(
                f"Relabeled indices file not found under {save_dir}. Tried: {tried}"
            )

    df = pd.read_csv(path)
    if "relabel_indices" in df.columns:
        series = df["relabel_indices"]
    elif "relabeled_indices" in df.columns:
        series = df["relabeled_indices"]
    elif "index" in df.columns:
        series = df["index"]
    elif "idx" in df.columns:
        series = df["idx"]
    else:
        # fallback: first column
        series = df.iloc[:, 0]
    return set(series.tolist()), path


def load_single_influence(save_dir, relabel, seed, stem):
    # stem examples: 'infl_tracin', 'infl_sgd'
    candidates = [
        f"{stem}_{int(relabel):03d}_pct_{seed:03d}.csv",
        f"{stem}_{seed:03d}.csv",
        f"{stem}_{int(relabel)}_pct_{seed:03d}.csv",
        os.path.join("..", f"{stem}_{int(relabel):03d}_pct_{seed:03d}.csv"),
    ]
    path = try_paths(save_dir, candidates)
    if path is None:
        return None, None
    df = pd.read_csv(path)
    if "sample_idx" not in df.columns:
        df = df.copy()
        df["sample_idx"] = range(len(df))
    if "influence" not in df.columns:
        # if it is a different schema, bail
        return None, None
    return df[["sample_idx", "influence"]], path


def load_wie_all_epochs(
    save_dir: str, relabel: int, seed: int, explicit_path: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    if explicit_path:
        path = explicit_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"WIE-all-epochs CSV not found: {path}")
    else:
        # Try standard and fallbacks. Include the legacy tim_all_epochs prefix
        # so pre-existing (pre-rename) outputs remain discoverable.
        candidates = []
        for _tok in ("wie_all_epochs", "tim_all_epochs"):
            candidates.extend(
                [
                    f"infl_{_tok}_relabel_{int(relabel):03d}_pct_{seed:03d}.csv",
                    f"infl_{_tok}_relabel_{int(relabel)}_pct_{seed:03d}.csv",
                    f"infl_{_tok}_{seed:03d}.csv",
                    f"infl_{_tok}.csv",
                    os.path.join(
                        "..",
                        f"infl_{_tok}_relabel_{int(relabel):03d}_pct_{seed:03d}.csv",
                    ),
                ]
            )
        path = try_paths(save_dir, candidates)

        # Fallback: recursive search across likely roots
        if path is None:
            likely_roots = [
                save_dir,
                os.getcwd(),
                os.path.dirname(save_dir),
            ]
            save_dir_name = os.path.basename(os.path.normpath(save_dir))
            for r in [os.getcwd(), os.path.dirname(os.getcwd())]:
                cand = os.path.join(r, save_dir_name)
                if os.path.isdir(cand):
                    likely_roots.append(cand)

            path = find_file_recursive(
                [
                    ["infl_wie_all_epochs", ".csv"],
                    ["wie_all_epochs", ".csv"],
                    # Legacy (pre-rename) outputs
                    ["infl_tim_all_epochs", ".csv"],
                    ["tim_all_epochs", ".csv"],
                ],
                start_dirs=likely_roots,
            )

        if path is None:
            tried = candidates
            raise FileNotFoundError(
                f"WIE-all-epochs CSV not found under {save_dir}. Tried: {tried}"
            )

    df = pd.read_csv(path)
    if "sample_idx" not in df.columns:
        # if absent, assume default index order
        df = df.copy()
        df["sample_idx"] = range(len(df))
    # collect all epoch segment columns
    seg_cols = [c for c in df.columns if c.startswith("influence_segment_")]
    if not seg_cols:
        raise ValueError(
            f"No influence_segment_* columns found in {path}. Columns: {list(df.columns)}"
        )
    # natural order
    seg_cols.sort(key=lambda x: int(x.split("_")[-1]))
    return df[["sample_idx"] + seg_cols], path


def compute_overlap_curve(df_ranked, relabeled_set):
    is_relabeled = df_ranked["sample_idx"].isin(relabeled_set).to_numpy()
    cum = is_relabeled.cumsum()
    total = len(df_ranked)
    # Build dataframe with counts and percentages
    out = pd.DataFrame(
        {
            "Sample Count": range(1, total + 1),
            "Overlap Count": cum,
        }
    )
    out["Overlap Percentage"] = out["Overlap Count"] / out["Sample Count"] * 100.0
    return out


def plot_curves(curves, num_relabeled, title, out_path, palette=None):
    if palette is None:
        palette = COLOR_PALETTE
    plt.figure(figsize=(7, 7))  # 绘图
    # Plot each epoch curve
    for i, (label, df) in enumerate(curves):
        color = palette[i % len(palette)]
        plt.plot(
            df["Sample Count"],
            df["Overlap Count"],
            label=label,
            linewidth=3,
            color=color,
        )

    # Reference line: from (0,0) to (N, num_relabeled)
    if curves:
        N = curves[0][1]["Sample Count"].iloc[-1]
        plt.plot([0, N], [0, num_relabeled], "--", color="gray", label="Reference")

    plt.xlabel("Number of Training Samples Checked")
    plt.ylabel("Number of Mislabeled Samples Identified")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_main_comparison(tracin_curve, wie_last_curve, num_relabeled, title, out_path):
    plt.figure(figsize=(7, 7))  # 绘图
    # Colors: use provided list: pick blue for TracIn, orange for WIE Last if available
    color_tracin = COLOR_LIST[1] if len(COLOR_LIST) > 1 else "#377eb8"
    color_tim = COLOR_LIST[0] if len(COLOR_LIST) > 0 else "#d95f02"

    line_tracin = None
    line_tim = None
    if tracin_curve is not None:
        (line_tracin,) = plt.plot(
            tracin_curve["Sample Count"],
            tracin_curve["Overlap Count"],
            label="TracIn",
            linewidth=6,
            color=color_tracin,
            zorder=1,
        )
    if wie_last_curve is not None:
        (line_tim,) = plt.plot(
            wie_last_curve["Sample Count"],
            wie_last_curve["Overlap Count"],
            label="WIE",
            linewidth=6,
            color=color_tim,
            zorder=2,
        )
    # Reference
    N = None
    if tracin_curve is not None:
        N = tracin_curve["Sample Count"].iloc[-1]
    elif wie_last_curve is not None:
        N = wie_last_curve["Sample Count"].iloc[-1]
    if N is not None:
        plt.plot(
            [0, N],
            [0, num_relabeled],
            "--",
            color="gray",
            linewidth=4,
            label="_nolegend_",
        )

    plt.xlabel("No. of data checked")
    plt.ylabel("No. of data identified")
    plt.grid(True)
    # Legend order: WIE first, then TracIn
    handles = []
    if line_tim is not None:
        handles.append(line_tim)
    if line_tracin is not None:
        handles.append(line_tracin)
    if handles:
        plt.legend(handles=handles)
    plt.tight_layout()
    # 固定坐标范围（按需求）
    plt.xlim(0, 8000)
    plt.ylim(bottom=0)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot overlap curves for WIE-all-epochs experiment"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory containing influence CSVs",
    )
    parser.add_argument(
        "--relabel", type=int, required=True, help="Relabel percentage (e.g., 30)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed used in file names (default: 0)"
    )
    parser.add_argument(
        "--relabeled_path",
        type=str,
        default=None,
        help="Explicit path to relabeled indices CSV",
    )
    parser.add_argument(
        "--wie_all_epochs_path",
        type=str,
        default=None,
        help="Explicit path to WIE-all-epochs CSV",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="ascending",
        choices=["ascending", "descending"],
        help="Sort order for influence scores per epoch (default: ascending)",
    )
    parser.add_argument(
        "--compare_tracin_timlast",
        action="store_true",
        help="Also plot a main comparison with TracIn & WIE Last curves",
    )
    args = parser.parse_args()

    # Load inputs
    relabeled_set, relabeled_path = load_relabeled_indices(
        args.save_dir, args.relabel, args.seed, explicit_path=args.relabeled_path
    )
    tim_df, tim_path = load_wie_all_epochs(
        args.save_dir, args.relabel, args.seed, explicit_path=args.wie_all_epochs_path
    )

    print(
        f"Loaded relabeled indices from: {relabeled_path} (count={len(relabeled_set)})"
    )
    print(f"Loaded WIE-all-epochs from: {tim_path}")

    # Determine an output directory that exists (or create it)
    output_dir = args.save_dir
    if not os.path.isdir(output_dir):
        # Prefer the directory containing the loaded CSVs
        for d in [os.path.dirname(tim_path), os.path.dirname(relabeled_path)]:
            if d and os.path.isdir(d):
                output_dir = d
                break
    os.makedirs(output_dir, exist_ok=True)

    seg_cols = [c for c in tim_df.columns if c.startswith("influence_segment_")]
    curves = []
    for seg in seg_cols:
        df_ranked = tim_df.sort_values(by=seg, ascending=(args.order == "ascending"))[
            ["sample_idx"]
        ]
        curve = compute_overlap_curve(df_ranked, relabeled_set)
        label = f"Epoch {seg.split('_')[-1]} ({args.order})"
        curves.append((label, curve))

    # Plot
    out_png = os.path.join(
        output_dir,
        f"cleansing_overlap_wie_all_epochs_{args.relabel:03d}_pct_seed_{args.seed:03d}_{args.order}.png",
    )
    title = f"WIE-all-epochs Overlap (relabel {args.relabel}%, seed {args.seed})"
    plot_curves(curves, num_relabeled=len(relabeled_set), title=title, out_path=out_png)
    print(f"Saved figure to: {out_png}")

    # Optional: Main comparison plot using TracIn and WIE Last
    if args.compare_tracin_timlast:
        tracin_df, tracin_path = load_single_influence(
            args.save_dir, args.relabel, args.seed, stem="infl_tracin"
        )
        sgd_df, sgd_path = load_single_influence(
            args.save_dir, args.relabel, args.seed, stem="infl_sgd"
        )
        if tracin_df is None and sgd_df is None:
            print(
                "Warning: Neither TracIn nor WIE Last CSV found; skipping main comparison plot."
            )
        else:
            order_asc = args.order == "ascending"
            tracin_curve = None
            wie_last_curve = None
            if tracin_df is not None:
                df_ranked = tracin_df.sort_values(by="influence", ascending=order_asc)
                tracin_curve = compute_overlap_curve(df_ranked, relabeled_set)
                print(f"Loaded TracIn CSV: {tracin_path}")
            if sgd_df is not None:
                df_ranked = sgd_df.sort_values(by="influence", ascending=order_asc)
                wie_last_curve = compute_overlap_curve(df_ranked, relabeled_set)
                print(f"Loaded WIE Last CSV: {sgd_path}")

            out_cmp = os.path.join(
                output_dir,
                f"compare_tracin_timlast_overlap_{args.relabel:03d}_pct_seed_{args.seed:03d}_{args.order}.png",
            )
            title_cmp = f"Comparison (TracIn vs WIE Last) — relabel {args.relabel}% seed {args.seed}"
            plot_main_comparison(
                tracin_curve, wie_last_curve, len(relabeled_set), title_cmp, out_cmp
            )
            print(f"Saved comparison figure to: {out_cmp}")

            # Also save with conventional filename used by prior scripts
            alt_out = os.path.join(
                output_dir, f"cleansing_plot_{int(args.relabel)}_pct.png"
            )
            plot_main_comparison(
                tracin_curve, wie_last_curve, len(relabeled_set), title_cmp, alt_out
            )
            print(f"Saved conventional figure to: {alt_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in plot script: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
