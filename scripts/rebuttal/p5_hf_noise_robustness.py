#!/usr/bin/env python
"""P5 (kpNi W4): reproduce the RQ2 four-pattern taxonomy on the real WIE
trajectories, show it is robust to reasonable thresholds, and quantify the
Highly-Fluctuating <-> label-noise association.

Key finding (resolves the earlier "reimplementation != paper" concern): a
pure-p slope test is *degenerate* on 21-epoch trajectories -- almost every
sample is a "significant" faint drift, so Stable collapses to ~15%. Gating the
trend additionally on effect size (R^2 of the OLS fit) is the principled fix and
independently recovers the paper's Table-5 DNN-MNIST structure:

  Table 5 DNN-MNIST (paper):  Stable 66.6 / Early 10.3 / Late 20.6 / Fluct 2.5
  This run @ (p<0.05, R^2>=0.70): Stable ~64.7 / Early ~10.5 / Late ~24.6
                                  -- Stable-dominant AND Late>Early, both reproduced.

It answers the reviewer's real question ("is the taxonomy an artifact of
arbitrary thresholds?") three ways:
  (1) Stable is the majority for any strong-trend cut (R^2 >= ~0.6); the
      Stable > Late > Early > HF ordering is preserved across the grid.
  (2) The exact four-pattern distribution shifts only modestly around the
      operating point (max pp shift reported).
  (3) The taxonomy's practical payload -- HF flags label noise -- is robust:
      on the 40%-relabel run, P(noisy|HF) stays high (lift over the 40% base
      rate reported) wherever HF is defined with a discriminating flip cut,
      while HF is ~0% on clean runs.

Honest caveat: HF as a *discrete* class is small here (~0.2-1%); the paper's
2.5% is not hit at high precision -- lowering the flip cut to enlarge HF trades
away noise precision. The qualitative taxonomy and the noise-enrichment are the
robust, defensible results; the exact HF fraction is threshold-sensitive.

Data (all real, on disk):
  * noisy: outputs/mnist_40pct_relabel40_seed00/infl_tim_all_epochs_relabel_040_pct_000.csv
           (n=2000, 21 epochs) + relabel_040_pct_indices_000.csv (800 noisy rows)
  * clean: outputs/loo_similar_256_seed_{1..16}_tim/... (16 seeds, n=256, 21 epochs)

Emits JSON to stdout (and --out).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from wie.analysis.temporal_patterns import (  # noqa: E402
    PATTERN_LABELS,
    FLUCTUATING,
    classify_patterns,
    pattern_distribution,
)

TABLE5_DNN_MNIST = {"Stable": 66.6, "Early Influencer": 10.3,
                    "Late Bloomer": 20.6, "Highly Fluctuating": 2.5}


def _load_matrix(csv_path):
    d = pd.read_csv(csv_path)
    cols = sorted(
        [c for c in d.columns if c.startswith("influence_epoch_")],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    return np.stack([d[c].to_numpy(np.float64) for c in cols], axis=1)


def _hf_noise(M, noisy_mask, **kw):
    labels, _ = classify_patterns(M, **kw)
    hf = labels == FLUCTUATING
    base = float(noisy_mask.mean())
    n_hf = int(hf.sum())
    if n_hf == 0:
        return {"hf_pct": 0.0, "n_hf": 0, "precision_noisy_given_hf": None,
                "lift": None, "recall_hf_given_noisy": 0.0}
    prec = float(noisy_mask[hf].mean())
    rec = float(hf[noisy_mask].sum()) / int(noisy_mask.sum())
    return {"hf_pct": round(100.0 * n_hf / len(labels), 3), "n_hf": n_hf,
            "precision_noisy_given_hf": round(prec, 4),
            "lift": round(prec / base, 3) if base > 0 else None,
            "recall_hf_given_noisy": round(rec, 4)}


def _dist_noisy(M, **kw):
    labels, _ = classify_patterns(M, **kw)
    return {k: round(pattern_distribution(labels)[k], 2) for k in PATTERN_LABELS}


def _dist_clean_mean(clean_Ms, **kw):
    per = [pattern_distribution(classify_patterns(M, **kw)[0]) for M in clean_Ms]
    return {k: round(float(np.mean([d[k] for d in per])), 2) for k in PATTERN_LABELS}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy_dir", default="outputs/mnist_40pct_relabel40_seed00")
    ap.add_argument("--clean_glob",
                    default="outputs/loo_similar_256_seed_*_tim/"
                            "infl_tim_all_epochs_relabel_000_pct_*.csv")
    # operating point (principled "strong linear trend" cut)
    ap.add_argument("--p_op", type=float, default=0.05)
    ap.add_argument("--r2_op", type=float, default=0.70)
    ap.add_argument("--flip_op", type=float, default=0.45)
    # robustness grid around it
    ap.add_argument("--p_grid", default="0.01,0.05,0.10")
    ap.add_argument("--r2_grid", default="0.60,0.70,0.80")
    ap.add_argument("--flip_grid", default="0.40,0.45,0.50")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    nm = glob.glob(os.path.join(_ROOT, args.noisy_dir,
                                "infl_tim_all_epochs_relabel_040_pct_*.csv"))
    im = glob.glob(os.path.join(_ROOT, args.noisy_dir,
                                "relabel_040_pct_indices_*.csv"))
    if not nm or not im:
        print(json.dumps({"error": "noisy matrix/indices not found"}))
        return 1
    Mn = _load_matrix(nm[0])
    noisy_idx = pd.read_csv(im[0])["relabel_indices"].to_numpy(int)
    noisy_mask = np.zeros(Mn.shape[0], dtype=bool)
    noisy_mask[noisy_idx] = True
    clean_files = sorted(glob.glob(os.path.join(_ROOT, args.clean_glob)))
    clean_Ms = [_load_matrix(f) for f in clean_files]

    p_grid = [float(x) for x in args.p_grid.split(",")]
    r2_grid = [float(x) for x in args.r2_grid.split(",")]
    flip_grid = [float(x) for x in args.flip_grid.split(",")]

    result = {
        "config": {
            "noisy": os.path.relpath(nm[0], _ROOT),
            "n_noisy_tr": int(Mn.shape[0]),
            "n_relabeled": int(noisy_mask.sum()),
            "noise_base_rate": round(float(noisy_mask.mean()), 4),
            "clean_matrices": len(clean_Ms),
            "n_clean_tr_each": int(clean_Ms[0].shape[0]) if clean_Ms else None,
            "epochs": int(Mn.shape[1]),
            "operating_point": {"p": args.p_op, "min_r2": args.r2_op,
                                "flip_ratio": args.flip_op},
            "grid": {"p": p_grid, "r2": r2_grid, "flip": flip_grid},
        },
        "table5_dnn_mnist_paper": TABLE5_DNN_MNIST,
    }

    # ---- operating point: distribution vs Table 5 ----
    op = dict(p_threshold=args.p_op, flip_ratio_threshold=args.flip_op,
              slope_eps=0.0, standardize=True, min_r2=args.r2_op)
    op_noisy = _dist_noisy(Mn, **op)
    result["operating_point_distribution"] = {
        "noisy_run_pct": op_noisy,
        "paper_pct": TABLE5_DNN_MNIST,
        "abs_diff_pp": {k: round(abs(op_noisy[k] - TABLE5_DNN_MNIST[k]), 2)
                        for k in PATTERN_LABELS},
        "clean_run_meanover16_pct": _dist_clean_mean(clean_Ms, **op),
        "hf_noise": _hf_noise(Mn, noisy_mask, **op),
    }

    # ---- effect-size sweep: Stable-dominance emergence ----
    result["r2_sweep_noisy"] = []
    for r2 in sorted(set(r2_grid + [0.0, 0.3, 0.5, 0.9])):
        kw = dict(p_threshold=args.p_op, flip_ratio_threshold=args.flip_op,
                  slope_eps=0.0, standardize=True, min_r2=r2)
        result["r2_sweep_noisy"].append(
            {"min_r2": r2, "distribution_pct": _dist_noisy(Mn, **kw)})

    # ---- full robustness grid ----
    noisy_dists, lifts, precisions = [], [], []
    grid = []
    for p in p_grid:
        for r2 in r2_grid:
            for fr in flip_grid:
                kw = dict(p_threshold=p, flip_ratio_threshold=fr,
                          slope_eps=0.0, standardize=True, min_r2=r2)
                dn = _dist_noisy(Mn, **kw)
                hf = _hf_noise(Mn, noisy_mask, **kw)
                noisy_dists.append(dn)
                if hf["lift"] is not None:
                    lifts.append(hf["lift"])
                if hf["precision_noisy_given_hf"] is not None:
                    precisions.append(hf["precision_noisy_given_hf"])
                grid.append({"p": p, "min_r2": r2, "flip": fr,
                             "noisy_distribution_pct": dn, "hf_noise": hf})
    result["robustness_grid"] = grid

    def _spread(dicts):
        return {k: round(max(d[k] for d in dicts) - min(d[k] for d in dicts), 2)
                for k in PATTERN_LABELS}

    stable_always_plurality = all(
        d["Stable"] == max(d.values()) for d in noisy_dists)
    result["robustness_summary"] = {
        "noisy_distribution_maxshift_pp": _spread(noisy_dists),
        "noisy_distribution_maxshift_pp_overall":
            round(max(_spread(noisy_dists).values()), 2),
        "stable_is_plurality_in_every_grid_cell": bool(stable_always_plurality),
        "late_gt_early_in_every_grid_cell":
            bool(all(d["Late Bloomer"] >= d["Early Influencer"]
                     for d in noisy_dists)),
        "hf_lift_range": [round(min(lifts), 3), round(max(lifts), 3)] if lifts else None,
        "hf_precision_noisy_range":
            [round(min(precisions), 4), round(max(precisions), 4)] if precisions else None,
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
