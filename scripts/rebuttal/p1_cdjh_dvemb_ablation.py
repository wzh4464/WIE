#!/usr/bin/env python
"""P1 / cdjh ablation: is WIE just "DVEmb applied to arbitrary checkpoint pairs"?

Recomputed on identical footing (same 16 seeds, same trajectories, same LOO
ground truth), |Pearson| fidelity. Two complementary comparisons:

  (A) All-epochs fidelity vs LOO (global = influence summed over epochs per
      sample; local = mean per-epoch): WIE ≫ DVEmb — the base estimator cdjh
      claims WIE reduces to is far less faithful on identical data.

  (B) DVEmb-DIFFERENCING across window endpoints (dve[t2]-dve[t1]) vs the
      window-LOO ground truth (loo[t2]-loo[t1]): reported as context only. NOTE
      the window-level comparison is NOT clean: WIE's native window fidelity is
      itself mixed on the small validation runs (strong early ~0.86-0.94, weak
      late ~0.2-0.34), so the window-difference numbers do not cleanly separate
      the methods. The DEFENSIBLE cdjh evidence is (A), the all-epochs gap.

IMPORTANT (integrity): these are RELATIVE numbers recomputed on identical
footing with |Pearson|; the protocol is not guaranteed identical to the paper's
Table 1 aggregation, so absolute values may differ from the published 0.96/0.95.
The claim supported here is the all-epochs *gap* (WIE >> DVEmb), not the absolute
WIE value and NOT a window-level ranking.

Usage: python scripts/rebuttal/p1_cdjh_dvemb_ablation.py [--out out.json]
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WINDOWS = {"early": (0, 7), "middle": (7, 14), "late": (14, 20), "full": (0, 20)}


def _load(csv):
    d = pd.read_csv(csv)
    cols = sorted([c for c in d.columns if c.startswith("influence_epoch_")],
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    return np.stack([d[c].to_numpy(np.float64) for c in cols], axis=1)


def _ac(a, b):
    f = np.isfinite(a) & np.isfinite(b)
    a, b = a[f], b[f]
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return abs(float(pearsonr(a, b)[0]))


def _scores(csv):
    d = pd.read_csv(csv)
    cols = [c for c in d.columns if "influence" in c.lower()]
    return d[cols[0]].to_numpy(np.float64) if cols else None


def _wie_native_window():
    """Honest per-window |Pearson| of WIE native windows vs true_* window-LOO,
    measured on the small val_mnist_dnn runs (strong early, weak late)."""
    out = {}
    for win in ["first", "middle", "last"]:
        vals = []
        for base in sorted(glob.glob(os.path.join(_ROOT, "outputs/val_mnist_dnn_s*"))):
            wf = glob.glob(os.path.join(base, f"infl_wie_{win}_*.csv"))
            tf = glob.glob(os.path.join(base, f"infl_true_{win}_*.csv"))
            if wf and tf:
                w, t = _scores(wf[0]), _scores(tf[0])
                if w is not None and t is not None and len(w) == len(t):
                    vals.append(_ac(w, t))
        out[win] = round(float(np.nanmean(vals)), 3) if vals else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/loo_similar_256_seed_{s}_loo")
    ap.add_argument("--seeds", default="1-16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    lo, hi = args.seeds.split("-")
    seeds = range(int(lo), int(hi) + 1)

    allep = {m: {"global": [], "local": []} for m in ["tim", "dve", "icml", "lava"]}
    dvediff = {w: [] for w in WINDOWS}
    for s in seeds:
        d = os.path.join(_ROOT, args.dir.format(s=s))
        lf = glob.glob(os.path.join(d, "infl_loo_all_epochs_*.csv"))
        if not lf:
            continue
        L = _load(lf[0])
        for m in allep:
            mf = glob.glob(os.path.join(d, f"infl_{m}_all_epochs_*.csv"))
            if not mf:
                continue
            M = _load(mf[0])
            allep[m]["global"].append(_ac(M.sum(1), L.sum(1)))
            allep[m]["local"].append(
                float(np.nanmean([_ac(M[:, e], L[:, e]) for e in range(M.shape[1])])))
        vf = glob.glob(os.path.join(d, "infl_dve_all_epochs_*.csv"))
        if vf:
            V = _load(vf[0])
            for w, (a, b) in WINDOWS.items():
                dvediff[w].append(_ac(V[:, b] - V[:, a], L[:, b] - L[:, a]))

    lab = {"tim": "WIE", "dve": "DVEmb", "icml": "ICML-IF", "lava": "LAVA"}
    result = {
        "note": "|Pearson| vs LOO, recomputed identical footing; gap is the claim, "
                "not absolute values (paper Table 1 protocol may differ).",
        "A_all_epochs_fidelity_vs_LOO": {
            lab[m]: {"global": round(float(np.nanmean(allep[m]["global"])), 3),
                     "local": round(float(np.nanmean(allep[m]["local"])), 3)}
            for m in allep},
        "B_dvemb_differencing_window_fidelity_vs_windowLOO_CONTEXT_ONLY": {
            w: round(float(np.nanmean(v)), 3) for w, v in dvediff.items()},
        "WIE_native_window_fidelity_measured": _wie_native_window(),
        "caveat_window_level": "Window-level comparison does NOT cleanly favor "
            "WIE (native last-window fidelity is weak on small val runs); the "
            "defensible cdjh evidence is (A), the all-epochs gap.",
        "n_seeds": 16,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
