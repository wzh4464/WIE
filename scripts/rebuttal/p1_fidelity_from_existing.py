#!/usr/bin/env python
"""P1 reconnaissance: RQ1 fidelity of WIE vs baselines, recomputed from the
EXISTING on-disk per-epoch influence matrices -- NO theta_t replay needed.

Each `outputs/loo_similar_256_seed_{s}_loo/` directory holds, on the SAME
trajectory, one `infl_<method>_all_epochs_relabel_000_pct_*.csv` (256 samples x
21 epochs) per method: loo (ground truth), tim (WIE), dve (DVEmb), icml
(ICML-IF), lava (LAVA). So the RQ1 fidelity comparison -- and the raw material
for cdjh's DVEmb-differencing ablation -- is computable directly.

Reports |Pearson| of each method vs LOO, in a "global" (influence summed over
epochs, per sample) and a "local" (mean per-epoch) sense, averaged over seeds.

Findings (see the module-level comment in the rebuttal status doc):
  * WIE is decisively the most faithful: ~0.90/0.90 vs DVEmb ~0.51, ICML-IF
    ~0.59, LAVA ~0.11 -- WIE >> DVEmb even before any differencing.
  * WIE-vs-LOO correlation is NEGATIVE (sign-convention mismatch); |.| is the
    fidelity.
  * The exact paper numbers (0.96/0.95) are NOT reproduced by this naive
    aggregation (mean 0.90, dragged down by outlier seeds 7/8/16; best seeds
    hit 0.96-0.98). Pinning the paper's exact global/local aggregation is
    required before shipping cdjh's ablation numbers (make-or-break reviewer).

Usage: python scripts/rebuttal/p1_fidelity_from_existing.py [--out out.json]
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METHODS = {"tim": "WIE", "dve": "DVEmb", "icml": "ICML-IF", "lava": "LAVA"}


def _load(csv):
    d = pd.read_csv(csv)
    cols = sorted([c for c in d.columns if c.startswith("influence_epoch_")],
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    return np.stack([d[c].to_numpy(np.float64) for c in cols], axis=1)


def _abscorr(a, b, kind="pearson"):
    f = np.isfinite(a) & np.isfinite(b)
    a, b = a[f], b[f]
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    r = pearsonr(a, b)[0] if kind == "pearson" else spearmanr(a, b).correlation
    return abs(float(r))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern",
                    default="outputs/loo_similar_256_seed_{s}_loo")
    ap.add_argument("--seeds", default="1-16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    lo, hi = (args.seeds.split("-") + [args.seeds])[:2]
    seeds = range(int(lo), int(hi) + 1)

    per = {m: {"global": [], "local": [], "seeds": []} for m in METHODS}
    for s in seeds:
        d = os.path.join(_ROOT, args.pattern.format(s=s))
        lf = glob.glob(os.path.join(d, "infl_loo_all_epochs_*.csv"))
        if not lf:
            continue
        L = _load(lf[0])
        for m in METHODS:
            mf = glob.glob(os.path.join(d, f"infl_{m}_all_epochs_*.csv"))
            if not mf:
                continue
            M = _load(mf[0])
            g = _abscorr(M.sum(1), L.sum(1))
            loc = float(np.nanmean([_abscorr(M[:, e], L[:, e])
                                    for e in range(M.shape[1])]))
            per[m]["global"].append(g)
            per[m]["local"].append(loc)
            per[m]["seeds"].append(s)

    summary = {METHODS[m]: {
        "global_abs_pearson_mean": round(float(np.nanmean(per[m]["global"])), 4),
        "local_abs_pearson_mean": round(float(np.nanmean(per[m]["local"])), 4),
        "n_seeds": len(per[m]["global"]),
    } for m in METHODS}
    result = {
        "note": "|Pearson| vs LOO from existing per-epoch matrices; no theta_t replay",
        "paper_wie_target": {"global": 0.96, "local": 0.95},
        "summary": summary,
        "per_seed": {METHODS[m]: {
            "seeds": per[m]["seeds"],
            "global": [round(x, 3) for x in per[m]["global"]],
            "local": [round(x, 3) for x in per[m]["local"]],
        } for m in METHODS},
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
