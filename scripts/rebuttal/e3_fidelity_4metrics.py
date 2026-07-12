#!/usr/bin/env python
"""E3: 4-metric RQ1 fidelity table (EXPERIMENTS-R2).

For Arnoldi / GEX / TracIn / WIE on rq1_regen_s{1..4} (same trajectories, exact
LOO ground truth), compute GLOBAL and LOCAL fidelity under four metrics matching
the paper's Table 1 set: Pearson r, Spearman rho, Kendall tau, Jaccard@top-30%.
mean +/- std over seeds.

Score formats: arnoldi/gex = single `influence` column (static; broadcast across
epochs for local); tracin/wie/loo = per-epoch matrix (global = sum over epochs,
local = per-epoch matched). Correlations use |.| (sign-convention differs across
methods); Jaccard uses top-30% by |score| (most-influential set), sign-robust.

Writes artifacts/e3_fidelity_4metrics.json + a markdown table.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

SEEDS = [1, 2, 3, 4]
METHODS = {"arnoldi": "Arnoldi", "gex": "GEX", "tracin": "TracIn", "wie": "WIE"}
FILE = {"arnoldi": "infl_arnoldi_{s:03d}.csv", "gex": "infl_gex_{s:03d}.csv",
        "tracin": "infl_tracin_{s:03d}.csv", "wie": "infl_wie_all_epochs_{s:03d}.csv",
        "loo": "infl_loo_all_epochs_{s:03d}.csv"}


def load(path):
    d = pd.read_csv(path)
    cols = sorted([c for c in d.columns if c.startswith("influence_epoch_")],
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    if cols:
        return np.stack([d[c].to_numpy(np.float64) for c in cols], axis=1)  # (n, E)
    return d["influence"].to_numpy(np.float64)[:, None]                     # (n, 1) static


def _clean(a, b):
    f = np.isfinite(a) & np.isfinite(b)
    return a[f], b[f]


def corr(a, b, kind):
    a, b = _clean(a, b)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    if kind == "pearson":
        r = pearsonr(a, b)[0]
    elif kind == "spearman":
        r = spearmanr(a, b).correlation
    else:
        r = kendalltau(a, b).correlation
    return abs(float(r))


def jaccard_top(a, b, frac=0.30):
    a, b = _clean(a, b)
    if a.size < 3:
        return np.nan
    k = max(1, int(frac * a.size))
    sa = set(np.argsort(-np.abs(a))[:k]); sb = set(np.argsort(-np.abs(b))[:k])
    return len(sa & sb) / len(sa | sb)


def metrics(mvec, lvec):
    return {"pearson": corr(mvec, lvec, "pearson"), "spearman": corr(mvec, lvec, "spearman"),
            "kendall": corr(mvec, lvec, "kendall"), "jaccard": jaccard_top(mvec, lvec)}


def main():
    per = {m: {"global": {k: [] for k in ["pearson", "spearman", "kendall", "jaccard"]},
               "local": {k: [] for k in ["pearson", "spearman", "kendall", "jaccard"]}}
           for m in METHODS}
    for s in SEEDS:
        d = f"outputs/rq1_regen_s{s}"
        lf = os.path.join(d, FILE["loo"].format(s=s))
        if not os.path.exists(lf):
            continue
        L = load(lf); Lg = L.sum(1); E = L.shape[1]
        for m in METHODS:
            mf = os.path.join(d, FILE[m].format(s=s))
            if not os.path.exists(mf):
                continue
            M = load(mf)
            Mg = M.sum(1) if M.shape[1] > 1 else M[:, 0]
            gm = metrics(Mg, Lg)                              # global
            for k in gm:
                per[m]["global"][k].append(gm[k])
            # local: per-epoch matched (static method broadcasts its single column)
            loc = {k: [] for k in ["pearson", "spearman", "kendall", "jaccard"]}
            for e in range(E):
                me = M[:, e] if M.shape[1] > 1 else M[:, 0]
                em = metrics(me, L[:, e])
                for k in em:
                    loc[k].append(em[k])
            for k in loc:
                per[m]["local"][k].append(float(np.nanmean(loc[k])))

    def agg(vals):
        v = np.array(vals, float)
        return {"mean": round(float(np.nanmean(v)), 3), "std": round(float(np.nanstd(v)), 3)}

    summary = {METHODS[m]: {scope: {k: agg(per[m][scope][k]) for k in per[m][scope]}
                            for scope in ["global", "local"]} for m in METHODS}

    def cell(m, scope):
        d = summary[METHODS[m]][scope]
        return " / ".join(f"{d[k]['mean']:.2f}" for k in ["pearson", "spearman", "kendall", "jaccard"])

    md = ["| Method | Global: r / ρ / τ / Jaccard | Local: r / ρ / τ / Jaccard |", "|---|---|---|"]
    for m in ["arnoldi", "gex", "tracin", "wie"]:
        name = f"**{METHODS[m]}**" if m == "wie" else METHODS[m]
        md.append(f"| {name} | {cell(m, 'global')} | {cell(m, 'local')} |")
    table_md = "\n".join(md)

    out = {"experiment": "E3 RQ1 fidelity, 4 metrics", "seeds": SEEDS,
           "metrics": ["Pearson", "Spearman", "Kendall", "Jaccard@top-30%"],
           "note": "|.| for correlations; Jaccard top-30% by |score|; arnoldi/gex static "
                   "(single score, broadcast for local); global = sum-over-epochs vs LOO sum.",
           "gex_construction": "isotropic Gaussian around FINAL checkpoint only (see gex_if.py)",
           "summary_mean_std": summary, "table_markdown": table_md}
    os.makedirs("rebuttal/rebuttal-drafts/artifacts", exist_ok=True)
    json.dump(out, open("rebuttal/rebuttal-drafts/artifacts/e3_fidelity_4metrics.json", "w"), indent=2)
    print(table_md)
    print("wrote artifacts/e3_fidelity_4metrics.json")


if __name__ == "__main__":
    sys.path.insert(0, "src")
    main()
