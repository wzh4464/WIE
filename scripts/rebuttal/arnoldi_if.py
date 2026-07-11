#!/usr/bin/env python
"""Arnoldi-IF baseline (Schioppa et al., 2022) for the RQ1 fidelity comparison.

Standard influence function accelerated via Arnoldi iteration: build a Krylov
subspace with Hessian-vector products on the final model, extract the top-k
Ritz pairs (approximate Hessian eigenpairs), and score

    infl(z_i) = sum_k (1/lambda_k) (u_k . g_val)(u_k . g_i)

a global final-model estimator with no notion of training window -- so it is
expected to be competitive globally but weak in the local (per-epoch/window)
setting, exactly the point of the comparison.

Reuses the calculator infrastructure (IcmlAdapterCalculator) for data/model/loss
loading, then adds the Arnoldi machinery. Writes per-sample scores and reports
|Pearson| vs LOO (global = summed over epochs, local = mean per-epoch).

Usage: python scripts/rebuttal/arnoldi_if.py --save_dir rq1_regen_s1 --seed 1 \
         --m 60 --k 20 [--out out.json]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from wie.infl.adapters import IcmlAdapterCalculator  # noqa: E402
from wie.infl.core import load_final_model  # noqa: E402
from wie.training.dataset_config import DATASET_NETWORK_CONFIG  # noqa: E402


def _flat(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def _unflat(v, ref):
    out, i = [], 0
    for r in ref:
        n = r.numel()
        out.append(v[i:i + n].view_as(r))
        i += n
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="rq1_regen_s1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--target", default="mnist")
    ap.add_argument("--model", default="dnn")
    ap.add_argument("--m", type=int, default=60, help="Arnoldi iterations")
    ap.add_argument("--k", type=int, default=20, help="top Ritz pairs kept")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = DATASET_NETWORK_CONFIG[(args.target, args.model)]
    kw = dict(key=args.target, gpu=-1,
              target=args.target, model_type=args.model, seed=args.seed,
              save_dir=args.save_dir, relabel_percentage=0,
              n_tr=256, n_val=256, batch_size=cfg["batch_size"], device="cpu")
    calc = IcmlAdapterCalculator("icml", **kw)
    dev = torch.device("cpu")
    x_tr, y_tr = calc.x_tr.to(dev), calc.y_tr.to(dev)
    x_val, y_val = calc.x_val.to(dev), calc.y_val.to(dev)
    loss_fn = calc.loss_fn

    state = load_final_model(calc.dn, args.seed, dev, calc.logger)
    model = calc._build_model_from_state(state).to(dev)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    # Hessian of the full-train mean loss at theta*, via one create_graph grad.
    g_full = torch.autograd.grad(loss_fn(model(x_tr), y_tr), params, create_graph=True)

    def hvp(vflat):
        vs = _unflat(vflat, params)
        dot = sum((gi * vi).sum() for gi, vi in zip(g_full, vs))
        hv = torch.autograd.grad(dot, params, retain_graph=True)
        return _flat([h.detach() for h in hv])

    n = int(sum(p.numel() for p in params))
    m = min(args.m, n)
    Q = torch.zeros(n, m + 1)
    H = torch.zeros(m + 1, m)
    g = torch.Generator().manual_seed(0)
    b = torch.randn(n, generator=g)
    Q[:, 0] = b / b.norm()
    used = m
    for j in range(m):
        w = hvp(Q[:, j])
        for i in range(j + 1):
            H[i, j] = torch.dot(Q[:, i], w)
            w = w - H[i, j] * Q[:, i]
        H[j + 1, j] = w.norm()
        if H[j + 1, j] < 1e-9:
            used = j + 1
            break
        Q[:, j + 1] = w / H[j + 1, j]

    Hm = H[:used, :used]
    eva, eve = torch.linalg.eig(Hm)
    lam = eva.real
    U = Q[:, :used] @ eve.real  # n x used Ritz vectors
    keep = torch.argsort(lam.abs(), descending=True)[:min(args.k, used)]
    lam_k = lam[keep]
    # drop near-zero eigenvalues (unstable inverse)
    valid = lam_k.abs() > 1e-6
    lam_k, Uk = lam_k[valid], U[:, keep][:, valid]

    # gradients: val (mean) and per train sample
    model.zero_grad()
    gv = _flat([g.detach() for g in torch.autograd.grad(
        loss_fn(model(x_val), y_val), params)])
    coef_val = (Uk.t() @ gv) / lam_k  # k

    scores = np.zeros(x_tr.shape[0])
    for i in range(x_tr.shape[0]):
        gi = _flat([g.detach() for g in torch.autograd.grad(
            loss_fn(model(x_tr[i:i + 1]), y_tr[i:i + 1]), params, retain_graph=False)])
        scores[i] = float(torch.dot(coef_val, Uk.t() @ gi))

    # write + correlate vs LOO
    pd.DataFrame({"sample_idx": np.arange(len(scores)), "influence": scores}).to_csv(
        os.path.join(_ROOT, "outputs", args.save_dir, f"infl_arnoldi_{args.seed:03d}.csv"),
        index=False)

    def load_m(csv):
        d = pd.read_csv(csv)
        cols = sorted([c for c in d.columns if c.startswith("influence_epoch_")],
                      key=lambda c: int(c.rsplit("_", 1)[1]))
        return np.stack([d[c].to_numpy(np.float64) for c in cols], axis=1)

    def ac(a, b):
        f = np.isfinite(a) & np.isfinite(b)
        a, b = a[f], b[f]
        return abs(float(pearsonr(a, b)[0])) if a.size > 2 and np.std(a) and np.std(b) else float("nan")

    lf = glob.glob(os.path.join(_ROOT, "outputs", args.save_dir, "infl_loo_all_epochs_*.csv"))
    res = {"m": used, "k": int(lam_k.numel()), "n_params": n}
    if lf:
        L = load_m(lf[0])
        res["global_abs_pearson"] = round(ac(scores, L.sum(1)), 3)
        res["local_abs_pearson"] = round(float(np.nanmean(
            [ac(scores, L[:, e]) for e in range(L.shape[1])])), 3)
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
