#!/usr/bin/env python
"""GEX baseline (Kim et al., 2023) for the RQ1 fidelity comparison.

GEX approximates influence with a *geometric ensemble* instead of a Hessian
inverse: it samples models around the SGD solution along the trajectory's
geometry and averages a gradient-alignment score. We use a SWAG-style diagonal
ensemble estimated from the per-epoch checkpoints: theta ~ N(theta_bar, diag(var))
over the last epochs; for each of E draws we score g_val . g_i and average.

Like Arnoldi-IF this is a global final-region estimator with no training-window
notion, so it is expected to be competitive globally but weak locally. Reported
as a geometric-ensemble approximation of GEX (implementation-verified for shape/
sanity, not a reference port).

Usage: python scripts/rebuttal/gex_if.py --save_dir rq1_regen_s1 --seed 1 \
         --ensemble 12 --last_epochs 8 [--out out.json]
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
from wie.training.dataset_config import DATASET_NETWORK_CONFIG  # noqa: E402


def _flat_state(sd, keys):
    return torch.cat([sd[k].reshape(-1).float() for k in keys])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="rq1_regen_s1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--target", default="mnist")
    ap.add_argument("--model", default="dnn")
    ap.add_argument("--ensemble", type=int, default=12)
    ap.add_argument("--last_epochs", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = DATASET_NETWORK_CONFIG[(args.target, args.model)]
    kw = dict(key=args.target, gpu=-1, target=args.target, model_type=args.model,
              seed=args.seed, save_dir=args.save_dir, relabel_percentage=0,
              n_tr=256, n_val=256, batch_size=cfg["batch_size"], device="cpu")
    calc = IcmlAdapterCalculator("icml", **kw)
    dev = torch.device("cpu")
    x_tr, y_tr, x_val, y_val = calc.x_tr, calc.y_tr, calc.x_val, calc.y_val
    loss_fn = calc.loss_fn

    # SWAG-diagonal from the last-K epoch checkpoints
    recs = os.path.join(_ROOT, "outputs", args.save_dir, "records")
    ep_files = sorted(glob.glob(os.path.join(recs, f"epoch_*_{args.seed:03d}.pt")),
                      key=lambda p: (len(p), p))
    ep_files = [f for f in ep_files if "final" not in f]
    if len(ep_files) < 3:
        print(json.dumps({"error": "not enough epoch checkpoints", "n": len(ep_files)}))
        return 1
    ep_files = ep_files[-args.last_epochs:]

    def _state(f):
        d = torch.load(f, map_location="cpu", weights_only=False)
        return d.get("model_state", d) if isinstance(d, dict) else d

    states = [_state(f) for f in ep_files]
    keys = [k for k, v in states[-1].items() if hasattr(v, "reshape")
            and v.dtype.is_floating_point]
    flats = torch.stack([_flat_state(s, keys) for s in states])  # (K, P)
    mean = flats.mean(0)
    std = flats.std(0).clamp_min(1e-8)

    model = calc._build_model_from_state(states[-1]).to(dev)
    mparams = [k for k, _ in model.named_parameters()]
    # map flat -> per-param assignment
    shapes = [(k, states[-1][k].shape, states[-1][k].numel()) for k in keys]

    def _assign(vflat):
        sd = {k: v.clone() for k, v in states[-1].items()}
        i = 0
        for k, shp, n in shapes:
            sd[k] = vflat[i:i + n].view(shp)
            i += n
        model.load_state_dict(sd, strict=False)

    def _val_grad():
        model.zero_grad()
        loss_fn(model(x_val), y_val).backward()
        return torch.cat([p.grad.reshape(-1) for p in model.parameters()
                          if p.grad is not None]).detach()

    def _sample_grad(i):
        model.zero_grad()
        loss_fn(model(x_tr[i:i + 1]), y_tr[i:i + 1]).backward()
        return torch.cat([p.grad.reshape(-1) for p in model.parameters()
                          if p.grad is not None]).detach()

    g = torch.Generator().manual_seed(0)
    scores = np.zeros(x_tr.shape[0])
    for e in range(args.ensemble):
        draw = mean + std * torch.randn(mean.shape, generator=g)
        _assign(draw)
        gv = _val_grad()
        for i in range(x_tr.shape[0]):
            scores[i] += float(torch.dot(gv, _sample_grad(i)))
    scores /= args.ensemble

    pd.DataFrame({"sample_idx": np.arange(len(scores)), "influence": scores}).to_csv(
        os.path.join(_ROOT, "outputs", args.save_dir, f"infl_gex_{args.seed:03d}.csv"),
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
    res = {"ensemble": args.ensemble, "last_epochs": len(ep_files)}
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
