#!/usr/bin/env python
"""GEX baseline (Kim et al., 2023) — FAITHFUL static geometric ensemble (R2/E3).

R1 bug: the ensemble was drawn from the last-8 *epoch* checkpoints, i.e. it used
cross-epoch trajectory history, making it inadvertently trajectory-aware
(over-performed, local 0.76). This version samples the ensemble **only in the
neighborhood of the FINAL checkpoint** — an isotropic Gaussian around theta* with
a small documented sigma — so it carries no within-training temporal information,
as a static final-model estimator must. Ensemble construction + sigma + size are
recorded in the JSON for auditability.

infl(z_i) = mean over ensemble draws of  <g_val, g_i>  (gradient-alignment,
Hessian-inversion-free, GEX's defining idea), evaluated at models sampled around
theta*.

Usage: python scripts/rebuttal/gex_if.py --save_dir rq1_regen_s1 --seed 1 \
         --ensemble 12 --sigma_rel 0.02 [--out out.json]
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from wie.infl.adapters import IcmlAdapterCalculator  # noqa: E402
from wie.infl.core import load_final_model  # noqa: E402
from wie.training.dataset_config import DATASET_NETWORK_CONFIG  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="rq1_regen_s1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--target", default="mnist")
    ap.add_argument("--model", default="dnn")
    ap.add_argument("--ensemble", type=int, default=12)
    ap.add_argument("--sigma_rel", type=float, default=0.02,
                    help="isotropic Gaussian std = sigma_rel * RMS(|theta*|), around FINAL ckpt only")
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    dev = torch.device("cpu" if args.gpu < 0 else f"cuda:{args.gpu}")
    cfg = DATASET_NETWORK_CONFIG[(args.target, args.model)]
    kw = dict(key=args.target, gpu=args.gpu, target=args.target, model_type=args.model,
              seed=args.seed, save_dir=args.save_dir, relabel_percentage=0,
              n_tr=256, n_val=256, batch_size=cfg["batch_size"],
              device=("cpu" if args.gpu < 0 else "cuda"))
    calc = IcmlAdapterCalculator("icml", **kw)
    x_tr, y_tr, x_val, y_val = calc.x_tr.to(dev), calc.y_tr.to(dev), calc.x_val.to(dev), calc.y_val.to(dev)
    loss_fn = calc.loss_fn

    # FINAL checkpoint only — no cross-epoch history
    state = load_final_model(calc.dn, args.seed, dev, calc.logger)
    model = calc._build_model_from_state(state).to(dev)
    keys = [k for k, v in state.items() if hasattr(v, "reshape") and v.dtype.is_floating_point]
    mean = torch.cat([state[k].reshape(-1).float().to(dev) for k in keys])
    sigma = float(args.sigma_rel) * float(mean.pow(2).mean().sqrt())      # isotropic, relative to theta* scale
    shapes = [(k, state[k].shape, state[k].numel()) for k in keys]

    def assign(vflat):
        sd = {k: v.clone() for k, v in state.items()}
        i = 0
        for k, shp, num in shapes:
            sd[k] = vflat[i:i + num].view(shp).to(sd[k].dtype); i += num
        model.load_state_dict(sd, strict=False)

    def grad_flat(xb, yb):
        model.zero_grad(set_to_none=True)
        loss_fn(model(xb), yb).backward()
        return torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]).detach()

    g = torch.Generator(device="cpu").manual_seed(0)
    scores = np.zeros(x_tr.shape[0])
    for _ in range(args.ensemble):
        draw = mean + sigma * torch.randn(mean.shape, generator=g).to(dev)   # around theta* ONLY
        assign(draw)
        gv = grad_flat(x_val, y_val)
        for i in range(x_tr.shape[0]):
            scores[i] += float(torch.dot(gv, grad_flat(x_tr[i:i + 1], y_tr[i:i + 1])))
    scores /= args.ensemble

    pd.DataFrame({"sample_idx": np.arange(len(scores)), "influence": scores}).to_csv(
        os.path.join(_ROOT, "outputs", args.save_dir, f"infl_gex_{args.seed:03d}.csv"), index=False)

    # quick |Pearson| vs LOO for a sanity print (full 4-metric table is e3_fidelity_4metrics.py)
    def load_m(csv):
        d = pd.read_csv(csv); cols = sorted([c for c in d.columns if c.startswith("influence_epoch_")],
                                            key=lambda c: int(c.rsplit("_", 1)[1]))
        return np.stack([d[c].to_numpy(np.float64) for c in cols], axis=1)

    def ac(a, b):
        f = np.isfinite(a) & np.isfinite(b); a, b = a[f], b[f]
        return abs(float(pearsonr(a, b)[0])) if a.size > 2 and np.std(a) and np.std(b) else float("nan")

    res = {"ensemble_construction": "isotropic Gaussian around FINAL checkpoint only",
           "ensemble_size": args.ensemble, "sigma_rel": args.sigma_rel, "sigma_abs": round(sigma, 5),
           "n_params": int(mean.numel())}
    lf = glob.glob(os.path.join(_ROOT, "outputs", args.save_dir, "infl_loo_all_epochs_*.csv"))
    if lf:
        L = load_m(lf[0])
        res["global_abs_pearson"] = round(ac(scores, L.sum(1)), 3)
        res["local_abs_pearson"] = round(float(np.nanmean([ac(scores, L[:, e]) for e in range(L.shape[1])])), 3)
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
