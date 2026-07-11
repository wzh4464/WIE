"""BERT-IMDB TracIn mislabel detection (clean, full-GPU v2).
Fixes the earlier NaN: accumulate ||g||^2 and <g,g_val> in float64 with NaN
guards. Computes self-influence and val-influence; reports precision@k vs the
40% relabel indices. Uses all 10 epoch checkpoints (strided to ~6)."""
import sys, os, glob, gc, json
sys.path.insert(0, "src")
import numpy as np, torch, pandas as pd
from wie.infl.core import InfluenceCalculatorFactory

dev = torch.device("cuda")
calc = InfluenceCalculatorFactory.create(
    "tracin", key="sentiment", gpu=0, target="sentiment", model_type="bert",
    seed=0, save_dir="bert_w5_detect", relabel_percentage=40,
    n_tr=1024, n_val=256, batch_size=8, device="cuda")  # dev=cuda:0 (GPU0, free)
x_tr, y_tr, x_val, y_val = calc.x_tr, calc.y_tr, calc.x_val, calc.y_val
loss_fn = calc.loss_fn
n_tr = x_tr.shape[0]
gi = json.load(open(glob.glob(os.path.join(calc.dn, "global_info_*.json"))[0]))
lr = float(gi.get("lr", 2e-5))
ckpts = sorted(glob.glob(os.path.join(calc.dn, "records", "relabel_040_pct_epoch_*.pt")),
               key=lambda p: int(p.rsplit("_", 2)[1]))
print("all ckpts:", len(ckpts), flush=True)

def state_of(f):
    d = torch.load(f, map_location="cpu", weights_only=False)
    return d.get("model_state", d) if isinstance(d, dict) else d

self_s = np.zeros(n_tr, dtype=np.float64)
val_s = np.zeros(n_tr, dtype=np.float64)
nan_ct = 0

for ci, cf in enumerate(ckpts):
    model = calc._build_model_from_state(state_of(cf)).to(dev); model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    # val gradient (full val set, big batch OK now), per-param, float64
    gval = [torch.zeros_like(p, dtype=torch.float64) for p in params]
    VB = 32; nvb = 0
    for a in range(0, x_val.shape[0], VB):
        model.zero_grad(set_to_none=True)
        loss_fn(model(x_val[a:a+VB].to(dev)), y_val[a:a+VB].to(dev)).backward()
        for gv, p in zip(gval, params):
            if p.grad is not None: gv += p.grad.detach().double()
        nvb += 1
    for gv in gval: gv /= max(nvb, 1)

    for i in range(n_tr):
        model.zero_grad(set_to_none=True)
        loss_fn(model(x_tr[i:i+1].to(dev)), y_tr[i:i+1].to(dev)).backward()
        s = 0.0; v = 0.0; bad = False
        for p, gv in zip(params, gval):
            if p.grad is None: continue
            g = p.grad.detach().double()
            if not torch.isfinite(g).all(): bad = True; break
            s += float((g*g).sum()); v += float((g*gv).sum())
        if bad: nan_ct += 1; continue
        self_s[i] += lr * s; val_s[i] += lr * v
        if i % 512 == 0: print(f"  ckpt {ci+1}/{len(ckpts)} sample {i}", flush=True)
    del model, params, gval; gc.collect(); torch.cuda.empty_cache()

idx = np.loadtxt(glob.glob(os.path.join(calc.dn, "relabel_040_pct_indices_*.csv"))[0],
                 delimiter=",", skiprows=1).astype(int).ravel()
noisy = np.zeros(n_tr, bool); noisy[idx] = True
base = float(noisy.mean())
print(f"nan-skipped grads: {nan_ct}", flush=True)
print(f"self-inf: clean mean={self_s[~noisy].mean():.4g} noisy mean={self_s[noisy].mean():.4g}", flush=True)
print(f"val-inf : clean mean={val_s[~noisy].mean():.4g} noisy mean={val_s[noisy].mean():.4g}", flush=True)

def prec(scores, desc, cr):
    k = int(cr*n_tr); o = np.argsort(-scores if desc else scores)[:k]
    return round(float(noisy[o].mean()), 3)

res = {"n_tr": n_tr, "noisy": int(noisy.sum()), "base_rate": round(base, 3), "nan_skipped": nan_ct,
       "self_influence_high": {f"prec@{int(c*100)}": prec(self_s, True, c) for c in (0.1,0.2,0.4)},
       "val_influence_neg":  {f"prec@{int(c*100)}": prec(val_s, False, c) for c in (0.1,0.2,0.4)},
       "val_influence_pos":  {f"prec@{int(c*100)}": prec(val_s, True, c) for c in (0.1,0.2,0.4)}}
print("RESULT", json.dumps(res), flush=True)
json.dump(res, open("outputs/bert_w5_detect/p3_tracin_detection_result.json","w"), indent=2)
pd.DataFrame({"sample_idx":np.arange(n_tr),"self":self_s,"val":val_s,"noisy":noisy}).to_csv(
    os.path.join(calc.dn,"tracin_detection_v2.csv"), index=False)
