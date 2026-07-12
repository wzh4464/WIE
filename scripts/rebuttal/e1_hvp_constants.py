#!/usr/bin/env python
"""E1a: measured HVP cost constants on BERT-base (EXPERIMENTS-R2).

Times, as medians over >=20 reps on one GPU (fp32, gradient checkpointing,
batch 8, seq 128):
  - one backward   : forward + loss.backward()
  - one HVP        : forward + grad(create_graph) + grad(<g,v>)  [Pearlmutter]
  - one train step : forward + backward + AdamW.step()
Reports:  x  = HVP / backward ;  x' = HVP / train step ;  y = 2-HVP WIE step / train step.
Writes artifacts/e1_hvp_constants.json (merged into the E1 table artifact later).
"""
import json, os, sys, time, statistics
import torch
from transformers import AutoModelForSequenceClassification

dev = torch.device("cuda")
torch.manual_seed(0)
m = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(dev)
try:
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:
    import torch.utils.checkpoint as _c; _o = _c.checkpoint
    _c.checkpoint = lambda *a, **k: _o(*a, **{**k, "use_reentrant": False}); m.gradient_checkpointing_enable()
m.config.use_cache = False
m.train()
params = [p for p in m.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=2e-5)
B, L = 8, 128
ids = torch.randint(0, 30000, (B, L), device=dev)
mask = torch.ones(B, L, device=dev, dtype=torch.long)
y = torch.randint(0, 2, (B,), device=dev)
lossf = torch.nn.CrossEntropyLoss()
P = sum(p.numel() for p in params)

def fwd_loss():
    return lossf(m(input_ids=ids, attention_mask=mask).logits, y)

def one_backward():
    m.zero_grad(set_to_none=True); fwd_loss().backward()

def one_hvp():
    m.zero_grad(set_to_none=True)
    g = torch.autograd.grad(fwd_loss(), params, create_graph=True)
    v = [torch.randn_like(p) for p in params]
    dot = sum((gi * vi).sum() for gi, vi in zip(g, v))
    torch.autograd.grad(dot, params, retain_graph=False)

def one_step():
    opt.zero_grad(set_to_none=True); fwd_loss().backward(); opt.step()

def bench(fn, n=25, warmup=5):
    for _ in range(warmup):
        fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)

t_bwd = bench(one_backward)
t_hvp = bench(one_hvp)
t_step = bench(one_step)
peak = torch.cuda.max_memory_allocated() / 1e9

x = t_hvp[0] / t_bwd[0]
xp = t_hvp[0] / t_step[0]
yv = 2 * t_hvp[0] / t_step[0]
out = {
    "experiment": "E1a HVP cost constants", "model": "bert-base-uncased",
    "params_M": round(P / 1e6, 1), "batch": B, "seq_len": L, "precision": "fp32",
    "gpu": torch.cuda.get_device_name(0), "reps": 25,
    "median_seconds": {"backward": round(t_bwd[0], 4), "hvp": round(t_hvp[0], 4),
                       "train_step": round(t_step[0], 4)},
    "HVP_over_backward_x": round(x, 2),
    "HVP_over_trainstep_xprime": round(xp, 2),
    "twoHVP_WIEstep_over_trainstep_y": round(yv, 2),
    "peak_gb": round(peak, 2),
}
os.makedirs("rebuttal/rebuttal-drafts/artifacts", exist_ok=True)
json.dump(out, open("rebuttal/rebuttal-drafts/artifacts/e1_hvp_constants.json", "w"), indent=2)
print(json.dumps(out, indent=2))
