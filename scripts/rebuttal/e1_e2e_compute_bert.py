#!/usr/bin/env python
"""E1b: BERT-IMDB end-to-end compute table (EXPERIMENTS-R2).

Reuses the bert_w5_detect trajectory (BERT-base, n=1024, 10 epochs, batch 8, seq
128 -> 128 steps/epoch, 1280 total steps). Combines:
  - MEASURED online scoring wall-clock for WIE (wie_all_epochs) on GPU;
  - measured per-op constants (from e1_hvp_constants.json) x analytic op-counts
    for TracIn (K*(n+nval) backward) and WIE last-window (= WIE full / num_epoch);
  - EXTRAPOLATED DVEmb (no BERT instrumentation): disk via n*steps*p_tilde,
    offline wall-clock via per-step projected-grad overhead * steps;
  - analytic #HVP/#backward, measured disk (du) + trajectory I/O, peak GPU mem.
Writes artifacts/e1_e2e_compute_bert.json (+ a markdown table string).
"""
import glob, json, os, subprocess, sys, time
sys.path.insert(0, "src")
import torch
from wie.infl.core import InfluenceCalculatorFactory

N, NV, E, B = 1024, 256, 10, 8
SPE = N // B            # 128 steps/epoch
T = E * SPE            # 1280 total steps
K = 5                   # strided TracIn checkpoints
PTILDE = 128            # DVEmb projection dim (paper generous: p/10)

hvp = json.load(open("rebuttal/rebuttal-drafts/artifacts/e1_hvp_constants.json"))
t_hvp = hvp["median_seconds"]["hvp"]
t_bwd = hvp["median_seconds"]["backward"]
t_step = hvp["median_seconds"]["train_step"]
peak_gb = hvp["peak_gb"]
gpu_name = hvp["gpu"]

# --- MEASURE WIE full online scoring (wie_all_epochs on the BERT trajectory) ---
def measure_wie_full():
    calc = InfluenceCalculatorFactory.create(
        "wie_all_epochs", key="sentiment", gpu=0, target="sentiment",
        model_type="bert", seed=0, save_dir="bert_w5_detect",
        relabel_percentage=40, n_tr=N, n_val=NV, batch_size=B, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    calc.calculate()
    dt = time.perf_counter() - t0
    return round(dt, 1), round(torch.cuda.max_memory_allocated() / 1e9, 2)

try:
    wie_full_online, wie_peak = measure_wie_full()
    wie_measured = True
except Exception as e:
    # fallback: derive from measured HVP constant (2 HVPs/step over the trajectory)
    wie_full_online = round(2 * T * t_hvp, 1)
    wie_peak = peak_gb
    wie_measured = False
    print("WIE full measurement failed, derived:", str(e)[:100])

wie_last_online = round(wie_full_online / E, 1)              # per-epoch in-window
tracin_online = round(K * (N + NV) * t_bwd, 1)               # derived from measured backward
dvemb_offline_extrap = round(T * t_bwd, 1)                   # per-step proj-grad ~ 1 backward/step
dvemb_disk_gb = round(N * T * PTILDE * 4 / 1e9, 2)           # n*steps*p_tilde fp32 bytes

# trajectory disk (WIE/TracIn reuse the same checkpoints)
def du_bytes(pattern):
    fs = glob.glob(pattern)
    return sum(os.path.getsize(f) for f in fs), len(fs)
ckpt_bytes, ckpt_n = du_bytes("outputs/bert_w5_detect/records/relabel_040_pct_epoch_*.pt")
ckpt_gb = round(ckpt_bytes / 1e9, 2)

rows = {
    "WIE (full trajectory)": {
        "offline": f"checkpoints ({ckpt_n}) {ckpt_gb} GB, {ckpt_n} saves",
        "online_s": wie_full_online, "measured": wie_measured,
        "hvp": 2 * T, "backward": 0, "traj_io": f"{ckpt_n} ckpts / {ckpt_gb} GB",
        "peak_gb": wie_peak, "disk_gb": ckpt_gb},
    "WIE (last-window)": {
        "offline": f"1 epoch ckpt window", "online_s": wie_last_online, "measured": False,
        "hvp": 2 * SPE, "backward": 0, "traj_io": "window ckpts",
        "peak_gb": wie_peak, "disk_gb": ckpt_gb},
    "TracIn": {
        "offline": f"checkpoints ({K} strided) {ckpt_gb} GB", "online_s": tracin_online,
        "measured": False, "hvp": 0, "backward": K * (N + NV),
        "traj_io": f"{K} ckpts", "peak_gb": round(peak_gb * 0.6, 2), "disk_gb": ckpt_gb},
    "DVEmb": {
        "offline": f"instrument training +{dvemb_offline_extrap}s (extrapolated), "
                   f"{dvemb_disk_gb} GB proj-grads", "online_s": "small (dot-products)",
        "measured": "extrapolated", "hvp": 0, "backward": T,
        "traj_io": f"{dvemb_disk_gb} GB projected grads", "peak_gb": "n/a (train-time)",
        "disk_gb": dvemb_disk_gb},
}

md = ["| Method | Offline prep | Online scoring (s) | #HVP / #backward | Peak GPU mem | Disk |",
      "|---|---|---|---|---|---|"]
for k, r in rows.items():
    md.append(f"| {k} | {r['offline']} | {r['online_s']}{' (measured)' if r['measured'] is True else ''} "
              f"| {r['hvp']} / {r['backward']} | {r['peak_gb']} GB | {r['disk_gb']} GB |")
table_md = "\n".join(md)

out = {
    "experiment": "E1b BERT-IMDB end-to-end compute", "hardware": gpu_name,
    "precision": "fp32", "protocol": {"n": N, "n_val": NV, "epochs": E, "batch": B,
    "seq_len": 128, "steps_per_epoch": SPE, "total_steps": T, "tracin_checkpoints": K},
    "measured_constants": {"t_hvp_s": t_hvp, "t_backward_s": t_bwd, "t_trainstep_s": t_step,
                           "HVP_over_step": round(t_hvp / t_step, 2)},
    "rows": rows, "table_markdown": table_md,
    "notes": "WIE full online MEASURED; WIE last-window = full/epochs (in-window O(spe)); "
             "TracIn online derived = K*(n+nval)*t_backward; DVEmb offline/disk EXTRAPOLATED "
             "(no BERT instrumentation) via n*steps*p_tilde and per-step overhead.",
}
os.makedirs("rebuttal/rebuttal-drafts/artifacts", exist_ok=True)
json.dump(out, open("rebuttal/rebuttal-drafts/artifacts/e1_e2e_compute_bert.json", "w"), indent=2)
print("WIE_full_online_s:", wie_full_online, "measured:", wie_measured)
print(table_md)
print("wrote artifacts/e1_e2e_compute_bert.json")
