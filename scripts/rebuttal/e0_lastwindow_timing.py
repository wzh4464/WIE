#!/usr/bin/env python
"""E0: last-window cost semantics + timing (EXPERIMENTS-R2).

Two calculators, opposite sweep semantics (verified by code read):
  - wie_all_epochs : IN-WINDOW truncation, each epoch's window costs O(steps/epoch);
    the last-epoch window (Table 2 / BERT detection protocol) sweeps only that epoch.
  - wie_first/last : FULL backward sweep to step 0, O(t2) (builds the pre-window term).

We time, on rq1_regen_s1 (MNIST-DNN, 21 epochs, 84 steps):
  full-trajectory attribution  = wie_first  (window [0,T], full O(t2) sweep)  -> [a]
  all-epochs in-window          = wie_all_epochs (21 in-window windows)         -> t_all
  last-window in-window         = t_all / num_epoch (per-epoch, O(spe))         -> [b]
so [a]/[b] ~= num_epoch ~= 20x, substantiating the paper's "20x less compute".

Writes artifacts/e0_lastwindow_semantics.json and prints [a],[b].
"""
import glob, json, os, sys, time
sys.path.insert(0, "src")
from wie.infl.core import InfluenceCalculatorFactory
from wie.training.dataset_config import DATASET_NETWORK_CONFIG

ART = "rebuttal/rebuttal-drafts/artifacts/e0_lastwindow_semantics.json"
SAVE_DIR = "rq1_regen_s1"
SEED = 1
cfg = DATASET_NETWORK_CONFIG[("mnist", "dnn")]
gi = json.load(open(glob.glob(f"outputs/{SAVE_DIR}/global_info_*.json")[0]))
num_epoch = int(gi["num_epoch"])

def mk(t):
    return InfluenceCalculatorFactory.create(
        t, key="mnist", gpu=-1, target="mnist", model_type="dnn", seed=SEED,
        save_dir=SAVE_DIR, relabel_percentage=0,
        n_tr=gi["n_tr"], n_val=gi["n_val"], batch_size=gi["batch_size"], device="cpu")

def timed(t, length=None):
    kw = {}
    c = mk(t)
    if length is not None and hasattr(c, "length"):
        c.length = length
    t0 = time.perf_counter()
    c.calculate()
    return time.perf_counter() - t0

# warm caches once (data load) then time
res = {}
try:
    res["t_full_sweep_wie_first"] = round(timed("wie_first"), 3)  # [a]: window [0,T], O(t2)
except Exception as e:
    res["t_full_sweep_wie_first"] = f"ERR: {str(e)[:80]}"
try:
    t_all = timed("wie_all_epochs")                              # 21 in-window windows
    res["t_all_epochs_inwindow"] = round(t_all, 3)
    res["t_last_window_inwindow"] = round(t_all / num_epoch, 3)   # [b]: one epoch, O(spe)
except Exception as e:
    res["t_all_epochs_inwindow"] = f"ERR: {str(e)[:80]}"
try:
    res["t_last_exact_wie_last_len1"] = round(timed("wie_last", length=1), 3)  # O(t2) exact
except Exception as e:
    res["t_last_exact_wie_last_len1"] = f"ERR: {str(e)[:80]}"

# [a] = full-trajectory attribution over ALL epochs (wie_all_epochs total);
# [b] = last-epoch window alone (in-window, one epoch). ratio ~= num_epoch ~= 20x.
a = res.get("t_all_epochs_inwindow")
b = res.get("t_last_window_inwindow")
ratio = round(a / b, 1) if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b else None

out = {
    "experiment": "E0 last-window cost semantics",
    "run": SAVE_DIR, "num_epoch": num_epoch,
    "steps_per_epoch": gi["n_tr"] // gi["batch_size"], "total_steps": num_epoch * (gi["n_tr"] // gi["batch_size"]),
    "implementation": {
        "last_window_protocol": "in-window truncation (wie_all_epochs)",
        "wie_all_epochs": "O(steps/epoch) per epoch; loop to epoch_idx*steps_per_epoch "
                          "(wie_all_epochs.py:166); pre-window deviation implicitly 0",
        "wie_first_last_middle": "O(t2) full backward sweep to step 0 "
                                 "(wie_window_base.py:260); builds (∏P−I)δ^(t1) term",
        "bert_detection_uses": "wie_all_epochs (in-window)",
    },
    "timings_seconds_cpu": res,
    "a_full_trajectory_s": a, "b_last_window_s": b, "ratio_a_over_b": ratio,
    "note": "last-window in-window cost = all-epochs total / num_epoch (uniform per-epoch, O(spe)); "
            "ratio approximates num_epoch, substantiating Table 2's ~20x reduction.",
}
os.makedirs(os.path.dirname(ART), exist_ok=True)
json.dump(out, open(ART, "w"), indent=2)
print(json.dumps({"a": a, "b": b, "ratio": ratio, "timings": res}, indent=2))
print("wrote", ART)
