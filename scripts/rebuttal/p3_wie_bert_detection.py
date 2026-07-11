"""WIE mislabel-detection precision on the SAME BERT-IMDB run, for an apples-to-
apples comparison with TracIn (52.5%). Ranks samples by summed WIE influence
(both sign directions reported); precision@k vs the 40% relabel indices."""
import glob, os, json, numpy as np, pandas as pd
import os; base = os.path.join(os.environ.get("WIE_OUTPUTS", "outputs"), "bert_w5_detect")
d = pd.read_csv(glob.glob(base + "/infl_wie_all_epochs_relabel_040_pct_*.csv")[0])
cols = sorted([c for c in d.columns if c.startswith("influence_epoch_")],
              key=lambda c: int(c.rsplit("_", 1)[1]))
M = np.stack([d[c].to_numpy(np.float64) for c in cols], axis=1)  # (n, E)
n = M.shape[0]
tot = M.sum(1)               # total influence over the trajectory
idx = np.loadtxt(glob.glob(base + "/relabel_040_pct_indices_*.csv")[0],
                 delimiter=",", skiprows=1).astype(int).ravel()
noisy = np.zeros(n, bool); noisy[idx] = True
base_rate = float(noisy.mean())

def prec(scores, desc, cr):
    k = int(cr * n); o = np.argsort(-scores if desc else scores)[:k]
    return round(float(noisy[o].mean()), 3)

res = {"n": n, "noisy": int(noisy.sum()), "base_rate": round(base_rate, 3),
       "shape": list(M.shape),
       "wie_sum_neg": {f"prec@{int(c*100)}": prec(tot, False, c) for c in (0.1, 0.2, 0.4)},
       "wie_sum_pos": {f"prec@{int(c*100)}": prec(tot, True, c) for c in (0.1, 0.2, 0.4)},
       "wie_absmag": {f"prec@{int(c*100)}": prec(np.abs(tot), True, c) for c in (0.1, 0.2, 0.4)}}
print("WIE_DETECT", json.dumps(res, indent=2))
json.dump(res, open("outputs/bert_w5_detect/p3_wie_detection_result.json", "w"), indent=2)
