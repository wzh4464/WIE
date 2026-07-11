#!/usr/bin/env python
"""End-to-end validation of the faithful WIE implementations vs the LOO oracle.

For one (target, model, seed) config this:
  1. trains with per-epoch counterfactual (LOO) models,
  2. builds segment_true_full (per-epoch full-LOO val-loss trajectory),
  3. computes the window LOO oracle true_{first,middle,last} (length L),
  4. computes the faithful wie_{first,middle,last} (length L) + wie_all_epochs,
  5. correlates each wie_X vs true_X (Spearman + Pearson), and
  6. classifies the wie_all_epochs matrix into the four RQ2 temporal patterns.

Emits a JSON summary to stdout (and --out). Designed to be fanned out over seeds.

Example:
  python scripts/validate_faithful_wie.py --target mnist --model dnn --seed 0 \
      --n_tr 200 --num_epoch 10 --length 3 --gpu -1
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from wie.analysis.temporal_patterns import (  # noqa: E402
    classify_patterns,
    pattern_distribution,
)


def _run(cmd, env, log):
    log.write(f"\n$ {' '.join(cmd)}\n")
    log.flush()
    r = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return r.returncode


def _infl_csv(base, infl_type, seed):
    p = os.path.join(base, f"infl_{infl_type}_{seed:03d}.csv")
    return p if os.path.isfile(p) else None


def _load_scores(base, infl_type, seed):
    p = _infl_csv(base, infl_type, seed)
    if p is None:
        return None
    d = pd.read_csv(p)
    cols = [c for c in d.columns if "influence" in c.lower()]
    return d[cols[0]].to_numpy(dtype=np.float64) if cols else None


def _load_matrix(base, infl_type, seed):
    p = _infl_csv(base, infl_type, seed)
    if p is None:
        return None
    d = pd.read_csv(p)
    cols = sorted(
        [c for c in d.columns if c.startswith("influence_epoch_")],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    return np.stack([d[c].to_numpy(np.float64) for c in cols], axis=1) if cols else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mnist")
    ap.add_argument("--model", default="dnn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_tr", type=int, default=200)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--num_epoch", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--length", type=int, default=3)
    ap.add_argument("--gpu", type=int, default=-1, help="-1 = CPU")
    ap.add_argument("--save_dir", default=None, help="basename (maps under outputs/)")
    ap.add_argument("--out", default=None, help="write JSON summary here too")
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args(argv)

    sd = args.save_dir or f"val_{args.target}_{args.model}_s{args.seed}"
    # Force a plain basename: the child train/influence commands map any save_dir
    # to REPO_ROOT/outputs/<basename>, so pinning the basename here keeps our
    # `base` in lockstep with where they actually write -- AND ensures the
    # rmtree below can only ever touch a directory INSIDE the repo outputs tree
    # (never an arbitrary absolute path the user might pass).
    sd = os.path.basename(os.path.normpath(sd))
    if not sd or sd in (".", ".."):
        print(f"ERROR: invalid --save_dir basename {sd!r}", file=sys.stderr)
        return 2
    base = os.path.join(_ROOT, "outputs", sd)
    # Fresh start when training: remove any prior outputs so a swallowed training
    # failure (the train CLI can catch exceptions and still exit 0) cannot leave
    # the influence jobs running against STALE artifacts from an earlier run.
    if not args.skip_train and os.path.isdir(base):
        shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base, exist_ok=True)
    env = dict(os.environ)
    # Child `python -m wie...` processes must find the package too (a fresh
    # checkout run without an editable install relies on src being on the path).
    src = os.path.join(_ROOT, "src")
    env["PYTHONPATH"] = (
        src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    )
    if args.gpu < 0:
        env["CUDA_VISIBLE_DEVICES"] = ""
    logpath = os.path.join(base, "validate_run.log")
    log = open(logpath, "w")

    common = [
        "--target",
        args.target,
        "--model",
        args.model,
        "--seed",
        str(args.seed),
        "--gpu",
        str(args.gpu),
        "--save_dir",
        sd,
        "--log_level",
        "ERROR",
    ]

    result = {"config": vars(args), "save_dir": base, "steps": {}}

    # 1) Train (with counterfactual LOO on by default)
    if not args.skip_train:
        rc = _run(
            [
                sys.executable,
                "-m",
                "wie.training.train",
                *common,
                "--n_tr",
                str(args.n_tr),
                "--n_val",
                str(args.n_val),
                "--num_epoch",
                str(args.num_epoch),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
            ],
            env,
            log,
        )
        result["steps"]["train"] = rc
        # The train CLI can swallow exceptions and still exit 0, so verify the
        # FRESH counterfactual artifacts actually exist rather than trusting rc.
        rec = os.path.join(base, "records")
        gi = os.path.join(base, f"global_info_{args.seed:03d}.json")
        cf0 = os.path.join(rec, f"counterfactual_0000_epoch_0_{args.seed:03d}.pt")
        ep0 = os.path.join(rec, f"epoch_0_{args.seed:03d}.pt")
        missing = [p for p in (gi, cf0, ep0) if not os.path.isfile(p)]
        if rc != 0 or missing:
            result["error"] = (
                f"train failed or produced no fresh counterfactual artifacts "
                f"(rc={rc}, missing={[os.path.relpath(p, base) for p in missing]})"
            )
            return _finish(result, args, log)

    # 2-4) influence computations
    infl_jobs = [
        ("segment_true_full", []),
        ("true_first", ["--length", str(args.length)]),
        ("true_middle", ["--length", str(args.length)]),
        ("true_last", ["--length", str(args.length)]),
        ("wie_first", ["--length", str(args.length)]),
        ("wie_middle", ["--length", str(args.length)]),
        ("wie_last", ["--length", str(args.length)]),
        ("wie_all_epochs", []),
    ]
    for t, extra in infl_jobs:
        rc = _run(
            [sys.executable, "-m", "wie.infl", *common, "--type", t, *extra],
            env,
            log,
        )
        result["steps"][t] = rc

    # 5) correlations wie_X vs true_X
    corr = {}
    for win in ["first", "middle", "last"]:
        w = _load_scores(base, f"wie_{win}", args.seed)
        tr = _load_scores(base, f"true_{win}", args.seed)
        if w is None or tr is None or len(w) != len(tr) or len(w) < 3:
            corr[win] = {"spearman": None, "pearson": None, "n": None}
            continue
        finite = np.isfinite(w) & np.isfinite(tr)
        w, tr = w[finite], tr[finite]
        sp = spearmanr(w, tr).correlation if np.std(w) > 0 and np.std(tr) > 0 else None
        pr = pearsonr(w, tr)[0] if np.std(w) > 0 and np.std(tr) > 0 else None
        corr[win] = {
            "spearman": None if sp is None else round(float(sp), 4),
            "pearson": None if pr is None else round(float(pr), 4),
            "n": int(len(w)),
        }
    result["correlation_wie_vs_true"] = corr

    # 6) RQ2 temporal-pattern distribution from wie_all_epochs
    M = _load_matrix(base, "wie_all_epochs", args.seed)
    if M is not None and M.ndim == 2 and M.shape[1] >= 3:
        labels, _ = classify_patterns(M)
        result["pattern_distribution_pct"] = {
            k: round(v, 2) for k, v in pattern_distribution(labels).items()
        }
        result["pattern_matrix_shape"] = list(M.shape)
    else:
        result["pattern_distribution_pct"] = None

    return _finish(result, args, log)


def _finish(result, args, log):
    log.close()
    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    # Non-zero exit on any failure so CI / batch runners don't treat a failed
    # validation (or one that consumed stale CSVs) as success.
    if result.get("error"):
        return 1
    if any(code != 0 for code in result.get("steps", {}).values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
