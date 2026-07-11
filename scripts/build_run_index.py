#!/usr/bin/env python3
"""Build a human-readable index of all experiment runs under outputs/.

Walks every ``global_info*.json`` (excluding ``*cleansed*`` variants) and emits one
CSV row per sibling ``infl_*.json`` (whose ``type`` field is the influence method),
so a directory holding several methods produces several rows. Directories with a
``global_info`` but no ``infl_*.json`` still get one row (method blank) so nothing is
dropped from the index.

Caveats baked into the data (documented, not bugs):
  * No run artifact records the git commit it was produced at (grep git/sha/commit
    over all global_info = 0 hits). ``git_sha`` therefore stores the *index-build*
    HEAD, NOT each run's original commit. Treat it as provenance of the index, not
    of the run.
  * ``keep_ratio`` lives only in the directory name (``keep<NN>``); legacy dirs
    (loo_similar_*, sentiment_bert_*) have no keep token, so the column is nullable.

Usage:
    python scripts/build_run_index.py --outputs-dir outputs --out runs_index.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

KEEP_RE = re.compile(r"keep_?(\d+)", re.IGNORECASE)
RELABEL_RE = re.compile(r"relabel_?(\d+)", re.IGNORECASE)
SEED_RE = re.compile(r"seed_?(\d+)", re.IGNORECASE)

COLUMNS = [
    "run_dir", "naming_scheme", "dataset", "model", "method", "method_family",
    "seed", "keep_ratio", "relabel_pct", "num_epoch", "batch_size", "lr", "alpha",
    "decay", "n_tr", "n_val", "n_test", "compute_counterfactual", "has_checkpoints",
    "infl_json", "infl_csv", "log_mtime", "git_sha",
]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return ""


def classify(dirname: str) -> str:
    if dirname.startswith("loo_similar"):
        return "loo_similar"
    if dirname.startswith("sentiment_bert"):
        return "sentiment_adhoc"
    if re.search(r"keep\d+", dirname) and re.search(r"seed\d+", dirname):
        return "grid"
    return "misc"


def first_int(pat: re.Pattern, s: str):
    m = pat.search(s)
    return int(m.group(1)) if m else None


# Known method tokens as they appear in directory names (grid dirs encode the
# method in the name even when no infl_*.json is present). Longest first so
# 'icml_all_epochs' wins over 'icml'. Baselines stay verbatim; do NOT rename
# here. Both the new team token 'wie' and the legacy 'tim' are kept so
# pre-existing (pre-rename) output dirs and new dirs both classify.
_METHOD_TOKENS = [
    "wie_all_epochs", "dve_all_epochs", "icml_all_epochs", "lava_all_epochs",
    "loo_all_epochs", "td_influence", "tracin", "nohess", "segment_true",
    "wie", "tim", "dve", "icml", "lava", "loo", "sgd", "true", "lie", "td",
]

# Legacy team-method tokens normalize to the current canonical name so the
# index reports a consistent method for old tim_* dirs and new wie_* dirs.
_LEGACY_METHOD_TOKEN_MAP = {"tim": "wie", "tim_all_epochs": "wie_all_epochs"}


def method_from_dirname(dirname: str):
    tokens = set(dirname.lower().split("_"))
    # try multi-word tokens first via substring, then single-word via segment
    for mt in _METHOD_TOKENS:
        if "_" in mt:
            if mt in dirname.lower():
                return _LEGACY_METHOD_TOKEN_MAP.get(mt, mt)
        elif mt in tokens:
            return _LEGACY_METHOD_TOKEN_MAP.get(mt, mt)
    return None


def newest_log_mtime(run_dir: Path):
    logs = list(run_dir.glob("*.log"))
    if not logs:
        return ""
    ts = max(p.stat().st_mtime for p in logs)
    # deterministic ISO date without importing datetime.now
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def has_checkpoints(run_dir: Path) -> bool:
    rec = run_dir / "records"
    if rec.is_dir():
        for _ in rec.glob("*.pt"):
            return True
    for _ in run_dir.glob("*.pt"):
        return True
    return False


def load_json(p: Path) -> dict:
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def build_rows(outputs_dir: Path, sha: str):
    rows = []
    ginfos = [
        p for p in outputs_dir.rglob("global_info*.json")
        if "cleansed" not in p.name
    ]
    for gi_path in sorted(ginfos):
        run_dir = gi_path.parent
        gi = load_json(gi_path)
        scheme = classify(run_dir.name)
        base = {
            "run_dir": str(run_dir.relative_to(outputs_dir.parent)),
            "naming_scheme": scheme,
            "dataset": gi.get("target"),
            "model": gi.get("model"),
            "seed": gi.get("seed", first_int(SEED_RE, run_dir.name)),
            "keep_ratio": first_int(KEEP_RE, run_dir.name),
            "num_epoch": gi.get("num_epoch"),
            "batch_size": gi.get("batch_size"),
            "lr": gi.get("lr"),
            "alpha": gi.get("alpha"),
            "decay": gi.get("decay"),
            "n_tr": gi.get("n_tr"),
            "n_val": gi.get("n_val"),
            "n_test": gi.get("n_test"),
            "compute_counterfactual": gi.get("compute_counterfactual"),
            "has_checkpoints": has_checkpoints(run_dir),
            "log_mtime": newest_log_mtime(run_dir),
            "git_sha": sha,
        }
        infl_jsons = sorted(
            p for p in run_dir.glob("infl_*.json") if "cleansed" not in p.name
        )
        if not infl_jsons:
            dm = method_from_dirname(run_dir.name)
            rows.append({**base, "method": dm,
                         "method_family": dm.split("_")[0] if dm else None,
                         "relabel_pct": gi.get("relabel_percentage"),
                         "infl_json": None, "infl_csv": None})
            continue
        for ij in infl_jsons:
            info = load_json(ij)
            mtype = info.get("type")
            csv_name = ij.name.replace(".json", ".csv")
            csv_path = run_dir / csv_name
            rows.append({
                **base,
                "method": mtype,
                "method_family": mtype.split("_")[0] if mtype else None,
                "relabel_pct": info.get("relabel_percentage",
                                        gi.get("relabel_percentage")),
                "infl_json": ij.name,
                "infl_csv": csv_name if csv_path.exists() else None,
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", default="outputs", type=Path)
    ap.add_argument("--out", default="runs_index.csv", type=Path)
    args = ap.parse_args()

    sha = git_sha()
    rows = build_rows(args.outputs_dir, sha)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    n_methods = len({r["method"] for r in rows if r["method"]})
    n_dirs = len({r["run_dir"] for r in rows})
    print(f"Wrote {len(rows)} rows ({n_dirs} run dirs, {n_methods} methods) -> {args.out}")
    print(f"git_sha column = index-build HEAD {sha} (NOT per-run provenance)")


if __name__ == "__main__":
    main()
