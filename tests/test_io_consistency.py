import os
import sys
import json
import hashlib
import pickle
import subprocess
from pathlib import Path

from wie.utils.paths import resolve_data_dir, resolve_output_dir


def _write_sentiment_cache(
    n_tr=10,
    n_val=10,
    n_test=256,
    seed=42,
    model_name="bert-base-uncased",
    max_length=128,
):
    import numpy as np

    np.random.seed(seed)
    data_dir = Path(resolve_data_dir("experiment"))
    data_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "-")
    tag = f"{safe_name}_ml{max_length}"
    cache = data_dir / f"SentimentModule_{tag}_{n_tr}_{n_val}_{n_test}_{seed}.pkl"

    def make_x(N):
        L = max_length
        input_ids = np.random.randint(0, 100, size=(N, L), dtype=np.int64)
        attn_mask = np.ones((N, L), dtype=np.int64)
        return np.stack([input_ids, attn_mask], axis=1)

    def make_y(N):
        return np.random.randint(0, 2, size=(N,), dtype=np.int64)

    x_tr, y_tr = make_x(n_tr), make_y(n_tr)
    x_val, y_val = make_x(n_val), make_y(n_val)
    x_test, y_test = make_x(n_test), make_y(n_test)
    with open(cache, "wb") as f:
        pickle.dump(((x_tr, y_tr), (x_val, y_val), (x_test, y_test)), f)
    return str(cache)


def _dir_md5(root):
    root = Path(root)
    mapping = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        # skip log files (contain timestamps)
        if p.name.startswith("log_") or "/logs/" in str(p):
            continue
        h = hashlib.md5()
        with open(p, "rb") as f:
            while True:
                b = f.read(8192)
                if not b:
                    break
                h.update(b)
        mapping[str(p.relative_to(root))] = h.hexdigest()
    return mapping


def test_epoch_wise_keep_ratio_md5():
    # Prepare tiny offline cache for SentimentModule
    _ = _write_sentiment_cache()
    env = os.environ.copy()
    env["HF_TEXT_MODEL"] = "bert-base-uncased"

    # Run BEFORE
    before_out = Path(resolve_output_dir("before"))
    if before_out.exists():
        # cleanup
        import shutil

        shutil.rmtree(before_out)
    cmd = [
        sys.executable,
        "-m",
        "scripts.epoch_wise_keep_ratio",
        "--target",
        "sentiment",
        "--model",
        "bert",
        "--save_dir",
        "before",
        "--relabel",
        "30",
        "--seed",
        "42",
        "--gpu",
        "2",
        "--type",
        "wie_all_epochs",
        "--log_level",
        "INFO",
        "--n_tr",
        "10",
        "--n_val",
        "10",
    ]
    subprocess.run(cmd, check=False, env=env)
    before_md5 = _dir_md5(before_out)

    # Run AFTER
    after_out = Path(resolve_output_dir("after"))
    if after_out.exists():
        import shutil

        shutil.rmtree(after_out)
    cmd[cmd.index("before") + 0] = "after"
    subprocess.run(cmd, check=False, env=env)
    after_md5 = _dir_md5(after_out)

    # Store for debugging
    Path("outputs").mkdir(parents=True, exist_ok=True)
    output_base = Path(resolve_output_dir())
    with open(output_base / "before_md5.json", "w") as f:
        json.dump(before_md5, f, indent=2)
    with open(output_base / "after_md5.json", "w") as f:
        json.dump(after_md5, f, indent=2)

    # Compare (ignoring logs)
    assert before_md5 == after_md5, (
        f"MD5 mismatch\nBEFORE={before_md5}\nAFTER ={after_md5}"
    )
