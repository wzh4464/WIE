import os
import json
import glob
import argparse
import numpy as np
import torch
from typing import List, Tuple, Dict, Any

from wie.utils.paths import resolve_output_dir

"""
DVE.py
- Consumes artifacts produced by the updated train.py:
  - records/dve/projection_last_layer.pt  (R: [d, p_last])
  - records/dve_raw/*.pt  (shards with keys: epoch, step, lr, idx, U[B,d])
- Builds per-training-sample Data Value Embeddings (DVE) using a backward recursion:
    E_batch = eta * (U_batch - U_batch @ M^T)
    M <- M + E_batch^T @ U_batch
  where:
    U_batch: [B, d] projected per-sample gradient (last-layer factorized & JL-projected)
    M: [d, d] running matrix
    E_batch: [B, d] final embeddings for that shard
- Writes embeddings as a memory map: records/dve/embeddings.memmap
  and optionally export to records/dve/embeddings.pt
"""


def _load_global_info(base_save_dir: str) -> Dict[str, Any]:
    # pick the first global_info_*.json
    pattern = os.path.join(base_save_dir, "global_info_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return {}
    with open(files[0], "r") as f:
        return json.load(f)


def _find_shards(raw_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(raw_dir, "*.pt")))


def _read_projection(dve_dir: str) -> torch.Tensor:
    path = os.path.join(dve_dir, "projection_last_layer.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Projection not found: {path}")
    R = torch.load(path, map_location="cpu")
    if R.dim() != 2:
        raise ValueError(f"projection_last_layer.pt must be 2-D, but got {R.shape}")
    return R  # [d, p_last]


def _detect_mode_and_sort(
    shard_paths: List[str],
) -> Tuple[str, List[Tuple[int, str, Dict[str, Any]]]]:
    """
    Returns:
      mode: "step" or "epoch"
      timeline: list of (time_key, path, meta)
        - For step mode: time_key = step (non-negative), monotone increasing
        - For epoch mode: time_key = epoch*10000 + part, monotone increasing
    We will process shards in reverse of this order.
    """
    records = []
    has_nonneg_step = False
    has_neg_step = False

    for p in shard_paths:
        d = torch.load(p, map_location="cpu")
        epoch = int(d.get("epoch", -1))
        step = int(d.get("step", -1))
        lr = float(d.get("lr", 0.0))
        idx = d.get("idx", None)
        U = d.get("U", None)
        if idx is None or U is None:
            raise ValueError(f"Shard missing keys idx/U: {p}")
        meta = {
            "epoch": epoch,
            "step": step,
            "lr": lr,
            "n": len(idx),
            "shape": tuple(U.shape),
        }
        if step >= 0:
            has_nonneg_step = True
            time_key = step
        else:
            has_neg_step = True
            # train.py encodes epoch-mode with step = -(epoch*10000 + part_counter + 1)
            # Recover integer 'part' only for sorting
            s = -step - 1
            part = s % 10000
            time_key = epoch * 10000 + part
            meta["part"] = part
        records.append((time_key, p, meta))

    if has_nonneg_step and has_neg_step:
        # Mixed mode is unexpected; we fallback to sorting by time_key anyway
        mode = "mixed"
    elif has_nonneg_step:
        mode = "step"
    else:
        mode = "epoch"

    # sort ascending in time; we'll later traverse backwards
    records.sort(key=lambda x: x[0])
    return mode, records


def _infer_N_from_shards(records: List[Tuple[int, str, Dict[str, Any]]]) -> int:
    max_idx = -1
    for _, path, _ in records:
        d = torch.load(path, map_location="cpu")
        idx = d["idx"]
        if len(idx) > 0:
            local_max = max(idx)
            if local_max > max_idx:
                max_idx = local_max
    return max_idx + 1


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _dtype_from_flag(fp16: bool):
    return np.float16 if fp16 else np.float32


def _torch_dtype(fp16: bool):
    return torch.float16 if fp16 else torch.float32


def build_embeddings(
    base_save_dir: str,
    eta_scale: float = 1.0,
    out_memmap: bool = True,
    export_pt: bool = True,
    prefer_fp16_memmap: bool = True,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Build DVE embeddings from shards under base_save_dir/records/dve_raw.

    Args
    - base_save_dir: folder created by train.py (e.g., ./outputs/myrun)
    - eta_scale: used when shard lr==0 (epoch mode) or as a global multiplier
    - out_memmap: whether to write embeddings as a memmap
    - export_pt: whether to export a single embeddings.pt
    - prefer_fp16_memmap: use float16 for memmap to reduce disk usage
    - device: "cpu" or "cuda" (matmul M is tiny, CPU is fine)

    Returns summary dict with paths and stats.
    """
    records_dir = os.path.join(base_save_dir, "records")
    dve_dir = os.path.join(records_dir, "dve")
    raw_dir = os.path.join(records_dir, "dve_raw")
    _ensure_dir(dve_dir)

    # Load global info (optional)
    ginfo = _load_global_info(base_save_dir)
    n_tr = int(ginfo.get("n_tr") or 0)
    dve_cfg = ginfo.get("dve", {})
    proj_dim_reported = int(dve_cfg.get("proj_dim") or 0)

    # Load projection to get d
    R = _read_projection(dve_dir)  # [d, p_last]
    d = int(R.shape[0])

    # Scan shards
    shard_paths = _find_shards(raw_dir)
    if not shard_paths:
        raise FileNotFoundError(
            f"No shards found in {raw_dir}. Did you enable DVE in train.py?"
        )

    mode, timeline = _detect_mode_and_sort(shard_paths)

    # Infer N if not in global info
    if n_tr <= 0:
        n_tr = _infer_N_from_shards(timeline)

    # Prepare output (memmap or in-memory fallback)
    emb_path = os.path.join(dve_dir, "embeddings.memmap")
    emb_dtype = _dtype_from_flag(prefer_fp16_memmap)
    torch_compute_dtype = torch.float32  # compute in fp32 for stability
    if out_memmap:
        E_mem = np.memmap(emb_path, dtype=emb_dtype, mode="w+", shape=(n_tr, d))
        E_mem[:] = 0
        E_buffer = None
    else:
        E_mem = None
        E_buffer = torch.zeros((n_tr, d), dtype=torch_compute_dtype)

    # Running matrix M (small dxd), stays on CPU (or tiny GPU)
    M = torch.zeros((d, d), dtype=torch_compute_dtype, device=device)

    # Backward traversal
    total_samples = 0
    total_shards = 0

    # Pre-log basic info
    meta_info = {
        "n_tr": n_tr,
        "d": d,
        "mode": mode,
        "proj_dim_reported": proj_dim_reported,
        "num_shards": len(timeline),
        "device": device,
        "eta_scale": eta_scale,
        "out_memmap": out_memmap,
        "export_pt": export_pt,
        "prefer_fp16_memmap": prefer_fp16_memmap,
    }

    # Traverse in reverse (from last to first)
    for _, path, meta in reversed(timeline):
        shard = torch.load(path, map_location="cpu")
        U = shard["U"]  # [B, d], half or float
        idx = shard["idx"]  # list[int]
        lr = float(shard.get("lr", 0.0))
        # normalize dtype and ensure device consistency
        if U.dtype == torch.float16:
            U = U.to(torch_compute_dtype)
        else:
            U = U.float()

        # Move U to the same device as M for computation
        U = U.to(device)

        # compute eta for this shard
        eta = lr if lr > 0 else eta_scale
        # E = eta * (U - U @ M^T)
        # note: U @ M.T -> [B,d], since M is [d,d]
        with torch.no_grad():
            UMt = torch.matmul(U, M.t())
            E = eta * (U - UMt)

        # Write E to output (scatter)
        if out_memmap:
            # chunked scatter to reduce peak RAM
            # (idx list is small, E on CPU)
            E_np = E.detach().cpu().numpy().astype(emb_dtype, copy=False)
            for row, j in enumerate(idx):
                if 0 <= j < n_tr:
                    E_mem[j, :] += E_np[row]
        else:
            # in-memory torch buffer
            # scatter_add on CPU
            if E_buffer.device.type != "cpu":
                E_buffer = E_buffer.cpu()
            # scatter_add for 2D → loop by row (safe)
            for row, j in enumerate(idx):
                E_buffer[j] += E[row].cpu()

        # Update M ← M + E^T @ U
        with torch.no_grad():
            # E^T @ U: (d,B) x (B,d) = (d,d)
            # E and U are already on the correct device
            S = torch.matmul(E.t(), U)
            M += S

        total_samples += len(idx)
        total_shards += 1

        # small GC
        del shard, U, E, UMt, S
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Flush memmap to disk
    if out_memmap and hasattr(E_mem, "flush"):
        E_mem.flush()

    # Optionally export embeddings.pt for convenience
    export_path = None
    if export_pt:
        export_path = os.path.join(dve_dir, "embeddings.pt")
        if out_memmap:
            # load memmap in chunks to avoid RAM spike
            step = max(1, 1_000_000 // max(1, d))  # rough heuristic
            E_accum = torch.zeros((n_tr, d), dtype=torch_compute_dtype)
            start = 0
            while start < n_tr:
                end = min(n_tr, start + step)
                block = np.array(E_mem[start:end], copy=False)  # view
                E_accum[start:end] = torch.from_numpy(
                    block.astype(np.float32, copy=False)
                )
                start = end
            torch.save(E_accum, export_path)
            del E_accum
        else:
            torch.save(E_buffer, export_path)

    # Save a small summary json
    info_path = os.path.join(dve_dir, "dve_build_info.json")
    summary = {
        "meta": meta_info,
        "totals": {
            "total_shards": total_shards,
            "total_samples_seen": total_samples,
        },
        "paths": {
            "embeddings_memmap": emb_path if out_memmap else None,
            "embeddings_pt": export_path,
            "projection": os.path.join(dve_dir, "projection_last_layer.pt"),
        },
        "M_shape": list(M.shape),
        "dtype_compute": "float32",
        "dtype_memmap": "float16" if prefer_fp16_memmap else "float32",
    }
    try:
        with open(info_path, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"[WARN] failed to write {info_path}: {e}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build Data Value Embeddings (DVE) from train.py artifacts."
    )
    parser.add_argument(
        "--base_save_dir",
        type=str,
        required=True,
        help="The SAME directory you passed to train.py's --save_dir, resolved under outputs/. "
        "If you used absolute save_dir in train.py, pass that absolute base path here.",
    )
    parser.add_argument(
        "--eta_scale",
        type=float,
        default=1.0,
        help="Global eta when shard.lr == 0 (epoch-mode) or as a multiplier.",
    )
    parser.add_argument(
        "--out_memmap",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=True,
        help="Write embeddings.memmap (recommended for large N).",
    )
    parser.add_argument(
        "--export_pt",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=True,
        help="Also export embeddings.pt for convenience.",
    )
    parser.add_argument(
        "--prefer_fp16_memmap",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=True,
        help="Use float16 memmap to save disk.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())],
        help="Device for tiny dxd matmuls. CPU is fine; CUDA optional.",
    )
    args = parser.parse_args()

    # Normalize base_save_dir:
    # train.py stores under outputs/<save_dir> if you passed a relative string.
    # Here we accept either:
    #   (1) absolute path to that base folder (e.g., .../outputs/myrun), or
    #   (2) relative name used by train.py (e.g., myrun) -> we resolve to ./outputs/myrun
    base_path = str(resolve_output_dir(args.base_save_dir))

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"base_save_dir not found: {base_path}")

    summary = build_embeddings(
        base_save_dir=base_path,
        eta_scale=args.eta_scale,
        out_memmap=args.out_memmap,
        export_pt=args.export_pt,
        prefer_fp16_memmap=args.prefer_fp16_memmap,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
