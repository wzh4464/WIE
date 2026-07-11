import numpy as np
import torch
import gc
import os
import json
from typing import List

from wie.models.networks import get_network  # type: ignore
from .core import (
    InfluenceCalculator,
    InfluenceCalculatorFactory,
)


@InfluenceCalculatorFactory.register("dve_all_epochs")
class DVEAllEpochsInfluenceCalculator(InfluenceCalculator):
    """
    Computes DVE-based influence for each epoch interval.
    For epoch k, computes DVE scores using model at epoch k (representing accumulated
    training from epochs [0, k]), then calculates incremental influence as:
    dve_scores[k] - dve_scores[k-1]

    This gives the marginal influence contribution of epoch k.
    """

    def __init__(self, infl_type: str, **kwargs):
        super().__init__(infl_type, **kwargs)

        # Check if DVE artifacts exist
        self.dve_dir = os.path.join(self.dn, "records", "dve")
        self.dve_raw_dir = os.path.join(self.dn, "records", "dve_raw")

        if not os.path.exists(self.dve_dir):
            os.makedirs(self.dve_dir, exist_ok=True)

        # Load or build DVE embeddings
        self._ensure_dve_embeddings()

        # Load projection matrix
        self.projection = self._load_projection()

        # Load final model for gradient computation
        self.final_model = self._load_final_model()

    def _get_infl_type(self) -> str:
        return "dve_all_epochs"

    def _ensure_dve_embeddings(self):
        """Check if DVE embeddings exist, build them if necessary."""
        embeddings_path = os.path.join(self.dve_dir, "embeddings.pt")
        memmap_path = os.path.join(self.dve_dir, "embeddings.memmap")

        if not os.path.exists(embeddings_path) and not os.path.exists(memmap_path):
            self.logger.info("DVE embeddings not found. Building them now...")

            # Check if we have the necessary raw data to build embeddings
            if not os.path.exists(self.dve_raw_dir):
                self._search_for_dve_files()

            raw_files = []
            if os.path.exists(self.dve_raw_dir):
                raw_files = [
                    f for f in os.listdir(self.dve_raw_dir) if f.endswith(".pt")
                ]

            if not raw_files:
                self._diagnose_missing_dve_files()
                raise RuntimeError(
                    f"DVE raw shards not found at {self.dve_raw_dir} or directory is empty. "
                    "Please ensure DVE was enabled during training (--dve-enable True) and "
                    "that the training completed successfully."
                )

            self.logger.info(f"Found {len(raw_files)} DVE shard files")
            self._build_dve_embeddings()
        else:
            self.logger.info(f"DVE embeddings found at {self.dve_dir}")

    def _search_for_dve_files(self):
        """Search for DVE files in alternative locations."""
        # Search in parent directory or other common locations
        parent_dir = os.path.dirname(self.dn)
        alternative_paths = [
            os.path.join(parent_dir, "records", "dve_raw"),
            os.path.join(self.dn, "dve_raw"),
            os.path.join(self.dn, "..", "records", "dve_raw"),
        ]

        for path in alternative_paths:
            if os.path.exists(path):
                self.logger.info(f"Found DVE raw files at alternative location: {path}")
                # Create symlink or copy files to expected location
                if not os.path.exists(self.dve_raw_dir):
                    os.makedirs(os.path.dirname(self.dve_raw_dir), exist_ok=True)
                    try:
                        os.symlink(path, self.dve_raw_dir)
                        self.logger.info(
                            f"Created symlink from {path} to {self.dve_raw_dir}"
                        )
                        return
                    except OSError:
                        # Symlink failed, try copying
                        import shutil

                        shutil.copytree(path, self.dve_raw_dir)
                        self.logger.info(
                            f"Copied DVE files from {path} to {self.dve_raw_dir}"
                        )
                        return

        self.logger.warning("No DVE files found in alternative locations")

    def _diagnose_missing_dve_files(self):
        """Provide diagnostic information about missing DVE files."""
        self.logger.error("DVE Diagnostic Information:")
        self.logger.error(f"Expected DVE directory: {self.dve_dir}")
        self.logger.error(f"Expected DVE raw directory: {self.dve_raw_dir}")
        self.logger.error(f"Base directory: {self.dn}")

        # Check if training was done with DVE enabled
        import glob

        global_info_pattern = os.path.join(self.dn, "global_info_*.json")
        global_info_files = glob.glob(global_info_pattern)

        if global_info_files:
            try:
                with open(global_info_files[0], "r") as f:
                    global_info = json.load(f)
                dve_enabled = global_info.get("dve_enable", False)
                self.logger.error(f"DVE was enabled during training: {dve_enabled}")
                if not dve_enabled:
                    self.logger.error(
                        "DVE was not enabled during training. Re-run training with --dve-enable"
                    )
            except Exception as e:
                self.logger.error(f"Could not read global_info.json: {e}")
        else:
            self.logger.error("global_info.json not found")

        # List available directories
        if os.path.exists(self.dn):
            subdirs = [
                d
                for d in os.listdir(self.dn)
                if os.path.isdir(os.path.join(self.dn, d))
            ]
            self.logger.error(f"Available subdirectories in {self.dn}: {subdirs}")

    def _build_dve_embeddings(self):
        """Build DVE embeddings for each epoch using DVE.py build_embeddings function."""
        if not os.path.exists(self.dve_raw_dir):
            raise RuntimeError(
                f"DVE raw shards not found at {self.dve_raw_dir}. "
                "Please ensure DVE was enabled during training (--dve_enable True)."
            )

        self.logger.info(
            f"Building per-epoch DVE embeddings from shards in {self.dve_raw_dir}"
        )

        # Build embeddings for each epoch
        for k in range(self.num_epoch):
            out_dir_k = os.path.join(self.dve_dir, f"epoch_{k}")
            if os.path.exists(
                os.path.join(out_dir_k, "embeddings.pt")
            ) or os.path.exists(os.path.join(out_dir_k, "embeddings.memmap")):
                self.logger.info(f"Epoch {k} embeddings already exist, skipping")
                continue

            os.makedirs(out_dir_k, exist_ok=True)

            try:
                # Build embeddings up to epoch k (including epoch k)
                summary = self._build_embeddings_up_to_epoch(k, out_dir_k)

                self.logger.info(f"Epoch {k} DVE embeddings built successfully")
                self.logger.debug(
                    f"Epoch {k} DVE build summary: {json.dumps(summary, indent=2)}"
                )

            except Exception as e:
                self.logger.error(f"Failed to build epoch {k} DVE embeddings: {e}")
                raise RuntimeError(f"Failed to build epoch {k} DVE embeddings: {e}")

    def _load_projection(self) -> torch.Tensor:
        """Load the projection matrix R."""
        proj_path = os.path.join(self.dve_dir, "projection_last_layer.pt")
        if not os.path.exists(proj_path):
            raise FileNotFoundError(
                f"Projection matrix not found at {proj_path}. "
                "Ensure DVE was enabled during training."
            )

        R = torch.load(proj_path, map_location="cpu")
        if self.device != "cpu" and not self.device.endswith(":-1"):
            R = R.to(self.device)
        self.logger.info(f"Loaded projection matrix with shape {R.shape}")
        return R

    def _build_embeddings_up_to_epoch(self, target_epoch: int, out_dir: str) -> dict:
        """Build DVE embeddings up to a specific epoch (inclusive).

        This method replicates the build_embeddings logic but only processes
        shards up to the specified epoch.
        """
        from wie.infl.dve_embeddings import (
            _find_shards,
            _detect_mode_and_sort,
            _read_projection,
            _infer_N_from_shards,
        )
        import numpy as np

        # Load projection to get d
        R = _read_projection(self.dve_dir)  # [d, p_last]
        d = int(R.shape[0])

        # Scan shards
        shard_paths = _find_shards(self.dve_raw_dir)
        if not shard_paths:
            raise FileNotFoundError(
                f"No shards found in {self.dve_raw_dir}. Did you enable DVE in train.py?"
            )

        mode, timeline = _detect_mode_and_sort(shard_paths)

        # Filter timeline to only include shards up to target_epoch
        filtered_timeline = []
        for time_key, path, meta in timeline:
            epoch = meta.get("epoch", -1)
            if epoch <= target_epoch:
                filtered_timeline.append((time_key, path, meta))

        if not filtered_timeline:
            self.logger.warning(f"No shards found for epochs 0-{target_epoch}")
            # Create empty embeddings
            n_tr = self.n_tr
            E_empty = torch.zeros((n_tr, d), dtype=torch.float32)
            torch.save(E_empty, os.path.join(out_dir, "embeddings.pt"))
            return {"meta": {"n_tr": n_tr, "d": d}, "totals": {"total_shards": 0}}

        # Infer N from filtered shards
        n_tr = _infer_N_from_shards(filtered_timeline)
        if n_tr <= 0:
            n_tr = self.n_tr

        # Prepare output
        emb_path = os.path.join(out_dir, "embeddings.memmap")
        emb_dtype = np.float16  # use fp16 for efficiency
        torch_compute_dtype = torch.float32

        E_mem = np.memmap(emb_path, dtype=emb_dtype, mode="w+", shape=(n_tr, d))
        E_mem[:] = 0

        # Running matrix M (small dxd)
        device = self.device if self.device.startswith("cuda") else "cpu"
        M = torch.zeros((d, d), dtype=torch_compute_dtype, device=device)

        # Backward traversal through filtered timeline
        total_samples = 0
        total_shards = 0
        eta_scale = 1.0

        for _, path, meta in reversed(filtered_timeline):
            shard = torch.load(path, map_location="cpu")
            U = shard["U"]  # [B, d]
            idx = shard["idx"]  # list[int]
            lr = float(shard.get("lr", 0.0))

            # Normalize dtype and device
            if U.dtype == torch.float16:
                U = U.to(torch_compute_dtype)
            else:
                U = U.float()
            U = U.to(device)

            # Compute eta
            eta = lr if lr > 0 else eta_scale

            # E = eta * (U - U @ M^T)
            with torch.no_grad():
                UMt = torch.matmul(U, M.t())
                E = eta * (U - UMt)

            # Write E to memmap
            E_np = E.detach().cpu().numpy().astype(emb_dtype, copy=False)
            for row, j in enumerate(idx):
                if 0 <= j < n_tr:
                    E_mem[j, :] += E_np[row]

            # Update M ← M + E^T @ U
            with torch.no_grad():
                S = torch.matmul(E.t(), U)
                M += S

            total_samples += len(idx)
            total_shards += 1

            del shard, U, E, UMt, S
            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.empty_cache()

        # Flush memmap
        if hasattr(E_mem, "flush"):
            E_mem.flush()

        # Export .pt file
        export_path = os.path.join(out_dir, "embeddings.pt")
        step = max(1, 1_000_000 // max(1, d))
        E_accum = torch.zeros((n_tr, d), dtype=torch_compute_dtype)
        start = 0
        while start < n_tr:
            end = min(n_tr, start + step)
            block = np.array(E_mem[start:end], copy=False)
            E_accum[start:end] = torch.from_numpy(block.astype(np.float32, copy=False))
            start = end
        torch.save(E_accum, export_path)
        del E_accum

        # Save build info
        info_path = os.path.join(out_dir, "dve_build_info.json")
        summary = {
            "meta": {
                "n_tr": n_tr,
                "d": d,
                "target_epoch": target_epoch,
                "mode": mode,
            },
            "totals": {
                "total_shards": total_shards,
                "total_samples_seen": total_samples,
            },
            "paths": {
                "embeddings_memmap": emb_path,
                "embeddings_pt": export_path,
            },
        }

        try:
            with open(info_path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to write {info_path}: {e}")

        return summary

    def _load_embeddings(self, epoch_idx: int = None) -> torch.Tensor:
        """Load pre-computed DVE embeddings.

        Args:
            epoch_idx: If specified, load embeddings for specific epoch.
                      If None, load final embeddings (all epochs).
        """
        # Choose directory based on epoch_idx
        if epoch_idx is not None:
            epoch_dir = os.path.join(self.dve_dir, f"epoch_{epoch_idx}")
            pt_path = os.path.join(epoch_dir, "embeddings.pt")
            memmap_path = os.path.join(epoch_dir, "embeddings.memmap")
            info_path = os.path.join(epoch_dir, "dve_build_info.json")
        else:
            # Default: load final embeddings (all epochs)
            pt_path = os.path.join(self.dve_dir, "embeddings.pt")
            memmap_path = os.path.join(self.dve_dir, "embeddings.memmap")
            info_path = os.path.join(self.dve_dir, "dve_build_info.json")

        # Try .pt file first (easier to load)
        if os.path.exists(pt_path):
            self.logger.info(f"Loading embeddings from {pt_path}")
            return torch.load(pt_path, map_location=self.device)

        # Fall back to memmap

        if not os.path.exists(memmap_path):
            raise FileNotFoundError(
                f"No DVE embeddings found. Expected {pt_path} or {memmap_path}"
            )

        # Load metadata to get shape and dtype
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)
            n_tr = info["meta"]["n_tr"]
            d = info["meta"]["d"]
            dtype_str = info.get("dtype_memmap", "float32")
            dtype = np.float16 if "16" in dtype_str else np.float32
        else:
            # Try to infer from global info and file size
            n_tr = self.n_tr
            d = self.projection.shape[0]

            # Detect dtype based on file size
            actual_size = os.path.getsize(memmap_path)
            expected_fp32 = n_tr * d * 4  # float32
            expected_fp16 = n_tr * d * 2  # float16

            if abs(actual_size - expected_fp16) < abs(actual_size - expected_fp32):
                dtype = np.float16
                self.logger.info("Detected float16 dtype based on file size")
            else:
                dtype = np.float32
                self.logger.info("Detected float32 dtype based on file size")

        # Validate file size before creating memmap
        expected_size = n_tr * d * np.dtype(dtype).itemsize
        actual_size = os.path.getsize(memmap_path)

        if actual_size < expected_size:
            raise ValueError(
                f"Memmap file {memmap_path} is too small. "
                f"Expected {expected_size} bytes for shape ({n_tr}, {d}) with dtype {dtype}, "
                f"but file is only {actual_size} bytes. "
                f"This suggests DVE embeddings were not built correctly."
            )

        self.logger.info(
            f"Loading memmap embeddings: shape=({n_tr}, {d}), dtype={dtype}"
        )
        E_mem = np.memmap(memmap_path, dtype=dtype, mode="r", shape=(n_tr, d))

        # Convert to torch tensor
        E = torch.from_numpy(np.array(E_mem)).float().to(self.device)
        return E

    def _get_last_layer_gradient(
        self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Extract gradient of last linear layer for given input."""
        model.zero_grad()

        # Forward and backward pass
        output = model(x)
        loss = self.loss_fn(output, y)
        loss.backward()

        # Find last linear layer
        last_linear = None
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                last_linear = module

        if last_linear is None:
            raise RuntimeError("No Linear layer found in model")

        # Extract only weight gradients (consistent with DVE training)
        if last_linear.weight.grad is None:
            raise RuntimeError("No weight gradients found in last layer")

        g_last = last_linear.weight.grad.flatten().detach()

        # Verify size matches projection matrix expectation
        if g_last.shape[0] != self.projection.shape[1]:
            raise RuntimeError(
                f"Weight gradient size {g_last.shape[0]} does not match projection size {self.projection.shape[1]}. "
                f"This suggests the model architecture changed between training and inference."
            )

        return g_last

    def _load_epoch_data_safe(self, epoch_idx: int):
        """Load epoch data with CPU mapping to avoid CUDA issues."""
        from wie.io import resolve_epoch_file

        path = resolve_epoch_file(
            self.dn, self.seed, epoch_idx, self.relabel_percentage
        )
        self.logger.debug(f"Attempting to load epoch file: {path}")
        return torch.load(path, map_location="cpu")

    def _load_final_model(self) -> torch.nn.Module:
        """Load the final trained model."""
        from wie.infl.core import load_final_model

        state = load_final_model(self.dn, self.seed, self.device, self.logger)
        model = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )

        try:
            result = model.load_state_dict(state, strict=False)
            if result is not None:
                if isinstance(result, tuple) and len(result) == 2:
                    missing, unexpected = result
                    if missing or unexpected:
                        self.logger.warning(
                            f"Non-strict load_state_dict for final model: "
                            f"missing={len(missing)}, unexpected={len(unexpected)}"
                        )
        except Exception as e:
            self.logger.error(f"Failed to load final model state: {e}")
            raise

        model.eval()
        return model

    def _compute_final_gradient_projection(self) -> torch.Tensor:
        """Compute and cache the final model gradient projection g_proj_T.

        This computes g_proj_T = R @ g_val(θ_T) once and reuses it for all epochs.
        """
        if hasattr(self, "_cached_g_proj_T"):
            return self._cached_g_proj_T

        self.logger.info("Computing final model gradient projection...")

        # Use batched computation for efficiency
        batch_size = min(128, self.x_val.shape[0])
        accumulated_g_proj = torch.zeros(self.projection.shape[0], device=self.device)

        n_val_samples = self.x_val.shape[0]
        for start_idx in range(0, n_val_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_val_samples)
            x_batch = self.x_val[start_idx:end_idx]
            y_batch = self.y_val[start_idx:end_idx]

            # Get gradient for this batch using FINAL model
            g_last = self._get_last_layer_gradient(
                self.final_model, x_batch, y_batch
            )  # [p_last]

            # Project gradient
            # projection is [d, p_last], g_last is [p_last] -> result is [d]
            if self.projection.shape[1] != g_last.shape[0]:
                # Try transposed projection
                if self.projection.shape[0] == g_last.shape[0]:
                    self.logger.warning(
                        f"Projection matrix orientation issue. Using transpose. "
                        f"projection: {self.projection.shape}, g_last: {g_last.shape}"
                    )
                    g_proj = torch.matmul(self.projection.t(), g_last).float()  # [d]
                else:
                    raise RuntimeError(
                        f"Dimension mismatch: projection {self.projection.shape}, "
                        f"gradient {g_last.shape}. Cannot perform matrix multiplication."
                    )
            else:
                g_proj = torch.matmul(self.projection, g_last).float()  # [d]

            accumulated_g_proj += g_proj

        # Average over validation samples
        self._cached_g_proj_T = accumulated_g_proj / n_val_samples

        self.logger.info(
            f"Final gradient projection computed with shape {self._cached_g_proj_T.shape}"
        )
        return self._cached_g_proj_T

    def _compute_dve_scores_with_embeddings(
        self, embeddings: torch.Tensor
    ) -> np.ndarray:
        """Compute DVE scores using provided embeddings and the cached final gradient projection.

        Args:
            embeddings: The embeddings E^(k) for epoch k

        Returns:
            DVE scores for all training samples
        """
        # Get the final gradient projection (computed once and cached)
        g_proj_T = self._compute_final_gradient_projection()

        # Compute scores via dot product: E^(k) @ g_proj_T
        all_scores = torch.matmul(embeddings, g_proj_T)  # [N]

        # Convert to numpy
        scores_np = all_scores.detach().cpu().numpy().astype(np.float32)
        return scores_np

    def calculate(self) -> List[np.ndarray]:
        """
        Calculate DVE-based influence for each epoch interval using epoch-wise embeddings.

        Returns:
            List of numpy arrays, one for each epoch, containing the incremental
            DVE influence scores for that epoch.
        """
        self.logger.info(
            "Starting DVE All Epochs influence calculation with epoch-wise embeddings..."
        )

        all_epoch_infl: List[np.ndarray] = []
        prev_scores = None

        for epoch_idx in range(self.num_epoch):
            self.logger.info(f"--- Calculating DVE Influence for Epoch {epoch_idx} ---")

            try:
                # Load embeddings for this epoch E^(k)
                embeddings_k = self._load_embeddings(epoch_idx)
                self.logger.info(
                    f"Loaded embeddings for epoch {epoch_idx} with shape {embeddings_k.shape}"
                )

                # Compute DVE scores using E^(k) and final gradient g_val(θ_T)
                current_scores = self._compute_dve_scores_with_embeddings(embeddings_k)

                # Compute incremental influence for this epoch
                if prev_scores is None:
                    # First epoch: incremental = cumulative
                    infl_epoch = current_scores.copy()
                else:
                    # Later epochs: incremental = current - previous
                    infl_epoch = current_scores - prev_scores

                self.logger.info(
                    f"Epoch {epoch_idx} DVE scores: "
                    f"mean={infl_epoch.mean():.6f}, "
                    f"std={infl_epoch.std():.6f}, "
                    f"min={infl_epoch.min():.6f}, "
                    f"max={infl_epoch.max():.6f}"
                )

                all_epoch_infl.append(infl_epoch)
                prev_scores = current_scores

                # Cleanup
                del embeddings_k

            except Exception as e:
                self.logger.error(f"Epoch {epoch_idx}: Error: {e}", exc_info=True)
                # Use zeros for this epoch
                infl_epoch = np.zeros(self.n_tr, dtype=np.float32)
                all_epoch_infl.append(infl_epoch)

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.logger.info("DVE All Epochs calculation finished.")
        return all_epoch_infl


__all__ = ["DVEAllEpochsInfluenceCalculator"]
