import os
import logging
import numpy as np
import torch
import json

from .core import (
    InfluenceCalculator,
    InfluenceCalculatorFactory,
)
from wie.models.networks import get_network
from wie.infl.dve_embeddings import build_embeddings


@InfluenceCalculatorFactory.register("dve")
class DVEInfluenceCalculator(InfluenceCalculator):
    """
    Data Value Embedding (DVE) based influence calculator.

    DVE computes influence scores by:
    1. Building embeddings from training (requires DVE enabled during training)
    2. Computing validation gradient on final model
    3. Projecting gradient using the same projection matrix R
    4. Computing dot product between projected gradient and embeddings
    """

    def __init__(self, infl_type: str, **kwargs):
        super().__init__(infl_type, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Check if DVE artifacts exist
        self.dve_dir = os.path.join(self.dn, "records", "dve")
        self.dve_raw_dir = os.path.join(self.dn, "records", "dve_raw")

        if not os.path.exists(self.dve_dir):
            os.makedirs(self.dve_dir, exist_ok=True)

        # Load or build DVE embeddings
        self._ensure_dve_embeddings()

        # Load projection matrix
        self.projection = self._load_projection()

        # Load final model
        self.final_model = self._load_final_model()

    def _get_infl_type(self) -> str:
        return "dve"

    def _ensure_dve_embeddings(self):
        """Check if DVE embeddings exist, build them if necessary."""
        embeddings_path = os.path.join(self.dve_dir, "embeddings.pt")
        memmap_path = os.path.join(self.dve_dir, "embeddings.memmap")

        if not os.path.exists(embeddings_path) and not os.path.exists(memmap_path):
            self.logger.info("DVE embeddings not found. Building them now...")

            # Check if we have the necessary raw data to build embeddings
            if not os.path.exists(self.dve_raw_dir):
                # Search for DVE files in alternative locations
                self._search_for_dve_files()

            raw_files = []
            if os.path.exists(self.dve_raw_dir):
                raw_files = [
                    f for f in os.listdir(self.dve_raw_dir) if f.endswith(".pt")
                ]

            if not raw_files:
                # Provide detailed diagnostic information
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
        global_info_path = os.path.join(self.dn, "global_info.json")
        if os.path.exists(global_info_path):
            try:
                with open(global_info_path, "r") as f:
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
        """Build DVE embeddings using DVE.py build_embeddings function."""
        # Check if raw shards exist
        if not os.path.exists(self.dve_raw_dir):
            raise RuntimeError(
                f"DVE raw shards not found at {self.dve_raw_dir}. "
                "Please ensure DVE was enabled during training (--dve_enable True)."
            )

        self.logger.info(f"Building DVE embeddings from shards in {self.dve_raw_dir}")

        try:
            # Build final embeddings (all epochs) - default behavior
            summary = build_embeddings(
                base_save_dir=self.dn,
                eta_scale=1.0,  # Default eta scale
                out_memmap=True,
                export_pt=True,
                prefer_fp16_memmap=True,
                device=self.device if self.device.startswith("cuda") else "cpu",
            )

            self.logger.info("DVE embeddings built successfully")
            self.logger.debug(f"DVE build summary: {json.dumps(summary, indent=2)}")

        except Exception as e:
            self.logger.error(f"Failed to build DVE embeddings: {e}")
            raise RuntimeError(f"Failed to build DVE embeddings: {e}")

    def _load_projection(self) -> torch.Tensor:
        """Load the projection matrix R."""
        proj_path = os.path.join(self.dve_dir, "projection_last_layer.pt")
        if not os.path.exists(proj_path):
            raise FileNotFoundError(
                f"Projection matrix not found at {proj_path}. "
                "Ensure DVE was enabled during training."
            )

        # Use CPU for loading, then move to device if needed
        R = torch.load(proj_path, map_location="cpu")
        if self.device != "cpu" and not self.device.endswith(":-1"):
            R = R.to(self.device)
        self.logger.info(f"Loaded projection matrix with shape {R.shape}")
        return R

    def _load_final_model(self) -> torch.nn.Module:
        """Load the final trained model."""
        from wie.infl.core import load_final_model

        state = load_final_model(self.dn, self.seed, self.device, self.logger)
        model = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )

        try:
            result = model.load_state_dict(state, strict=False)
            # Handle both None return (when successful) and tuple return (missing, unexpected)
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

    def _get_last_layer_gradient(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Extract gradient of last linear layer for given input."""
        # Ensure logger exists for minimal/new instances
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.final_model.zero_grad()

        # Ensure inputs are on the same device as the model parameters
        try:
            model_device = next(self.final_model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        x = x.to(model_device)
        y = y.to(model_device)

        # Forward and backward pass
        output = self.final_model(x)
        loss = self.loss_fn(output, y)
        loss.backward()

        # Find last linear layer
        last_linear = None
        for module in self.final_model.modules():
            if isinstance(module, torch.nn.Linear):
                last_linear = module

        if last_linear is None:
            raise RuntimeError("No Linear layer found in model")

        # Extract weight and bias gradients and concatenate
        if last_linear.weight.grad is None:
            raise RuntimeError("No weight gradients found in last layer")
        g_w = last_linear.weight.grad.flatten().detach()
        # if last_linear.bias is not None:
        #     g_b = (
        #         last_linear.bias.grad.flatten().detach()
        #         if last_linear.bias.grad is not None
        #         else torch.zeros_like(last_linear.bias).flatten()
        #     )
        #     g_last = torch.cat([g_w, g_b], dim=0)
        # else:
        #     g_last = g_w
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.debug(
                f"Last layer weight gradient shape: {last_linear.weight.grad.shape}, flattened: {g_w.shape}"
            )

        return g_w

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

    def calculate(self) -> np.ndarray:
        """
        Calculate DVE-based influence scores.

        For validation data, computes:
        score_i = <g_val_proj, E_i>
        where g_val_proj = R @ g_val_last
        """
        # Load embeddings
        E = self._load_embeddings()  # [N, d]
        self.logger.info(f"Loaded embeddings with shape {E.shape}")

        # Compute validation gradient (using all validation data)
        self.logger.info("Computing validation gradient on final model...")

        # Use batched computation for efficiency
        batch_size = min(128, self.x_val.shape[0])
        all_scores = torch.zeros(self.n_tr, device=self.device)

        n_val_samples = self.x_val.shape[0]
        for start_idx in range(0, n_val_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_val_samples)
            x_batch = self.x_val[start_idx:end_idx]
            y_batch = self.y_val[start_idx:end_idx]

            # Get gradient for this batch
            g_last = self._get_last_layer_gradient(x_batch, y_batch)  # [p_last]

            # Project gradient
            # projection is [d, p_last], g_last is [p_last] -> result is [d]
            # If dimensions don't match, we need to check projection matrix orientation
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

            # Compute scores via dot product
            batch_scores = torch.matmul(E, g_proj)  # [N]
            all_scores += batch_scores

        # Average over validation samples
        all_scores /= n_val_samples

        # Convert to numpy
        scores_np = all_scores.detach().cpu().numpy().astype(np.float32)

        self.logger.info(f"Computed DVE scores for {len(scores_np)} training samples")
        self.logger.info(
            f"Score statistics: mean={scores_np.mean():.6f}, "
            f"std={scores_np.std():.6f}, "
            f"min={scores_np.min():.6f}, max={scores_np.max():.6f}"
        )

        return scores_np

    def _scores_for_model(self, model: torch.nn.Module) -> np.ndarray:
        """
        Compute DVE scores using provided model instead of final model.
        This is used for per-epoch evaluation.
        """
        # Temporarily replace final model
        orig_model = self.final_model
        self.final_model = model

        try:
            # Reuse main calculation logic
            return self.calculate()
        finally:
            # Restore original model
            self.final_model = orig_model


__all__ = ["DVEInfluenceCalculator"]
