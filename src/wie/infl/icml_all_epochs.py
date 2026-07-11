import numpy as np
import torch
import torch.optim as optim
import gc
from typing import List, Tuple

from wie.models.networks import get_network  # type: ignore
from wie.infl.vis import sum_norm  # type: ignore
from .core import (
    InfluenceCalculator,
    InfluenceCalculatorFactory,
    compute_gradient,
    load_step_data,
    load_epoch_data,
    BATCH_SIZE_ICML,
    LR_ICML,
    NUM_EPOCHS_ICML,
)


@InfluenceCalculatorFactory.register("icml_all_epochs")
class ICMLAllEpochsInfluenceCalculator(InfluenceCalculator):
    """
    Computes ICML'17 influence for each epoch interval using epoch-level score differencing.
    For epoch k, computes cumulative influence scores S_k using the model state at the end of epoch k,
    then calculates incremental influence as: infl[k] = S_k - S_{k-1} (for k > 0) or S_0 (for k = 0).

    This approach mimics dve_all_epochs by computing the marginal influence contribution
    of each epoch through differencing of cumulative scores.
    """

    def __init__(self, infl_type: str, **kwargs):
        super().__init__(infl_type, **kwargs)
        # Store kwargs for configuration access
        self.kwargs = kwargs

    def _get_infl_type(self) -> str:
        return "icml_all_epochs"

    def calculate(self) -> List[np.ndarray]:
        self.logger.info(
            "Starting ICML All Epochs influence calculation with epoch-level score differencing..."
        )

        # Try epochs-only mode first, fallback to step-by-step if needed
        if self._has_epoch_files():
            self.logger.info("Found epoch files, using epochs-only mode")
            return self._calculate_epochs_only_with_differencing()
        else:
            self.logger.info("No epoch files found, using step-by-step mode")
            return self._calculate_step_by_step_with_differencing()

    def _has_epoch_files(self) -> bool:
        """Check if epoch files are available for all epochs"""
        try:
            for epoch_idx in range(self.num_epoch):
                epoch_data = load_epoch_data(
                    self.dn, epoch_idx, self.seed, self.relabel_percentage, self.logger
                )
                if not isinstance(epoch_data, dict) or "model_state" not in epoch_data:
                    return False
            return True
        except Exception:
            return False

    def _calculate_epochs_only_with_differencing(self) -> List[np.ndarray]:
        """Calculate influence using only epoch .pt files with epoch-level score differencing"""
        all_epoch_infl: List[np.ndarray] = []
        prev_scores = None

        for epoch_idx in range(self.num_epoch):
            self.logger.info(
                f"--- Calculating ICML Influence for Epoch {epoch_idx} (epochs-only mode with differencing) ---"
            )

            try:
                # Load epoch data containing model state
                epoch_data = load_epoch_data(
                    self.dn, epoch_idx, self.seed, self.relabel_percentage, self.logger
                )

                if not isinstance(epoch_data, dict) or "model_state" not in epoch_data:
                    self.logger.warning(f"Epoch {epoch_idx}: No valid epoch data found")
                    infl_epoch = np.zeros(self.n_tr, dtype=np.float64)
                    all_epoch_infl.append(infl_epoch)
                    continue

                # Load the model state for this epoch (end-of-epoch state)
                model = get_network(
                    self.model_type, self.input_dim, logger=self.logger
                ).to(self.device)
                model.load_state_dict(epoch_data["model_state"])
                model.eval()

                # Compute validation gradient u at this epoch's model state
                u = compute_gradient(self.x_val, self.y_val, model, self.loss_fn)
                u = [uu.to(self.device) for uu in u]

                # Optimize v ≈ H^{-1}u using the epoch model
                v = self._optimize_v_for_epoch(model, u, epoch_idx)

                # Compute cumulative influence scores using v (representing training up to epoch_idx)
                current_scores = self._compute_influence_from_v(model, v)

                # Compute incremental influence for this epoch using differencing
                if prev_scores is None:
                    # First epoch: incremental = cumulative
                    infl_epoch = current_scores.copy()
                else:
                    # Later epochs: incremental = current - previous
                    infl_epoch = current_scores - prev_scores

                self.logger.info(
                    f"Epoch {epoch_idx} ICML influence scores: "
                    f"mean={infl_epoch.mean():.6f}, "
                    f"std={infl_epoch.std():.6f}, "
                    f"min={infl_epoch.min():.6f}, "
                    f"max={infl_epoch.max():.6f}"
                )

                all_epoch_infl.append(infl_epoch)
                prev_scores = current_scores

                del model, u, v

            except Exception as e:
                self.logger.error(f"Epoch {epoch_idx}: Error: {e}", exc_info=True)
                infl_epoch = np.zeros(self.n_tr, dtype=np.float64)
                all_epoch_infl.append(infl_epoch)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.logger.info(
            "ICML All Epochs (epochs-only mode with differencing) calculation finished."
        )
        return all_epoch_infl

    def _calculate_step_by_step_with_differencing(self) -> List[np.ndarray]:
        """Calculate influence using step-by-step model loading (fallback method) with differencing"""
        all_epoch_infl: List[np.ndarray] = []
        prev_scores = None

        for epoch_idx in range(self.num_epoch):
            self.logger.info(
                f"--- Calculating ICML Influence for Epoch {epoch_idx} (step-by-step with differencing) ---"
            )

            # Get the step number at the end of this epoch
            end_step = min((epoch_idx + 1) * self.steps_per_epoch, self.total_steps)

            try:
                # Load model state at end of epoch
                model, u = self._load_model_and_u_at_step(end_step)

                # Optimize v ≈ H^{-1}u using the epoch model
                v = self._optimize_v_for_epoch(model, u, epoch_idx)

                # Compute cumulative influence scores using v (representing training up to epoch_idx)
                current_scores = self._compute_influence_from_v(model, v)

                # Compute incremental influence for this epoch using differencing
                if prev_scores is None:
                    # First epoch: incremental = cumulative
                    infl_epoch = current_scores.copy()
                else:
                    # Later epochs: incremental = current - previous
                    infl_epoch = current_scores - prev_scores

                self.logger.info(
                    f"Epoch {epoch_idx} ICML influence scores: "
                    f"mean={infl_epoch.mean():.6f}, "
                    f"std={infl_epoch.std():.6f}, "
                    f"min={infl_epoch.min():.6f}, "
                    f"max={infl_epoch.max():.6f}"
                )

                all_epoch_infl.append(infl_epoch)
                prev_scores = current_scores

                del model, u, v

            except Exception as e:
                self.logger.error(f"Epoch {epoch_idx}: Error: {e}", exc_info=True)
                infl_epoch = np.zeros(self.n_tr, dtype=np.float64)
                all_epoch_infl.append(infl_epoch)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.logger.info(
            "ICML All Epochs (step-by-step with differencing) calculation finished."
        )
        return all_epoch_infl

    # -----------------------------
    # Helpers (private)
    # -----------------------------
    def _load_model_and_u_at_step(
        self, step: int
    ) -> Tuple[torch.nn.Module, List[torch.Tensor]]:
        """Load model state and compute validation gradient at a specific step"""
        step_data = load_step_data(
            self.dn, step, self.seed, self.relabel_percentage, self.logger
        )
        if (
            not isinstance(step_data, dict)
            or "model_state" not in step_data
            or step_data["model_state"] is None
        ):
            raise FileNotFoundError(
                f"Step file {step} has no model_state; cannot load model."
            )

        model = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )
        model.load_state_dict(step_data["model_state"])
        model.eval()

        u = compute_gradient(self.x_val, self.y_val, model, self.loss_fn)
        u = [uu.to(self.device) for uu in u]

        return model, u

    def _optimize_v_for_epoch(
        self, model: torch.nn.Module, u: List[torch.Tensor], epoch_idx: int
    ) -> List[torch.Tensor]:
        """
        Optimize v ≈ H^{-1}u for a specific epoch using ICML'17 method.
        Fixed to properly implement the objective J(v) = 0.5 * v^T H v - u^T v + 0.5 * λ * ||v||^2
        where H is the Hessian and λ is the damping term (weight decay from training).
        """
        # Use fixed damping based on training weight decay (consistent with ICML'17)
        damping = self.kwargs.get("weight_decay", 0.0)
        if damping == 0.0:
            damping = self.alpha  # fallback to alpha if no weight_decay specified

        # Enforce a conservative upper bound on ICML optimization batch size to avoid OOM
        batch_size_icml = min(
            BATCH_SIZE_ICML, int(self.kwargs.get("icml_max_batch_size", 16))
        )
        num_steps_icml = int(np.ceil(self.n_tr / batch_size_icml))
        v = [uu.clone().detach().requires_grad_(True) for uu in u]

        # Remove momentum for more stable convergence to the linear system solution
        optimizer = optim.SGD(v, lr=LR_ICML, momentum=0.0)

        self.logger.info(
            f"Epoch {epoch_idx}: Optimizing v ≈ H^-1 u ({NUM_EPOCHS_ICML} epochs, {num_steps_icml} steps/epoch), "
            f"damping: {damping:.6f}"
        )

        for opt_epoch in range(NUM_EPOCHS_ICML):
            model.eval()
            np.random.seed(
                opt_epoch + epoch_idx * 1000
            )  # Ensure different seeds per epoch
            idx_list = np.array_split(np.random.permutation(self.n_tr), num_steps_icml)

            for i, idx_batch in enumerate(idx_list):
                idx_tensor = torch.tensor(
                    idx_batch, dtype=torch.long, device=self.device
                )
                x_batch = self.x_tr[idx_tensor]
                y_batch = self.y_tr[idx_tensor]

                z = model(x_batch)
                loss = self.loss_fn(z, y_batch)

                # Critical scaling: adjust mini-batch mean loss to empirical risk scaling
                # This ensures HVP corresponds to H = (1/n_tr) * sum(grad^2 ell_i)
                loss = loss * (len(idx_batch) / self.n_tr)

                # Compute first-order gradients (keeping computational graph)
                g = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)  # type: ignore[arg-type]

                # Compute Hessian-vector product H*v using the correct approach
                # This keeps v in the computational graph, so gradients are computed correctly
                hvp = torch.autograd.grad(
                    g, list(model.parameters()), grad_outputs=v, create_graph=True
                )  # type: ignore[arg-type]

                # Correct objective function: J(v) = 0.5 * v^T H v - u^T v + 0.5 * damping * ||v||^2
                # This will give gradient: ∇J = H v - u + damping * v = (H + damping * I) v - u
                J = torch.tensor(0.0, device=self.device, dtype=u[0].dtype)  # ensure J is a Tensor for backward()/item()
                for h, vv, uu in zip(hvp, v, u):
                    J = (
                        J
                        + 0.5 * (h * vv).sum()
                        - (uu * vv).sum()
                        + 0.5 * damping * (vv * vv).sum()
                    )

                optimizer.zero_grad()
                J.backward()
                optimizer.step()

                # Apply conservative gradient clipping to prevent divergence
                with torch.no_grad():
                    v_norm = sum_norm(v)
                    if v_norm > 1e10:  # More conservative threshold
                        scale = 1e10 / v_norm
                        for idx in range(len(v)):
                            v[idx].mul_(scale)

                if self.tb_writer is not None:
                    global_step = (
                        epoch_idx * NUM_EPOCHS_ICML * num_steps_icml
                        + opt_epoch * num_steps_icml
                        + i
                    )
                    self.tb_writer.add_scalar(
                        f"{self.infl_type}/epoch_{epoch_idx}_optim_loss",
                        J.item(),
                        global_step,
                    )

            if (opt_epoch + 1) % max(1, NUM_EPOCHS_ICML // 5) == 0:
                self.logger.info(
                    f"Epoch {epoch_idx}, Optimization epoch {opt_epoch + 1}/{NUM_EPOCHS_ICML}, "
                    f"Loss: {J.item():.6f}, v norm: {sum_norm(v):.4e}"
                )

        return v

    def _compute_influence_from_v(
        self, model: torch.nn.Module, v: List[torch.Tensor]
    ) -> np.ndarray:
        """Compute final influence scores using optimized v"""
        infl = np.zeros(self.n_tr, dtype=np.float64)

        for i in range(self.n_tr):
            x_i = self.x_tr[[i]]
            y_i = self.y_tr[[i]]
            z = model(x_i)
            loss = self.loss_fn(z, y_i)
            model.zero_grad()
            loss.backward()

            infl_i = 0.0
            with torch.no_grad():
                for j, param in enumerate(model.parameters()):
                    if param.grad is not None and j < len(v):
                        param_grad_device = param.grad.data.to(
                            v[j].device, dtype=v[j].dtype
                        )
                        infl_i += torch.sum(param_grad_device * v[j].data).item()

            infl[i] = -infl_i / self.n_tr

            if (i + 1) % 500 == 0:
                self.logger.info(
                    f"Calculated ICML influence for {i + 1}/{self.n_tr} samples."
                )

        return infl


__all__ = ["ICMLAllEpochsInfluenceCalculator"]
