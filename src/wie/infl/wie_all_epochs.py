import numpy as np
import torch
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
)


@InfluenceCalculatorFactory.register("wie_all_epochs")
class WieAllEpochsInfluenceCalculator(InfluenceCalculator):
    """
    Computes influence for each epoch interval using reverse SGD.
    For epoch k, reverse from step (k+1)*SPE down to k*SPE+1, with u initialized
    as the validation gradient at (k+1)*SPE.
    """

    def _get_infl_type(self) -> str:
        return "wie_all_epochs"

    def calculate(self) -> list[np.ndarray]:
        self.logger.info("Starting WIE All Epochs influence calculation...")

        # Try epochs-only mode first, fallback to step-by-step if needed
        if self._has_epoch_files():
            self.logger.info("Found epoch files, using epochs-only mode")
            return self._calculate_epochs_only()
        else:
            self.logger.info("No epoch files found, using step-by-step mode")
            return self._calculate_step_by_step()

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

    def _calculate_epochs_only(self) -> list[np.ndarray]:
        """Calculate influence using only epoch .pt files with fixed model state per epoch"""
        all_epoch_infl: list[np.ndarray] = []

        for epoch_idx in range(self.num_epoch):
            self.logger.info(
                f"--- Calculating Influence for Epoch {epoch_idx} (epochs-only mode) ---"
            )
            infl_epoch = np.zeros(self.n_tr, dtype=np.float64)

            try:
                # Load epoch data containing model state and step info
                epoch_data = load_epoch_data(
                    self.dn, epoch_idx, self.seed, self.relabel_percentage, self.logger
                )

                if not isinstance(epoch_data, dict) or "model_state" not in epoch_data:
                    self.logger.warning(f"Epoch {epoch_idx}: No valid epoch data found")
                    all_epoch_infl.append(infl_epoch)
                    continue

                # Load the fixed model state for this epoch (end-of-epoch state)
                m_fixed = get_network(
                    self.model_type, self.input_dim, logger=self.logger
                ).to(self.device)
                m_fixed.load_state_dict(epoch_data["model_state"])
                m_fixed.eval()

                # Compute u at this epoch's end state
                u_current = compute_gradient(
                    self.x_val, self.y_val, m_fixed, self.loss_fn
                )
                u_dtype = self._u_dtype()
                u_current = [uu.to(self.device).to(u_dtype) for uu in u_current]

                # Get step info for this epoch
                step_info = epoch_data.get("step_info", [])
                if not step_info:
                    self.logger.warning(f"Epoch {epoch_idx}: No step_info found")
                    all_epoch_infl.append(infl_epoch)
                    del m_fixed
                    continue

                # Reverse through steps within this epoch using fixed model state
                for step_record in reversed(step_info):
                    try:
                        idx_raw = step_record.get("idx", [])
                        lr = float(step_record.get("lr", 0.0))

                        if not isinstance(idx_raw, (list, np.ndarray, torch.Tensor)):
                            idx_raw = [idx_raw]
                        idx = torch.as_tensor(idx_raw, device=self.device)
                        valid_idx_mask = (idx >= 0) & (idx < self.n_tr)
                        idx = idx[valid_idx_mask]

                        if idx.numel() == 0:
                            continue

                        x_batch, y_batch = self.x_tr[idx], self.y_tr[idx]

                        # Compute gradients using the fixed model state
                        param_grads_list = self._compute_param_grads_list(
                            m_fixed, x_batch, y_batch, u_current[0].dtype
                        )

                        # Accumulate influence
                        self._accumulate_influence(
                            infl_epoch, idx, param_grads_list, u_current, lr
                        )

                        # Update u using the fixed model state
                        u_current = self._safe_update_u(
                            m_fixed,
                            x_batch,
                            y_batch,
                            u_current,
                            lr,
                            f"Epoch {epoch_idx} (fixed model)",
                        )

                    except Exception as e:
                        self.logger.error(
                            f"Epoch {epoch_idx} step error: {e}", exc_info=True
                        )
                        continue

                all_epoch_infl.append(infl_epoch)
                del m_fixed, u_current

            except Exception as e:
                self.logger.error(f"Epoch {epoch_idx}: Error: {e}", exc_info=True)
                all_epoch_infl.append(infl_epoch)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.logger.info("WIE All Epochs (epochs-only mode) calculation finished.")
        return all_epoch_infl

    def _calculate_step_by_step(self) -> list[np.ndarray]:
        """Original step-by-step calculation method"""
        all_epoch_infl: list[np.ndarray] = []
        m_step = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )

        for epoch_idx in range(self.num_epoch):
            self.logger.info(f"--- Calculating Influence for Epoch {epoch_idx} ---")
            infl_epoch = np.zeros(self.n_tr, dtype=np.float64)

            start_step_incl = min(
                (epoch_idx + 1) * self.steps_per_epoch, self.total_steps
            )
            end_step_excl = epoch_idx * self.steps_per_epoch
            if start_step_incl <= end_step_excl:
                self.logger.warning(
                    f"Epoch {epoch_idx}: Start step {start_step_incl} <= end step {end_step_excl}. Skipping."
                )
                all_epoch_infl.append(infl_epoch)
                continue

            # Compute u at end-of-epoch state
            try:
                u_current = self._u_at_step(start_step_incl)
            except Exception as e:
                self.logger.error(
                    f"Epoch {epoch_idx}: Failed to get end-of-epoch state at step {start_step_incl}: {e}",
                    exc_info=True,
                )
                all_epoch_infl.append(infl_epoch)
                continue

            # Reverse steps within the epoch
            for t in range(start_step_incl, end_step_excl, -1):
                step_log_prefix = f"Epoch {epoch_idx} Step {t}"
                try:
                    m_step, idx, lr, x_batch, y_batch = self._load_step_model_and_batch(
                        t
                    )
                    if idx.numel() == 0:
                        continue

                    param_grads_list = self._compute_param_grads_list(
                        m_step, x_batch, y_batch, u_current[0].dtype
                    )
                    self._accumulate_influence(
                        infl_epoch, idx, param_grads_list, u_current, lr
                    )
                    u_current = self._safe_update_u(
                        m_step, x_batch, y_batch, u_current, lr, step_log_prefix
                    )
                except Exception as e:
                    self.logger.error(f"{step_log_prefix}: Error: {e}", exc_info=True)
                    continue

                if (start_step_incl - t) % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            all_epoch_infl.append(infl_epoch)
            del u_current
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del m_step
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("WIE All Epochs calculation finished.")
        return all_epoch_infl

    # -----------------------------
    # Helpers (private)
    # -----------------------------
    def _u_at_step(self, step: int) -> List[torch.Tensor]:
        step_data_end = load_step_data(
            self.dn, step, self.seed, self.relabel_percentage, self.logger
        )
        if (
            not isinstance(step_data_end, dict)
            or "model_state" not in step_data_end
            or step_data_end["model_state"] is None
        ):
            raise FileNotFoundError(
                f"Step file {step} has no model_state; cannot initialize u."
            )
        m = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )
        m.load_state_dict(step_data_end["model_state"])
        m.eval()
        u = compute_gradient(self.x_val, self.y_val, m, self.loss_fn)
        u_dtype = self._u_dtype()
        u = [uu.to(self.device).to(u_dtype) for uu in u]
        del m
        return u

    def _load_step_model_and_batch(
        self, t: int
    ) -> Tuple[torch.nn.Module, torch.Tensor, float, torch.Tensor, torch.Tensor]:
        step_data = load_step_data(
            self.dn, t, self.seed, self.relabel_percentage, self.logger
        )
        current_model_state = step_data["model_state"]
        idx_raw, lr = step_data["idx"], float(step_data["lr"])  # type: ignore

        m_step = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )
        m_step.load_state_dict(current_model_state)
        m_step.eval()

        if not isinstance(idx_raw, (list, np.ndarray, torch.Tensor)):
            idx_raw = [idx_raw]
        idx = torch.as_tensor(idx_raw, device=self.device)
        valid_idx_mask = (idx >= 0) & (idx < self.n_tr)
        idx = idx[valid_idx_mask]
        x_batch, y_batch = self.x_tr[idx], self.y_tr[idx]
        return m_step, idx, lr, x_batch, y_batch

    def _compute_param_grads_list(
        self,
        m_step: torch.nn.Module,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        dtype: torch.dtype,
    ) -> List[List[torch.Tensor]]:
        batch_size = x_batch.shape[0]
        param_grads_list: List[List[torch.Tensor]] = []
        for i_local in range(batch_size):
            m_step.zero_grad()
            z_i = m_step(x_batch[[i_local]])
            loss_i = self.loss_fn(z_i, y_batch[[i_local]])
            if self.alpha > 0:
                for p in m_step.parameters():
                    loss_i += 0.5 * self.alpha * (p * p).sum()
            loss_i.backward()
            grad_i = [
                (
                    p.grad.data.clone().to(dtype=dtype)
                    if p.grad is not None
                    else torch.zeros_like(p, dtype=dtype)
                )
                for p in m_step.parameters()
            ]
            param_grads_list.append(grad_i)
        m_step.zero_grad()
        return param_grads_list

    def _accumulate_influence(
        self,
        infl: np.ndarray,
        idx: torch.Tensor,
        param_grads_list: List[List[torch.Tensor]],
        u: List[torch.Tensor],
        lr: float,
    ) -> None:
        batch_size = len(idx)
        for i_local, sample_idx in enumerate(idx.tolist()):
            grad_i = param_grads_list[i_local]
            grad_sum = sum(
                torch.sum(u[j].data * param_grad).item()
                for j, param_grad in enumerate(grad_i)
                if j < len(u)
            )
            infl[sample_idx] += lr * grad_sum / max(batch_size, 1)

    def _finite_diff_hvp(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        v: List[torch.Tensor],
        epsilon: float = 1e-4,
    ) -> List[torch.Tensor]:
        params = [p for p in model.parameters()]
        assert len(params) == len(v), "v must align with model parameters"

        theta = [p.detach().clone() for p in params]

        def grad_list() -> List[torch.Tensor]:
            model.zero_grad()
            out = model(x)
            loss = self.loss_fn(out, y)
            if self.alpha > 0:
                for p in model.parameters():
                    loss += 0.5 * self.alpha * (p * p).sum()
            loss.backward()
            gl = [
                (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
                for p in model.parameters()
            ]
            model.zero_grad()
            return gl

        with torch.no_grad():
            for p, th, vp in zip(params, theta, v):
                p.copy_(th + epsilon * vp)
        g_pos = grad_list()
        with torch.no_grad():
            for p, th, vp in zip(params, theta, v):
                p.copy_(th - epsilon * vp)
        g_neg = grad_list()
        with torch.no_grad():
            for p, th in zip(params, theta):
                p.copy_(th)

        hvp = [(gp - gn) / (2 * epsilon) for gp, gn in zip(g_pos, g_neg)]
        hvp = [hh.to(dtype=v[0].dtype, device=v[0].device) for hh in hvp]
        return hvp

    def _safe_update_u(
        self,
        m_step: torch.nn.Module,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        u: List[torch.Tensor],
        lr: float,
        step_log_prefix: str,
    ) -> List[torch.Tensor]:
        u_prev = [uu.clone() for uu in u]
        # try:
        hvp = self._finite_diff_hvp(m_step, x_batch, y_batch, u)
        sn_u = sum_norm(u)
        u_norm = float(sn_u.item() if isinstance(sn_u, torch.Tensor) else sn_u)
        sn_hu = sum_norm(hvp)
        hu_norm = float(sn_hu.item() if isinstance(sn_hu, torch.Tensor) else sn_hu)
        lambda_reg = 0.0
        if hu_norm > 0:
            lambda_reg = 0.05 * (u_norm / (hu_norm + 1e-12))
        hvp_reg = [hv + lambda_reg * uu for hv, uu in zip(hvp, u)]
        new_u: List[torch.Tensor] = []
        for j in range(len(u)):
            new_u_val = u[j] - lr * hvp_reg[j]
            if torch.isnan(new_u_val).any() or torch.isinf(new_u_val).any():
                new_u.append(u_prev[j])
            else:
                new_u.append(new_u_val)
        return new_u
        # except Exception as e:
        #     self.logger.warning(
        #         f"{step_log_prefix}: Error during HVP update: {e}. Skipping u update."
        #     )
            # return u_prev

    def _u_dtype(self) -> torch.dtype:
        """Select a safe dtype for u on the current device.

        On MPS, float64 is not supported, so we use float32. Otherwise, prefer float64
        for numerical stability.
        """
        try:
            device_type = (
                self.device.type
                if isinstance(self.device, torch.device)
                else str(self.device)
            )
        except Exception:
            device_type = "cpu"
        if device_type == "mps":
            return torch.float32
        return torch.float64
