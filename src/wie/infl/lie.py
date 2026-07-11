import numpy as np
import torch
import gc
from typing import List

from ..models.networks import get_network
from .vis import sum_norm
from .core import (
    InfluenceCalculator,
    InfluenceCalculatorFactory,
    compute_gradient,
    load_step_data,
    load_epoch_data,
    load_final_model,
    save_results,
)


@InfluenceCalculatorFactory.register("lie")
class LieInfluenceCalculator(InfluenceCalculator):
    """Computes LIE influence state at the end of each epoch and saves list as 'lie_full'."""

    def _get_infl_type(self) -> str:
        return "lie"

    def _save(self, infl_data):
        save_results(
            infl_data,
            self.dn,
            self.seed,
            "lie_full",
            self.logger,
            self.relabel_percentage,
        )
        if self.tb_writer:
            self.tb_writer.close()

    def _lie_helper(self, target_epoch: int) -> np.ndarray:
        """Compute LIE state integrated up to the given epoch.

        Breaks the original long implementation into smaller helpers:
        - initialize u at the end of the target epoch
        - iterate steps backward, accumulate influence, update u stably
        """
        self.logger.debug(f"LIE Helper: Calculating state for epoch {target_epoch}")
        target_step_for_u = min(
            (target_epoch + 1) * self.steps_per_epoch, self.total_steps
        )
        # 1) Initialize u from end-of-epoch state (with fallbacks)
        u = self._init_u_for_epoch_end(target_epoch, target_step_for_u)

        # 2) Reverse through steps accumulating influence
        infl_epoch = np.zeros(self.n_tr, dtype=np.float64)
        start_step_incl = target_step_for_u
        end_step_excl = 0
        m_step = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )

        for t in range(start_step_incl, end_step_excl, -1):
            step_log_prefix = f"LIE Helper Epoch {target_epoch} Step {t}"
            try:
                # Load model+batch for this step
                step_data = load_step_data(
                    self.dn, t, self.seed, self.relabel_percentage, self.logger
                )
                current_model_state = step_data["model_state"]
                idx_raw, lr = step_data["idx"], float(step_data["lr"])  # type: ignore
                m_step.load_state_dict(current_model_state)
                m_step.eval()

                if not isinstance(idx_raw, (list, np.ndarray, torch.Tensor)):
                    idx_raw = [idx_raw]
                idx = torch.as_tensor(idx_raw, device=self.device)
                valid_idx_mask = (idx >= 0) & (idx < self.n_tr)
                idx = idx[valid_idx_mask]
                if idx.numel() == 0:
                    continue
                x_batch, y_batch = self.x_tr[idx], self.y_tr[idx]

                # Per-sample grads
                param_grads_list = self._compute_param_grads_list(
                    m_step, x_batch, y_batch, u[0].dtype
                )
                # Accumulate influence
                self._accumulate_influence(infl_epoch, idx, param_grads_list, u, lr)
                # Update u stably
                u = self._safe_update_u(
                    m_step, x_batch, y_batch, u, lr, step_log_prefix
                )

            except FileNotFoundError:
                self.logger.warning(
                    f"{step_log_prefix}: Step file not found. Skipping."
                )
                continue
            except Exception as e:
                self.logger.error(f"{step_log_prefix}: Error: {e}", exc_info=True)
                continue

        self.logger.debug(
            f"LIE Helper: Finished epoch {target_epoch}, final u norm: {sum_norm(u):.4e}"
        )
        del m_step
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return infl_epoch

    # -----------------------------
    # Helpers (private)
    # -----------------------------
    def _init_u_for_epoch_end(
        self, target_epoch: int, target_step_for_u: int
    ) -> List[torch.Tensor]:
        model_helper = get_network(
            self.model_type, self.input_dim, logger=self.logger
        ).to(self.device)
        try:
            step_data_end = load_step_data(
                self.dn,
                target_step_for_u,
                self.seed,
                self.relabel_percentage,
                self.logger,
            )
            model_helper.load_state_dict(step_data_end["model_state"])
            model_helper.eval()
            u = compute_gradient(self.x_val, self.y_val, model_helper, self.loss_fn)
            u = [uu.to(self.device) for uu in u]
            u = [uu.to(torch.float64) for uu in u]
            self.logger.debug(f"LIE Helper: Computed u from step {target_step_for_u}")
        except FileNotFoundError:
            self.logger.warning(
                f"LIE Helper: Step file {target_step_for_u} not found. Trying epoch file for epoch {target_epoch}."
            )
            try:
                epoch_data_end = load_epoch_data(
                    self.dn,
                    target_epoch,
                    self.seed,
                    self.relabel_percentage,
                    self.logger,
                )
                model_helper.load_state_dict(epoch_data_end["model_state"])
                model_helper.eval()
                u = compute_gradient(self.x_val, self.y_val, model_helper, self.loss_fn)
                u = [uu.to(self.device).to(torch.float64) for uu in u]
                self.logger.debug(
                    f"LIE Helper: Computed u from epoch {target_epoch} fallback file."
                )
            except FileNotFoundError:
                self.logger.warning(
                    f"LIE Helper: Epoch file {target_epoch} missing. Using FINAL model u (approximation)."
                )
                model_helper.load_state_dict(
                    load_final_model(self.dn, self.seed, self.device, self.logger)
                )
                model_helper.eval()
                u = compute_gradient(self.x_val, self.y_val, model_helper, self.loss_fn)
                u = [uu.to(self.device).to(torch.float64) for uu in u]
        finally:
            del model_helper
        return u

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
            for p, th, vp in zip(params, theta, v):
                p.copy_(th - epsilon * vp)
            g_neg = grad_list()
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
        try:
            hvp = self._finite_diff_hvp(m_step, x_batch, y_batch, u)
            # Light damping based on norms
            u_norm = (
                float(sum_norm(u).item())
                if hasattr(sum_norm(u), "item")
                else float(sum_norm(u))
            )
            hu_norm = (
                float(sum_norm(hvp).item())
                if hasattr(sum_norm(hvp), "item")
                else float(sum_norm(hvp))
            )
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
        except Exception as e:
            self.logger.warning(
                f"{step_log_prefix}: Error during HVP update: {e}. Skipping u update."
            )
            return u_prev

    def calculate(self) -> list[np.ndarray]:
        infl_list: list[np.ndarray] = []
        # Include state up to each epoch boundary: 0..num_epoch
        for epoch in range(self.num_epoch + 1):
            self.logger.info(
                f"Calculating LIE influence state integrated up to epoch {epoch}"
            )
            infl = self._lie_helper(epoch)
            infl_list.append(infl)
            self.logger.info(f"Completed LIE influence state up to epoch {epoch}")
        return infl_list
