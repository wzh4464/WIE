import numpy as np
import torch
from typing import List, Tuple

from ..models.networks import get_network
from .core import (
    InfluenceCalculator,
    InfluenceCalculatorFactory,
    compute_gradient,
    load_step_data,
)


@InfluenceCalculatorFactory.register("nohess")
class NoHessInfluenceCalculator(InfluenceCalculator):
    """Computes influence using reverse SGD but ignoring the HVP term (u is fixed)."""

    def _get_infl_type(self) -> str:
        return "nohess"

    def calculate(self) -> np.ndarray:
        """Reverse-SGD accumulation ignoring HVP (u is fixed)."""
        model = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )
        from .core import load_final_model  # local import to avoid cycles

        model.load_state_dict(
            load_final_model(self.dn, self.seed, self.device, self.logger)
        )
        model.eval()

        u = self._init_u(model)
        infl = np.zeros(self.n_tr, dtype=np.float64)

        for t in range(self.total_steps, 0, -1):
            step_log_prefix = f"Step {t}/{self.total_steps}"
            try:
                m_step, idx, lr, x_batch, y_batch = self._load_step_model_and_batch(t)
                if idx.numel() == 0:
                    continue
                param_grads_list = self._compute_param_grads_list(
                    m_step, x_batch, y_batch, u[0].dtype
                )
                self._accumulate_influence(infl, idx, param_grads_list, u, lr)
            except FileNotFoundError:
                self.logger.warning(
                    f"{step_log_prefix}: Step file not found. Skipping step (epoch fallback not standard for nohess)."
                )
                continue
            except Exception as e:
                self.logger.error(
                    f"{step_log_prefix}: Error processing step: {e}", exc_info=True
                )
                continue

            if t % 100 == 0:
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return infl

    # -----------------------------
    # Helpers (private)
    # -----------------------------
    def _init_u(self, model: torch.nn.Module) -> List[torch.Tensor]:
        u = compute_gradient(self.x_val, self.y_val, model, self.loss_fn)
        u = [uu.to(self.device) for uu in u]
        try:
            self.logger.info("Using float64 precision for fixed u vector.")
            return [uu.to(torch.float64) for uu in u]
        except TypeError:
            self.logger.warning(
                "float64 not supported for u vector, falling back to float32."
            )
            return [uu.to(torch.float32) for uu in u]

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
