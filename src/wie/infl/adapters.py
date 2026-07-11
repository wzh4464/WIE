import numpy as np
import torch
import torch.nn as nn
from typing import List

from .core import (
    InfluenceCalculator,
    InfluenceCalculatorFactory,
    load_epoch_data,
)


def _flatten_grads(params: List[torch.nn.Parameter]) -> torch.Tensor:
    grads = [p.grad.flatten() for p in params if p.grad is not None]
    if not grads:
        return torch.tensor([], device=params[0].device if params else "cpu")
    return torch.cat(grads)


class _GradDotBase(InfluenceCalculator):
    """Baseline gradient-dot product influence (single global vector for given model).

    This serves as a light-weight default for single-global methods to enable
    per-epoch delta via the base class. It computes a validation gradient once
    and scores each training sample by the dot-product with its per-sample
    gradient under the provided model.
    """

    val_batch_size: int = 512
    use_last_layer_only: bool = False

    def _val_gradient(self, model: nn.Module) -> torch.Tensor:
        model.eval()
        model.zero_grad()
        n_val = self.x_val.shape[0]
        if n_val <= 0:
            raise ValueError(
                "Validation set is empty; cannot compute validation gradient."
            )
        bsz = min(int(self.val_batch_size), int(n_val))
        # Fixed subset for reproducibility
        torch.manual_seed(42)
        idx = torch.randperm(n_val)[:bsz]
        x_val = self.x_val[idx].to(self.device, non_blocking=True)
        y_val = self.y_val[idx].to(self.device, non_blocking=True)
        out = model(x_val)
        loss = self.loss_fn(out, y_val)
        loss.backward()

        if self.use_last_layer_only:
            last = None
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    last = m
            params = (
                list(last.parameters())
                if last is not None
                else list(model.parameters())
            )
        else:
            params = list(model.parameters())

        g_val = _flatten_grads(params).detach()
        model.zero_grad()
        return g_val

    def _scores_for_model(self, model: nn.Module) -> np.ndarray:  # type: ignore[override]
        g_val = self._val_gradient(model)
        g_val = g_val.to(self.device)

        N = int(self.x_tr.shape[0])
        scores = np.zeros((N,), dtype=np.float32)
        for i in range(N):
            x_i = self.x_tr[i : i + 1].to(self.device, non_blocking=True)
            y_i = self.y_tr[i : i + 1].to(self.device, non_blocking=True)

            model.zero_grad()
            out = model(x_i)
            loss = self.loss_fn(out, y_i)
            loss.backward()

            params = list(model.parameters())
            g_i = _flatten_grads(params)
            # Basic gradient alignment score; subclasses may override sign/normalization policy
            val = (
                torch.dot(g_i, g_val[: g_i.numel()])
                if g_i.numel() and g_val.numel()
                else torch.tensor(0.0, device=self.device)
            )
            scores[i] = float(val.detach().to("cpu"))

            if (i + 1) % 512 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return scores


@InfluenceCalculatorFactory.register("sgd")
class SgdAdapterCalculator(_GradDotBase):
    def _get_infl_type(self) -> str:
        return "sgd"

    def _scores_for_model(self, model: nn.Module) -> np.ndarray:  # type: ignore[override]
        # Reverse-SGD inspired approximation: total LR along path × gradient alignment at current params
        g_val = self._val_gradient(model).to(self.device)

        # Sum learning rates from available epoch files up to current horizon
        total_lr = 0.0
        try:
            horizon = int(
                getattr(self, "_current_epoch", getattr(self, "_total_epochs", 0))
            )
            horizon = max(horizon, 0)
            for e in range(horizon + 1):
                epoch_data = load_epoch_data(
                    self.dn, e, self.seed, self.relabel_percentage, self.logger
                )
                step_info = epoch_data.get("step_info", [])
                if isinstance(step_info, list):
                    for s in step_info:
                        total_lr += float(s.get("lr", 0.0))
        except Exception as ex:
            self.logger.warning(
                f"SGD total LR accumulation failed: {ex}. Using 1.0 as weight."
            )
            total_lr = 1.0

        N = int(self.x_tr.shape[0])
        scores = np.zeros((N,), dtype=np.float32)
        for i in range(N):
            x_i = self.x_tr[i : i + 1].to(self.device, non_blocking=True)
            y_i = self.y_tr[i : i + 1].to(self.device, non_blocking=True)
            model.zero_grad()
            out = model(x_i)
            loss = self.loss_fn(out, y_i)
            loss.backward()
            g_i = _flatten_grads(list(model.parameters()))
            val = (
                torch.dot(g_i, g_val[: g_i.numel()])
                if g_i.numel() and g_val.numel()
                else torch.tensor(0.0, device=self.device)
            )
            scores[i] = float(total_lr * val.detach().to("cpu"))
            if (i + 1) % 512 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return scores


@InfluenceCalculatorFactory.register("icml")
class IcmlAdapterCalculator(_GradDotBase):
    def _get_infl_type(self) -> str:
        return "icml"

    def _scores_for_model(self, model: nn.Module) -> np.ndarray:  # type: ignore[override]
        # Influence functions via H^{-1} v (ICML-style). Solve (H+lambda I) x = v with CG; I[i] = - grad_i^T x
        # 1) Validation gradient v
        v = self._val_gradient(model).to(self.device)

        # 2) Shapes map for (un)flatten
        params = [p for p in model.parameters() if p.requires_grad]
        sizes = [p.numel() for p in params]

        def unflatten(vec: torch.Tensor) -> List[torch.Tensor]:
            out = []
            offset = 0
            for p, n in zip(params, sizes):
                out.append(vec[offset : offset + n].view_as(p).to(p.dtype))
                offset += n
            return out

        # Fixed validation subset so the CG operator A = H + lambda*I stays
        # consistent across iterations (re-sampling a new batch each hvp_fn call
        # changes A and prevents CG from converging).
        n_val = int(self.x_val.shape[0])
        bsz = min(256, max(1, n_val))
        idx = torch.randperm(n_val)[:bsz]
        x_val = self.x_val[idx].to(self.device, non_blocking=True)
        y_val = self.y_val[idx].to(self.device, non_blocking=True)
        lambda_reg = float(getattr(self, "alpha", 0.01) or 0.01)

        def hvp_fn(vec: torch.Tensor) -> torch.Tensor:
            # Exact Hessian-vector product via double backprop (Pearlmutter),
            # then Tikhonov damping: returns (H + lambda*I) @ vec. The previous
            # code called a finite-difference helper that returns a 0-D loss
            # difference (not a per-parameter HVP), which crashed on zip().
            v_list = unflatten(vec)
            model.zero_grad()
            loss = self.loss_fn(model(x_val), y_val)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            hv_list = torch.autograd.grad(
                grads, params, grad_outputs=v_list, retain_graph=False
            )
            hv_flat = torch.cat(
                [
                    (hv.to(self.device) + lambda_reg * vv).reshape(-1)
                    for hv, vv in zip(hv_list, v_list)
                ]
            )
            return hv_flat.detach()

        # Conjugate gradient to solve Ax=b
        b = v.detach().reshape(-1)
        x = torch.zeros_like(b)
        r = b - hvp_fn(x)
        p = r.clone()
        rs_old = torch.dot(r, r)
        max_iter = 10
        eps = 1e-8
        for _ in range(max_iter):
            Ap = hvp_fn(p)
            denom = torch.dot(p, Ap) + eps
            alpha = rs_old / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            if torch.sqrt(rs_new) < 1e-5:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        u = x  # Approx H^{-1} v

        # 3) Per-sample influence: - grad_i^T u
        N = int(self.x_tr.shape[0])
        scores = np.zeros((N,), dtype=np.float32)
        for i in range(N):
            x_i = self.x_tr[i : i + 1].to(self.device, non_blocking=True)
            y_i = self.y_tr[i : i + 1].to(self.device, non_blocking=True)
            model.zero_grad()
            out = model(x_i)
            loss = self.loss_fn(out, y_i)
            loss.backward()
            g_i = torch.cat([p.grad.reshape(-1) for p in params])
            val = (
                -torch.dot(g_i, u[: g_i.numel()])
                if g_i.numel() and u.numel()
                else torch.tensor(0.0, device=self.device)
            )
            scores[i] = float(val.detach().to("cpu"))
            if (i + 1) % 512 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return scores


@InfluenceCalculatorFactory.register("tracin")
class TracinAdapterCalculator(_GradDotBase):
    def _get_infl_type(self) -> str:
        return "tracin"

    def _scores_for_model(self, model: nn.Module) -> np.ndarray:  # type: ignore[override]
        # TracIn-style: accumulate gradient alignment over checkpoints up to current epoch
        try:
            cur_e = int(
                getattr(self, "_current_epoch", getattr(self, "_total_epochs", 0))
            )
        except Exception:
            cur_e = int(self.global_info.get("num_epoch", 1)) - 1
        cur_e = max(cur_e, 0)

        # Choose checkpoints with a stride to limit cost (aim ~5 checkpoints)
        total_epochs = int(self.global_info.get("num_epoch", cur_e + 1))
        if total_epochs <= 0:
            total_epochs = cur_e + 1
        target_count = 5
        stride = max(1, (cur_e + 1) // target_count)
        ckpts = list(range(0, cur_e + 1, stride))
        if ckpts[-1] != cur_e:
            ckpts.append(cur_e)

        # Precompute validation gradients per checkpoint
        val_grads: List[torch.Tensor] = []
        models: List[nn.Module] = []
        for e in ckpts:
            epoch_data = load_epoch_data(
                self.dn, e, self.seed, self.relabel_percentage, self.logger
            )
            state_dict = epoch_data.get("model_state", None)
            if state_dict is None:
                continue
            m_e = self._build_model_from_state(state_dict)
            models.append(m_e)
            g_val_e = self._val_gradient(m_e).to(self.device)
            val_grads.append(g_val_e)

        N = int(self.x_tr.shape[0])
        scores = np.zeros((N,), dtype=np.float32)
        for i in range(N):
            s = 0.0
            for m_e, g_val_e in zip(models, val_grads):
                x_i = self.x_tr[i : i + 1].to(self.device, non_blocking=True)
                y_i = self.y_tr[i : i + 1].to(self.device, non_blocking=True)
                m_e.zero_grad()
                out = m_e(x_i)
                loss = self.loss_fn(out, y_i)
                loss.backward()
                g_i = _flatten_grads(list(m_e.parameters()))
                val = (
                    torch.dot(g_i, g_val_e[: g_i.numel()])
                    if g_i.numel() and g_val_e.numel()
                    else torch.tensor(0.0, device=self.device)
                )
                s += float(val.detach().to("cpu"))
            scores[i] = s
            if (i + 1) % 256 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Cleanup
        for m in models:
            del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return scores
