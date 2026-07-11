import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .core import (
    InfluenceCalculator,
    InfluenceCalculatorFactory,
)


@InfluenceCalculatorFactory.register("td_influence")
class TDInfluenceCalculator(InfluenceCalculator):
    """
    Temporal-Dependence Influence calculator (TD-Influence).
    Ported under src while keeping external API stable.
    """

    def __init__(self, infl_type: str, **kwargs):
        super().__init__(infl_type, **kwargs)

        # Hyperparameters
        self.lambda_trace = kwargs.get("lambda_trace", 0.9)
        self.gamma = kwargs.get("gamma", 0.97)
        self.alpha = kwargs.get("alpha", [1.0, 0.5, 0.25, 0.25])
        self.norm_type = kwargs.get("norm_type", "cosine")
        self.use_projection = kwargs.get("use_projection", False)
        self.proj_dim = kwargs.get("proj_dim", 128)
        self.proj_type = kwargs.get("proj_type", "gaussian")
        self.use_last_layer_only = kwargs.get("use_last_layer_only", True)
        self.val_batch_size = kwargs.get("val_batch_size", 512)
        self.e_clip = kwargs.get("e_clip", 10.0)

        if isinstance(self.alpha, (int, float)):
            self.alpha = [self.alpha] * 4
        elif len(self.alpha) != 4:
            self.alpha = self.alpha[:4] + [0.25] * (4 - len(self.alpha))

        self.logger.info(
            f"TD-Influence: lambda={self.lambda_trace}, gamma={self.gamma}, alpha={self.alpha}"
        )

    def _get_infl_type(self) -> str:
        return "td_influence"

    # ---- Helpers: model & factorized gradients ----
    # Model building is handled by the base class; this calculator focuses on scoring.

    def _find_last_linear(self, model: nn.Module) -> nn.Linear:
        last_linear = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            raise RuntimeError("No nn.Linear layer found for factorized TD-Influence")
        return last_linear

    def _extract_last_layer_factors(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        last_fc = self._find_last_linear(model)
        saved_a: Optional[torch.Tensor] = None

        def fwd_hook(module, inp, out):
            nonlocal saved_a
            saved_a = inp[0].detach()

        h1 = last_fc.register_forward_hook(fwd_hook)
        with torch.no_grad():
            logits = model(x)
        h1.remove()

        if saved_a is None:
            raise RuntimeError(
                "Failed to capture last-layer input activation via forward hook"
            )

        # Compute delta analytically from logits
        if logits.dim() == 2 and logits.size(1) == 1:
            delta = torch.sigmoid(logits) - y
        else:
            probs = torch.softmax(logits, dim=1)
            if y.dim() == 2 and y.size(1) > 1:
                y_idx = torch.argmax(y.long(), dim=1, keepdim=True)
            else:
                y_idx = y.long().view(-1, 1)
            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(1, y_idx, 1)
            delta = probs - one_hot

        a = saved_a.reshape(saved_a.size(0), -1)[0].to(self.device)  # (d_in,)
        d = delta.reshape(delta.size(0), -1)[0].to(self.device)  # (d_out,)
        return a, d

    def _compute_validation_factors(
        self, model: nn.Module, val_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        last_fc = self._find_last_linear(model)
        saved_a: Optional[torch.Tensor] = None

        def fwd_hook(module, inp, out):
            nonlocal saved_a
            saved_a = inp[0].detach()

        x_val_batch = self.x_val[val_indices].to(self.device, non_blocking=True)
        y_val_batch = self.y_val[val_indices].to(self.device, non_blocking=True)

        h1 = last_fc.register_forward_hook(fwd_hook)
        with torch.no_grad():
            logits = model(x_val_batch)
        h1.remove()

        if saved_a is None:
            raise RuntimeError(
                "Failed to capture validation last-layer inputs via forward hook"
            )

        if logits.dim() == 2 and logits.size(1) == 1:
            delta = torch.sigmoid(logits) - y_val_batch
        else:
            probs = torch.softmax(logits, dim=1)
            if y_val_batch.dim() == 2 and y_val_batch.size(1) > 1:
                y_idx = torch.argmax(y_val_batch.long(), dim=1, keepdim=True)
            else:
                y_idx = y_val_batch.long().view(-1, 1)
            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(1, y_idx, 1)
            delta = probs - one_hot

        A_val = saved_a  # (B, d_in)
        D_val = delta  # (B, d_out)
        return A_val, D_val

    def _maybe_setup_projection_factors(
        self, d_in: int, d_out: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_projection:
            return None, None
        torch.manual_seed(42)
        k_a = int(min(self.proj_dim, d_in))
        k_d = int(min(self.proj_dim, max(1, d_out)))

        def gaussian(k: int, d: int) -> torch.Tensor:
            return torch.randn(k, d, device=self.device, dtype=torch.float32) / (k**0.5)

        def achlioptas(k: int, d: int) -> torch.Tensor:
            s = 3
            probs = torch.rand(k, d, device=self.device)
            R = torch.zeros(k, d, device=self.device)
            R[probs < (1.0 / (2 * s))] = 1.0
            R[(probs >= (1.0 / (2 * s))) & (probs < (1.0 / s))] = -1.0
            R = R * ((s / k) ** 0.5)
            return R

        maker = gaussian if self.proj_type == "gaussian" else achlioptas
        R_a = maker(k_a, d_in)
        R_d = maker(k_d, max(1, d_out))
        self.logger.info(
            f"TD-Influence projection (factors): R_a {tuple(R_a.shape)}, R_d {tuple(R_d.shape)}, type={self.proj_type}"
        )
        return R_a, R_d

    @torch.no_grad()
    def _compute_factorized_cosine(
        self,
        a_i: torch.Tensor,
        d_i: torch.Tensor,
        A_val: torch.Tensor,
        D_val: torch.Tensor,
        R_a: Optional[torch.Tensor] = None,
        R_d: Optional[torch.Tensor] = None,
    ) -> float:
        eps = 1e-12
        if R_a is not None and R_d is not None:
            a_eff = R_a @ a_i  # (k_a,)
            d_eff = R_d @ d_i  # (k_d,)
            A_eff = A_val @ R_a.t()  # (B, k_a)
            D_eff = D_val @ R_d.t()  # (B, k_d)
        else:
            a_eff, d_eff, A_eff, D_eff = a_i, d_i, A_val, D_val

        num_vec1 = A_eff @ a_eff  # (B,)
        num_vec2 = D_eff @ d_eff  # (B,)
        numerator = torch.sum(num_vec1.view(-1) * num_vec2.view(-1))

        norm_i = torch.norm(a_eff) * torch.norm(d_eff) + eps
        G_a = A_eff @ A_eff.t()  # (B,B)
        G_d = D_eff @ D_eff.t()  # (B,B)
        val_sq = torch.sum(G_a * G_d)
        norm_val = torch.sqrt(torch.clamp(val_sq, min=eps))
        cos = (numerator / (norm_i * norm_val + eps)).item()
        return float(cos)

    def _setup_projection_matrix(self, grad_dim: int) -> Optional[torch.Tensor]:
        if not self.use_projection:
            return None
        torch.manual_seed(42)
        if self.proj_type == "gaussian":
            R = torch.randn(self.proj_dim, grad_dim, device=self.device) / np.sqrt(
                self.proj_dim
            )
        elif self.proj_type == "achlioptas":
            s = 3
            R = torch.zeros(self.proj_dim, grad_dim, device=self.device)
            mask = torch.rand(self.proj_dim, grad_dim) < (s / grad_dim)
            pos_mask = torch.rand(self.proj_dim, grad_dim) < 0.5
            R[mask & pos_mask] = 1.0
            R[mask & ~pos_mask] = -1.0
            R = R * np.sqrt(s / self.proj_dim)
        else:
            raise ValueError(f"Unsupported projection type: {self.proj_type}")
        self.logger.info(
            f"Projection matrix created: {R.shape}, type: {self.proj_type}"
        )
        return R

    def _compute_cosine_similarity(
        self,
        g1: torch.Tensor,
        g2: torch.Tensor,
        projection_matrix: Optional[torch.Tensor] = None,
    ) -> float:
        eps = 1e-12
        if projection_matrix is not None:
            g1 = projection_matrix @ g1
            g2 = projection_matrix @ g2
        num = (g1 * g2).sum()
        den = g1.norm() * g2.norm() + eps
        return (num / den).item()

    def _extract_last_layer_gradients(self, model: torch.nn.Module) -> torch.Tensor:
        last_layer_params = []
        for name, param in model.named_parameters():
            if ("weight" in name or "bias" in name) and param.grad is not None:
                last_layer_params.append(param.grad.flatten())
        if not last_layer_params:
            all_grads = [
                p.grad.flatten() for p in model.parameters() if p.grad is not None
            ]
            return (
                torch.cat(all_grads)
                if all_grads
                else torch.tensor([], device=self.device)
            )
        return torch.cat(last_layer_params)

    def _compute_sample_metrics(
        self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[float, float, bool]:
        model.eval()
        with torch.no_grad():
            logits = model(x)
            loss = self.loss_fn(logits, y).item()
            if logits.dim() > 1 and logits.shape[1] > 1:
                pred_class = torch.argmax(logits, dim=1)
                correct = (pred_class == y.long().flatten()).item()
                y_true = y.long().flatten()
                correct_logit = logits[0, y_true]
                other_logits = torch.cat([logits[0, :y_true], logits[0, y_true + 1 :]])
                max_other_logit = other_logits.max()
                margin = (correct_logit - max_other_logit).item()
            else:
                pred = (logits > 0).long()
                correct = (pred == y.long()).item()
                margin = abs(logits.item())
        return loss, margin, correct

    def _compute_validation_gradient(
        self, model: torch.nn.Module, val_indices: List[int]
    ) -> torch.Tensor:
        model.eval()
        model.zero_grad()
        x_val_batch = self.x_val[val_indices].to(self.device, non_blocking=True)
        y_val_batch = self.y_val[val_indices].to(self.device, non_blocking=True)
        output = model(x_val_batch)
        loss = self.loss_fn(output, y_val_batch)
        loss.backward()
        if self.use_last_layer_only:
            grad_val = self._extract_last_layer_gradients(model)
        else:
            all_grads = [
                p.grad.flatten() for p in model.parameters() if p.grad is not None
            ]
            grad_val = (
                torch.cat(all_grads)
                if all_grads
                else torch.tensor([], device=self.device)
            )
        model.zero_grad()
        return grad_val

    def _td_influence_update(
        self,
        sample_idx: int,
        epoch: int,
        loss_t: float,
        margin_t: float,
        correct_t: bool,
        grad_i_t: torch.Tensor,
        grad_val_t: torch.Tensor,
        projection_matrix: Optional[torch.Tensor],
        states: Dict,
    ) -> None:
        forgetting = (
            1 if (states["last_correct"][sample_idx] == 1 and correct_t == 0) else 0
        )
        dloss_pos = max(0, loss_t - states["loss_last"][sample_idx])
        hard_prob = max(0, -margin_t)
        align_val = -self._compute_cosine_similarity(
            grad_i_t, grad_val_t, projection_matrix
        )
        align_val = max(0, align_val)
        r = (
            self.alpha[0] * dloss_pos
            + self.alpha[1] * forgetting
            + self.alpha[2] * hard_prob
            + self.alpha[3] * align_val
        )
        states["e"][sample_idx] = self.lambda_trace * states["e"][sample_idx] + 1.0
        states["e"][sample_idx] = min(states["e"][sample_idx], self.e_clip)
        states["w"][sample_idx] = self.gamma * states["w"][sample_idx] + r
        states["last_correct"][sample_idx] = 1 if correct_t else 0
        states["loss_last"][sample_idx] = loss_t

    def _scores_for_model(self, model: nn.Module) -> np.ndarray:
        # Fixed validation subset and factorized validation factors for given model
        n_val = self.x_val.shape[0]
        vbs = min(int(self.val_batch_size), int(n_val)) if n_val > 0 else 0
        if vbs <= 0:
            raise ValueError(
                "Validation set is empty; cannot compute TD-Influence alignment."
            )
        torch.manual_seed(42)
        val_indices = torch.randperm(n_val)[:vbs].tolist()
        A_val, D_val = self._compute_validation_factors(model, val_indices)

        # Optional projection matrices for factor space
        R_a, R_d = self._maybe_setup_projection_factors(
            d_in=A_val.size(1), d_out=D_val.size(1)
        )

        N = int(self.x_tr.shape[0])
        scores = np.zeros((N,), dtype=np.float32)
        for i in range(N):
            x_i = self.x_tr[i : i + 1]
            y_i = self.y_tr[i : i + 1]

            # Metrics
            loss_t, margin_t, correct_t = self._compute_sample_metrics(model, x_i, y_i)

            try:
                a_i, d_i = self._extract_last_layer_factors(model, x_i, y_i)
                cos_val = self._compute_factorized_cosine(
                    a_i, d_i, A_val, D_val, R_a, R_d
                )
            except Exception as e:
                self.logger.warning(
                    f"Factorized path failed for sample {i}: {e}. Falling back to full gradients."
                )
                # Validation gradient (full)
                grad_val = self._compute_validation_gradient(model, val_indices)
                # Sample gradient (full)
                model.eval()
                model.zero_grad()  # explicit
                logits = model(x_i)
                loss = self.loss_fn(logits, y_i)
                loss.backward()
                all_grads = [
                    p.grad.flatten() for p in model.parameters() if p.grad is not None
                ]
                grad_i = (
                    torch.cat(all_grads)
                    if all_grads
                    else torch.tensor([], device=self.device)
                )
                model.zero_grad()
                R = (
                    self._setup_projection_matrix(grad_i.numel())
                    if self.use_projection
                    else None
                )
                cos_val = self._compute_cosine_similarity(grad_i, grad_val, R)

            align_val = max(0.0, -float(cos_val))
            hard_prob = max(0.0, -float(margin_t))
            r = self.alpha[2] * hard_prob + self.alpha[3] * align_val
            scores[i] = float(r)

            if (i + 1) % 512 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return scores.astype(np.float32)


__all__ = ["TDInfluenceCalculator"]
