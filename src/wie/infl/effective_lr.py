"""Appendix E checkpoint-based effective-gradient heuristic (scalar variant).

When only checkpoints ``{theta^[t]}`` are available without optimizer state, the
momentum/Adam coupled recurrences cannot be evaluated. The paper's heuristic
(App. E, "Checkpoint-Based Heuristic") defines an effective gradient

    g_eff^[t] := (theta^[t] - theta^[t+1]) / eta_t

and applies the plain-SGD recurrence with ``g_eff`` in place of the batch-mean
gradient. Placed in WIE's *reverse, per-sample* sweep, the well-defined and
numerically robust content of that heuristic is a per-step **scalar** effective
learning rate that replaces the nominal recorded ``eta_t``:

    eta_eff^[t] = || theta^[t] - theta^[t+1] || / || g_bar^[t] ||,

where ``g_bar^[t]`` is the (recomputed) batch-mean loss gradient at
``theta^[t]``. For a genuine SGD step ``theta^[t+1] = theta^[t] - eta_t g_bar``
this recovers ``eta_t`` exactly; for an Adam/AdamW step it recovers the effective
step *magnitude* the optimizer actually took, so WIE applied to an Adam
trajectory uses the observed step scale rather than the (wrong) nominal base
learning rate.

NOTE (per the paper): this conflates the optimizer's adaptive direction with a
true gradient, so the Theorem 1 error bound no longer holds. When optimizer
state is available, prefer the principled coupled recurrences.
"""

from typing import List

import torch


def effective_gradient(
    theta_t: List[torch.Tensor],
    theta_next: List[torch.Tensor],
    eta_t: float,
) -> List[torch.Tensor]:
    """Paper's effective gradient ``g_eff = (theta^[t] - theta^[t+1]) / eta_t``.

    ``eta_t`` must be non-zero. Returns one tensor per parameter, aligned with
    the inputs.
    """
    if eta_t == 0:
        raise ValueError("eta_t must be non-zero to form the effective gradient.")
    return [(a - b) / eta_t for a, b in zip(theta_t, theta_next)]


def _flat_l2(tensors: List[torch.Tensor]) -> float:
    # Reduce in the tensor's native dtype and accumulate the scalar in Python
    # double. Avoids torch.float64 tensor ops, which are unsupported on MPS
    # (the surrounding WIE code likewise keeps to float32 there).
    total = 0.0
    for x in tensors:
        total += float((x * x).sum().item())
    return total**0.5


def scalar_effective_lr(
    theta_t: List[torch.Tensor],
    theta_next: List[torch.Tensor],
    grad_mean: List[torch.Tensor],
    nominal_lr: float,
    grad_floor: float = 1e-12,
) -> float:
    """Scalar effective learning rate ``||theta^[t]-theta^[t+1]|| / ||g_bar^[t]||``.

    Robust to a near-zero gradient norm: if ``||g_bar|| <= grad_floor`` the step
    scale cannot be recovered and ``nominal_lr`` is returned unchanged. For a true
    SGD step (``theta_next = theta_t - nominal_lr * grad_mean``) the result equals
    ``nominal_lr``.
    """
    gnorm = _flat_l2(grad_mean)
    if gnorm <= grad_floor:
        return float(nominal_lr)
    dtheta = [a - b for a, b in zip(theta_t, theta_next)]
    return _flat_l2(dtheta) / gnorm
