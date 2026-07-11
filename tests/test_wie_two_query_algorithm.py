"""Correctness test for the two-query/two-term WIE estimator (paper Algorithm 1).

The backward-sweep :meth:`_WieWindowInfluenceCalculator.calculate` must compute,
for every training sample ``j``,

    Q_j = <q(t2), delta_j[t2]> - <q(t1), delta_j[t1]>,

where ``delta_j`` follows the forward recurrence (Eq. 14)

    delta_j[t+1] = P^[t] delta_j[t] + 1_{j in S_t} (eta_t/|S_t|) g(z_j; theta^[t]).

We verify this by FORWARD-unrolling that recurrence with the SAME per-step
primitives the sweep uses (``_safe_update_u`` for ``P^[t]``,
``_compute_param_grads_list`` for ``g``), then comparing to ``calculate()``.
Because ``P`` is now a clean LINEAR propagator (exact HVP, no damping), the
forward accumulation and the backward adjoint sweep are algebraically identical
and must agree to floating-point tolerance.

A tiny in-memory trajectory (double precision) is served through overridden I/O
hooks so no disk checkpoints are needed.
"""

import logging
import unittest

import numpy as np
import torch
import torch.nn as nn

# Stub optional heavy deps (emnist/matplotlib) BEFORE importing wie, so unittest
# discovery does not fail at import time in a lean/pip environment where they are
# absent (wie.infl -> wie.data.modules -> emnist).
from tests.conftest import ensure_dummy_modules

ensure_dummy_modules()

from wie.infl.wie_window_base import _WieWindowInfluenceCalculator


class _MockWie(_WieWindowInfluenceCalculator):
    """Window calculator backed by an in-memory, double-precision trajectory."""

    def __init__(self, traj, q_dict, w_start, w_end, n_tr):
        self._traj = traj  # {t: (model, idx_tensor, lr, x_batch, y_batch)}
        self._q_dict = q_dict  # {step: [param-shaped tensors]}
        self._w = (w_start, w_end)
        self.n_tr = n_tr
        self.total_steps = w_end
        self.length = 1
        self.steps_per_epoch = w_end
        self.alpha = 1e-3
        self.loss_fn = nn.CrossEntropyLoss()
        self.logger = logging.getLogger("mock_wie")
        self.device = torch.device("cpu")

    def _get_infl_type(self):
        return "mock_wie"

    def _window_step_bounds(self):
        return self._w

    def _clamped_window_step_bounds(self):  # skip on-disk clamp
        return self._w

    def _init_query_u(self):
        return [t.clone() for t in self._q_dict[self._w[1]]]

    def _u_at_step(self, step):
        return [t.clone() for t in self._q_dict[step]]

    def _load_step_model_and_batch(self, t):
        return self._traj[t]


def _build_trajectory(seed, n_tr, w_end, dim=3, n_cls=2):
    torch.manual_seed(seed)
    x_tr = torch.randn(n_tr, dim, dtype=torch.float64)
    y_tr = torch.randint(0, n_cls, (n_tr,))
    traj = {}
    for t in range(1, w_end + 1):
        model = nn.Linear(dim, n_cls).double()
        with torch.no_grad():  # perturb so H^[t] varies across steps
            for p in model.parameters():
                p.add_(0.1 * t * torch.randn_like(p))
        bs = 3
        idx = torch.randperm(n_tr)[:bs]
        traj[t] = (model, idx, 0.05 + 0.01 * t, x_tr[idx], y_tr[idx])
    q_dict = {}
    for step in range(0, w_end + 1):
        ref = nn.Linear(dim, n_cls).double()
        q_dict[step] = [
            torch.randn_like(p, dtype=torch.float64) for p in ref.parameters()
        ]
    return traj, q_dict


def _forward_reference(calc, traj, q_dict, w_start, w_end, n_tr, subtract_u1):
    """Directly unroll Eq. 14 forward for each sample, then form the query(s)."""
    dtype = torch.float64
    Q = np.zeros(n_tr, dtype=np.float64)
    q_t2 = q_dict[w_end]
    q_t1 = q_dict[w_start]
    for j in range(n_tr):
        delta = [torch.zeros_like(t) for t in q_t2]
        delta_t1 = None
        for t in range(1, w_end + 1):
            model_t, idx_t, lr_t, xb, yb = traj[t]
            idx_list = idx_t.tolist()
            # delta <- P^[t] delta (identical primitive to the sweep)
            delta = calc._safe_update_u(model_t, xb, yb, delta, lr_t, "ref")
            if j in idx_list:
                grads = calc._compute_param_grads_list(model_t, xb, yb, dtype)
                jl = idx_list.index(j)
                gj = grads[jl]
                bs = len(idx_list)
                delta = [d + (lr_t / bs) * g for d, g in zip(delta, gj)]
            if t == w_start:
                delta_t1 = [d.clone() for d in delta]
        Q[j] = sum(float((a * b).sum()) for a, b in zip(q_t2, delta))
        if subtract_u1 and w_start > 0:
            Q[j] -= sum(float((a * b).sum()) for a, b in zip(q_t1, delta_t1))
    return Q


class TestTwoQueryAlgorithm(unittest.TestCase):
    def _run(self, w_start, w_end, n_tr=5, seed=0):
        traj, q_dict = _build_trajectory(seed, n_tr, w_end)
        calc = _MockWie(traj, q_dict, w_start, w_end, n_tr)
        got = calc.calculate()
        ref = _forward_reference(
            calc, traj, q_dict, w_start, w_end, n_tr, subtract_u1=True
        )
        return got, ref, calc, traj, q_dict

    def _assert_close(self, got, ref):
        num = float(np.linalg.norm(got - ref))
        den = float(np.linalg.norm(ref)) + 1e-12
        self.assertLess(
            num / den,
            1e-8,
            f"backward sweep != forward unroll (rel L2={num / den:.2e})\n"
            f"got={got}\nref={ref}",
        )

    def test_matches_forward_unroll_midwindow(self):
        # t1 > 0: exercises BOTH the u2 (in-window) and u1 (pre-window) terms.
        got, ref, *_ = self._run(w_start=3, w_end=6)
        self._assert_close(got, ref)

    def test_matches_forward_unroll_full_trajectory(self):
        # t1 = 0: second query and pre-window term vanish (wie_first case).
        got, ref, *_ = self._run(w_start=0, w_end=6)
        self._assert_close(got, ref)

    def test_full_trajectory_ignores_second_query(self):
        # With t1 = 0, u1 must never activate: result == single-u result.
        traj, q_dict = _build_trajectory(1, 5, 6)
        calc = _MockWie(traj, q_dict, 0, 6, 5)
        got = calc.calculate()
        single_u = _forward_reference(calc, traj, q_dict, 0, 6, 5, subtract_u1=False)
        self.assertLess(
            float(np.linalg.norm(got - single_u)) / (np.linalg.norm(single_u) + 1e-12),
            1e-8,
        )

    def test_second_query_is_not_a_noop(self):
        # For t1 > 0 the pre-window subtraction must actually change the result,
        # i.e. the faithful estimator differs from the old single-u one.
        got, _, calc, traj, q_dict = self._run(w_start=3, w_end=6)
        single_u = _forward_reference(calc, traj, q_dict, 3, 6, 5, subtract_u1=False)
        self.assertGreater(
            float(np.linalg.norm(got - single_u)),
            1e-6,
            "two-query result equals single-u result: pre-window term inactive",
        )


if __name__ == "__main__":
    unittest.main()
