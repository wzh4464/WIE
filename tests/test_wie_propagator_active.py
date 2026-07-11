"""Regression guard: the WIE window variants' influence propagator must stay ACTIVE.

The propagator ``P=I-eta*H`` is applied by :meth:`_safe_update_u`, which calls
the exact Pearlmutter HVP :meth:`_hvp`. A degenerate HVP (or a swallowed
exception) would leave ``u`` unchanged every step, collapsing ``P`` to identity
(TracIn-like). This test pins that the HVP is non-zero and that a u-update
actually changes ``u``, for the shared base used by wie_first/wie_middle/wie_last.

(Historically the propagator was a central finite-difference approximation whose
``loss.backward()`` ran inside ``torch.no_grad()`` and silently raised; it is now
an exact double-backprop HVP with no damping.)
"""

import logging
import unittest

import torch
import torch.nn as nn

# Stub optional heavy deps (emnist/matplotlib) BEFORE importing wie, so unittest
# discovery does not fail at import time in a lean/pip environment where they are
# absent (wie.infl -> wie.data.modules -> emnist).
from tests.conftest import ensure_dummy_modules

ensure_dummy_modules()

from wie.infl.wie_last import WieLastInfluenceCalculator


def _make_calc():
    calc = WieLastInfluenceCalculator.__new__(WieLastInfluenceCalculator)
    calc.loss_fn = nn.CrossEntropyLoss()
    calc.alpha = 0.001
    calc.logger = logging.getLogger("test_wie_propagator")
    return calc


class TestPropagatorActive(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.calc = _make_calc()
        self.model = nn.Linear(5, 2)
        self.x = torch.randn(4, 5)
        self.y = torch.randint(0, 2, (4,))
        self.v = [torch.randn_like(p).double() for p in self.model.parameters()]

    def test_hvp_is_nonzero(self):
        hv = self.calc._hvp(self.model, self.x, self.y, self.v)
        l2 = sum(float((h * h).sum()) for h in hv)
        self.assertGreater(
            l2,
            1e-9,
            "HVP is ~0: propagator is a no-op (autograd graph not built?)",
        )

    def test_hvp_matches_finite_difference(self):
        # The exact Pearlmutter HVP must agree with a central finite difference
        # of the (regularized) mean-batch-loss gradient, to a loose tolerance.
        eps = 1e-3
        params = [p for p in self.model.parameters()]
        theta = [p.detach().clone() for p in params]

        def grad_list():
            self.model.zero_grad()
            loss = self.calc.loss_fn(self.model(self.x), self.y)
            for p in self.model.parameters():
                loss = loss + 0.5 * self.calc.alpha * (p * p).sum()
            loss.backward()
            gl = [p.grad.detach().clone() for p in self.model.parameters()]
            self.model.zero_grad()
            return gl

        with torch.no_grad():
            for p, th, vp in zip(params, theta, self.v):
                p.copy_(th + eps * vp)
        gp = grad_list()
        with torch.no_grad():
            for p, th, vp in zip(params, theta, self.v):
                p.copy_(th - eps * vp)
        gn = grad_list()
        with torch.no_grad():
            for p, th in zip(params, theta):
                p.copy_(th)
        fd = [(a - b) / (2 * eps) for a, b in zip(gp, gn)]

        hv = self.calc._hvp(self.model, self.x, self.y, self.v)
        num = sum(float(((a - b) ** 2).sum()) for a, b in zip(hv, fd))
        den = sum(float((b * b).sum()) for b in fd) + 1e-12
        self.assertLess(
            (num / den) ** 0.5,
            1e-2,
            "exact HVP disagrees with finite-difference HVP",
        )

    def test_safe_update_u_changes_u(self):
        u2 = self.calc._safe_update_u(self.model, self.x, self.y, self.v, 0.1, "t")
        drift = sum(float(((a - b) ** 2).sum()) for a, b in zip(u2, self.v))
        self.assertGreater(
            drift,
            1e-9,
            "u did not change: HVP update was silently skipped (no-op propagator)",
        )


if __name__ == "__main__":
    unittest.main()
