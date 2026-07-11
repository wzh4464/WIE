"""Tests for the Appendix-E checkpoint effective-learning-rate heuristic.

Covers the pure helpers (:func:`effective_gradient`, :func:`scalar_effective_lr`)
and the ``_step_lr`` wiring in the WIE window base: it must (a) return the
nominal ``eta_t`` unchanged when the heuristic is off, (b) reduce to ``eta_t``
exactly on a genuine SGD step, (c) recover a different scale on a non-SGD step,
and (d) fall back to nominal when the next checkpoint is unavailable.
"""

import logging
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

# Stub optional heavy deps before importing wie (lean-env discovery safety).
from tests.conftest import ensure_dummy_modules

ensure_dummy_modules()

from wie.infl.effective_lr import (  # noqa: E402
    effective_gradient,
    scalar_effective_lr,
)
from wie.infl.wie_last import WieLastInfluenceCalculator  # noqa: E402


def _mk_calc(use_eff=True, alpha=0.0):
    calc = WieLastInfluenceCalculator.__new__(WieLastInfluenceCalculator)
    calc.use_effective_lr = use_eff
    calc.alpha = alpha
    calc.loss_fn = nn.CrossEntropyLoss()
    calc.logger = logging.getLogger("test_eff_lr")
    calc.device = torch.device("cpu")
    calc.seed = 7
    calc.relabel_percentage = None
    return calc


class TestPureHelpers(unittest.TestCase):
    def test_effective_gradient(self):
        a = [torch.tensor([2.0, 4.0])]
        b = [torch.tensor([1.0, 1.0])]
        g = effective_gradient(a, b, eta_t=0.5)
        torch.testing.assert_close(g[0], torch.tensor([2.0, 6.0]))  # (a-b)/0.5

    def test_effective_gradient_rejects_zero_eta(self):
        with self.assertRaises(ValueError):
            effective_gradient([torch.zeros(2)], [torch.zeros(2)], 0.0)

    def test_scalar_lr_reduces_to_nominal_on_sgd_step(self):
        g = [torch.tensor([3.0, 4.0]), torch.tensor([[1.0]])]
        theta_t = [torch.tensor([10.0, 10.0]), torch.tensor([[2.0]])]
        eta = 0.25
        theta_next = [t - eta * gi for t, gi in zip(theta_t, g)]
        eff = scalar_effective_lr(theta_t, theta_next, g, nominal_lr=eta)
        self.assertAlmostEqual(eff, eta, places=6)

    def test_scalar_lr_recovers_step_scale(self):
        # ||dtheta|| = 5 (3,4), ||g|| = 1 -> eff = 5
        g = [torch.tensor([1.0, 0.0])]
        theta_t = [torch.tensor([3.0, 4.0])]
        theta_next = [torch.tensor([0.0, 0.0])]
        eff = scalar_effective_lr(theta_t, theta_next, g, nominal_lr=0.1)
        self.assertAlmostEqual(eff, 5.0, places=6)

    def test_scalar_lr_zero_gradient_falls_back(self):
        g = [torch.zeros(3)]
        theta_t = [torch.tensor([1.0, 2.0, 3.0])]
        theta_next = [torch.zeros(3)]
        eff = scalar_effective_lr(theta_t, theta_next, g, nominal_lr=0.07)
        self.assertEqual(eff, 0.07)


class TestStepLrWiring(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = nn.Linear(5, 2)
        self.x = torch.randn(6, 5)
        self.y = torch.randint(0, 2, (6,))

    def _sgd_step_models(self, calc, eta):
        """A prev model theta[t-1] and current model theta[t]=theta[t-1]-eta*g(prev)."""
        prev = nn.Linear(5, 2)
        g_prev = calc._batch_mean_grad(prev, self.x, self.y)  # g at theta[t-1]
        cur = nn.Linear(5, 2)
        with torch.no_grad():
            for pc, pp, g in zip(cur.parameters(), prev.parameters(), g_prev):
                pc.copy_(pp - eta * g)
        return prev, cur

    def test_off_returns_nominal(self):
        calc = _mk_calc(use_eff=False)
        # Even if a prev checkpoint existed, the flag being off returns nominal.
        calc._load_prev_step_model = lambda t: nn.Linear(5, 2)
        self.assertEqual(calc._step_lr(3, self.model, self.x, self.y, 0.1), 0.1)

    def test_on_reduces_to_nominal_on_sgd_step(self):
        # g_bar is recomputed at the PRE-update state theta[t-1], so a genuine
        # SGD step recovers eta EXACTLY (nonlinear-loss safe).
        calc = _mk_calc(use_eff=True, alpha=1e-3)
        calc._has_step_checkpoint = lambda step: True
        eta = 0.1
        prev, cur = self._sgd_step_models(calc, eta)
        calc._load_prev_step_model = lambda t: prev
        eff = calc._step_lr(3, cur, self.x, self.y, nominal_lr=eta)
        self.assertAlmostEqual(eff, eta, places=5)

    def test_on_recovers_different_scale_on_non_sgd_step(self):
        calc = _mk_calc(use_eff=True, alpha=0.0)
        calc._has_step_checkpoint = lambda step: True
        eta = 0.1
        prev = nn.Linear(5, 2)
        g_prev = calc._batch_mean_grad(prev, self.x, self.y)
        cur = nn.Linear(5, 2)
        with torch.no_grad():  # step 3x the SGD step in a shifted direction
            for pc, pp, g in zip(cur.parameters(), prev.parameters(), g_prev):
                pc.copy_(pp - (3.0 * eta) * g - 0.05)
        calc._load_prev_step_model = lambda t: prev
        eff = calc._step_lr(3, cur, self.x, self.y, nominal_lr=eta)
        self.assertGreater(eff, eta)  # larger observed step -> larger effective lr

    def test_on_reduces_to_nominal_for_batchnorm_model(self):
        # BatchNorm gradients differ between train and eval mode; the update ran
        # in TRAIN mode, so the denominator must too. This would fail if the
        # gradient were recomputed in eval mode.
        calc = _mk_calc(use_eff=True, alpha=0.0)
        calc._has_step_checkpoint = lambda step: True
        eta = 0.05
        prev = nn.Sequential(nn.Linear(5, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
        prev.train()
        g_prev = calc._batch_mean_grad(prev, self.x, self.y)  # train-mode grad
        cur = nn.Sequential(nn.Linear(5, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
        with torch.no_grad():
            for pc, pp, g in zip(cur.parameters(), prev.parameters(), g_prev):
                pc.copy_(pp - eta * g)
        calc._load_prev_step_model = lambda t: prev
        eff = calc._step_lr(3, cur, self.x, self.y, nominal_lr=eta)
        self.assertAlmostEqual(eff, eta, places=4)

    def test_on_falls_back_when_no_prev_checkpoint(self):
        calc = _mk_calc(use_eff=True)
        calc._has_step_checkpoint = lambda step: True
        calc._load_prev_step_model = lambda t: None
        self.assertEqual(calc._step_lr(9, self.model, self.x, self.y, 0.1), 0.1)

    def test_on_falls_back_without_genuine_step_checkpoints(self):
        # Epoch-only runs (no seeded step files): the delta is not one SGD step,
        # so effective-lr must fall back to nominal even though a prev state loads,
        # AND it must warn loudly (once) rather than silently reproduce defaults.
        calc = _mk_calc(use_eff=True)
        calc._has_step_checkpoint = lambda step: False
        calc._load_prev_step_model = lambda t: nn.Linear(5, 2)
        with self.assertLogs(calc.logger, level="WARNING") as cm:
            self.assertEqual(calc._step_lr(3, self.model, self.x, self.y, 0.1), 0.1)
        self.assertTrue(
            any("effective-lr heuristic is INACTIVE" in m for m in cm.output)
        )
        # Warns only once across steps.
        self.assertEqual(calc._step_lr(4, self.model, self.x, self.y, 0.1), 0.1)
        self.assertTrue(getattr(calc, "_eff_lr_warned", False))


class TestInitialCheckpoint(unittest.TestCase):
    def test_has_step_checkpoint_step0_uses_init_file(self):
        calc = _mk_calc(use_eff=True)
        with tempfile.TemporaryDirectory() as d:
            calc.dn = d
            os.makedirs(os.path.join(d, "records"))
            self.assertFalse(calc._has_step_checkpoint(0))
            open(os.path.join(d, "records", "init_007.pt"), "wb").close()
            self.assertTrue(calc._has_step_checkpoint(0))

    def test_load_prev_step_model_uses_initial_for_first_step(self):
        # For t == 1, theta[t-1] == theta[0] must come from the initial checkpoint
        # (init_{seed}.pt), NOT load_step_data(0).
        calc = _mk_calc(use_eff=True)
        calc.dn = "/nonexistent"
        calc.model_type = "logreg"
        calc.input_dim = 5
        with (
            patch(
                "wie.infl.wie_window_base.load_initial_model",
                return_value=nn.Linear(5, 2).state_dict(),
            ) as m_init,
            patch("wie.infl.wie_window_base.get_network", return_value=nn.Linear(5, 2)),
            patch("wie.infl.wie_window_base.load_step_data") as m_step,
        ):
            model = calc._load_prev_step_model(1)
        m_init.assert_called_once()
        m_step.assert_not_called()
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
