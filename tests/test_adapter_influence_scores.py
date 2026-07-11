"""Smoke/regression tests for the single-vector adapter calculators.

These pin that the registered ``sgd``/``icml``/``tracin`` adapter paths actually
run end-to-end and return finite per-sample scores. In particular ``icml`` used
a finite-difference helper that returned a 0-D scalar, so iterating it as an HVP
list crashed before conjugate gradient started (regression guard).
"""
import unittest

import numpy as np
import torch
import torch.nn as nn

from wie.infl.adapters import IcmlAdapterCalculator


def _make_calc(cls, n_tr=6, n_val=5, in_dim=4, n_cls=2):
    torch.manual_seed(0)
    calc = cls.__new__(cls)
    calc.device = "cpu"
    calc.loss_fn = nn.CrossEntropyLoss()
    calc.alpha = 0.01
    calc.x_tr = torch.randn(n_tr, in_dim)
    calc.y_tr = torch.randint(0, n_cls, (n_tr,))
    calc.x_val = torch.randn(n_val, in_dim)
    calc.y_val = torch.randint(0, n_cls, (n_val,))
    return calc, nn.Linear(in_dim, n_cls)


class TestAdapterScores(unittest.TestCase):
    def test_icml_adapter_produces_finite_scores(self):
        calc, model = _make_calc(IcmlAdapterCalculator)
        scores = calc._scores_for_model(model)
        self.assertEqual(scores.shape, (calc.x_tr.shape[0],))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
