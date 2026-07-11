"""Tests for the extended query toolkit (prediction-change & feature-saliency).

Validates the query builders against manual autograd references and checks the
``_query`` routing hook in the WIE window base (loss/prediction/saliency), plus
alignment with ``model.parameters()`` and frozen-parameter handling.
"""

import logging
import unittest

import torch
import torch.nn as nn

# Stub optional heavy deps before importing wie (lean-env discovery safety).
from tests.conftest import ensure_dummy_modules

ensure_dummy_modules()

from wie.infl.core import compute_gradient  # noqa: E402
from wie.infl.queries import prediction_query, saliency_query  # noqa: E402
from wie.infl.wie_last import WieLastInfluenceCalculator  # noqa: E402


def _params(model):
    return list(model.parameters())


class TestPredictionQuery(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = nn.Linear(5, 3)
        self.x = torch.randn(1, 5)

    def test_matches_manual_grad_of_logit(self):
        q = prediction_query(self.x, self.model, class_index=1)
        out = self.model(self.x)
        ref = torch.autograd.grad(out[0, 1], _params(self.model))
        self.assertEqual(len(q), len(ref))
        for a, b in zip(q, ref):
            torch.testing.assert_close(a, b)

    def test_default_class_is_argmax(self):
        c = int(self.model(self.x).reshape(1, -1).argmax(dim=1).item())
        q_default = prediction_query(self.x, self.model, None)
        q_c = prediction_query(self.x, self.model, c)
        for a, b in zip(q_default, q_c):
            torch.testing.assert_close(a, b)

    def test_out_of_range_class_raises(self):
        with self.assertRaises(ValueError):
            prediction_query(self.x, self.model, class_index=9)

    def test_binary_head(self):
        binm = nn.Linear(5, 1)  # single-logit head
        q = prediction_query(self.x, binm, class_index=0)
        ref = torch.autograd.grad(binm(self.x)[0, 0], _params(binm))
        for a, b in zip(q, ref):
            torch.testing.assert_close(a, b)

    def test_frozen_params_zero_and_aligned(self):
        self.model.bias.requires_grad_(False)
        q = prediction_query(self.x, self.model, 0)
        self.assertEqual(len(q), len(_params(self.model)))  # weight, bias
        self.assertEqual(int(torch.count_nonzero(q[1])), 0)  # bias -> zeros
        self.assertGreater(float((q[0] * q[0]).sum()), 0)  # weight nonzero


class TestSaliencyQuery(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = nn.Linear(5, 3)
        self.x = torch.randn(1, 5)
        self.y = torch.tensor([1])
        self.loss_fn = nn.CrossEntropyLoss()

    def test_matches_manual_double_autograd(self):
        q = saliency_query(self.x, self.y, self.model, self.loss_fn, coord=2)
        xg = self.x.detach().clone().requires_grad_(True)
        loss = self.loss_fn(self.model(xg), self.y)
        gx = torch.autograd.grad(loss, xg, create_graph=True)[0]
        ref = torch.autograd.grad(gx.reshape(-1)[2], _params(self.model))
        self.assertEqual(len(q), len(ref))
        for a, b in zip(q, ref):
            torch.testing.assert_close(a, b)

    def test_nonzero_and_differs_from_loss_query(self):
        q = saliency_query(self.x, self.y, self.model, self.loss_fn, coord=0)
        self.assertGreater(sum(float((g * g).sum()) for g in q), 0)
        q_loss = compute_gradient(self.x, self.y, self.model, self.loss_fn)
        same = all(torch.allclose(a, b) for a, b in zip(q, q_loss))
        self.assertFalse(same, "saliency query must differ from the loss query")

    def test_out_of_range_coord_raises(self):
        with self.assertRaises(ValueError):
            saliency_query(self.x, self.y, self.model, self.loss_fn, coord=99)

    def test_non_float_input_raises(self):
        # Token-id models (long inputs) have no differentiable path to x.
        xi = torch.randint(0, 10, (1, 5))
        with self.assertRaises(ValueError):
            saliency_query(xi, self.y, self.model, self.loss_fn, coord=0)

    def test_float_input_disconnected_in_forward_raises(self):
        # Float input that the model casts back to long (like BERT) -> the loss
        # is disconnected from x; must raise the clear error, not a RuntimeError.
        class _TokenLike(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(5, 3)

            def forward(self, x):
                return self.lin(x.long().float())  # round-trip disconnects grad

        with self.assertRaises(ValueError):
            saliency_query(self.x, self.y, _TokenLike(), self.loss_fn, coord=0)


def _mk_query_calc(**kw):
    calc = WieLastInfluenceCalculator.__new__(WieLastInfluenceCalculator)
    calc.loss_fn = nn.CrossEntropyLoss()
    calc.logger = logging.getLogger("test_queries")
    calc.query_type = kw.get("query_type", "loss")
    calc.query_index = kw.get("query_index", 0)
    calc.query_class = kw.get("query_class", None)
    calc.query_coord = kw.get("query_coord", 0)
    torch.manual_seed(2)
    calc.x_val = torch.randn(4, 5)
    calc.y_val = torch.randint(0, 3, (4,))
    return calc


class TestQueryRouting(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(5, 3)

    def test_loss_default_matches_compute_gradient(self):
        calc = _mk_query_calc(query_type="loss")
        q = calc._query(self.model)
        ref = compute_gradient(calc.x_val, calc.y_val, self.model, calc.loss_fn)
        for a, b in zip(q, ref):
            torch.testing.assert_close(a, b)

    def test_prediction_routing(self):
        calc = _mk_query_calc(query_type="prediction", query_class=2, query_index=1)
        q = calc._query(self.model)
        ref = prediction_query(calc.x_val[1:2], self.model, 2)
        for a, b in zip(q, ref):
            torch.testing.assert_close(a, b)

    def test_saliency_routing(self):
        calc = _mk_query_calc(query_type="saliency", query_coord=3, query_index=2)
        q = calc._query(self.model)
        ref = saliency_query(
            calc.x_val[2:3], calc.y_val[2:3], self.model, calc.loss_fn, 3
        )
        for a, b in zip(q, ref):
            torch.testing.assert_close(a, b)

    def test_query_index_out_of_range_raises(self):
        # Out-of-range index is rejected (not clamped), so the saved filename's
        # ..._i{idx}... always identifies the actual example used.
        calc = _mk_query_calc(query_type="prediction", query_index=999)
        with self.assertRaises(ValueError):
            calc._query_test_input()

    def test_query_index_in_range_ok(self):
        calc = _mk_query_calc(query_type="prediction", query_index=2)
        x = calc._query_test_input()
        torch.testing.assert_close(x, calc.x_val[2:3])

    def test_unknown_query_type_raises(self):
        calc = _mk_query_calc(query_type="bogus")
        with self.assertRaises(ValueError):
            calc._query(self.model)


class TestOutputIdentity(unittest.TestCase):
    def _calc(self, **kw):
        calc = WieLastInfluenceCalculator.__new__(WieLastInfluenceCalculator)
        calc.infl_type = "wie_last"
        calc.query_type = kw.get("query_type", "loss")
        calc.query_index = kw.get("query_index", 0)
        calc.query_class = kw.get("query_class", None)
        calc.query_coord = kw.get("query_coord", 0)
        return calc

    def test_loss_output_type_unchanged(self):
        self.assertEqual(self._calc(query_type="loss")._output_infl_type(), "wie_last")

    def test_prediction_output_type_is_distinct(self):
        t = self._calc(
            query_type="prediction", query_index=7, query_class=3
        )._output_infl_type()
        self.assertNotEqual(t, "wie_last")
        for token in ("wie_last", "query-prediction", "i7", "c3"):
            self.assertIn(token, t)

    def test_prediction_argmax_marker(self):
        t = self._calc(query_type="prediction", query_class=None)._output_infl_type()
        self.assertIn("cargmax", t)

    def test_saliency_output_type_is_distinct(self):
        t = self._calc(query_type="saliency", query_coord=4)._output_infl_type()
        self.assertNotEqual(t, "wie_last")
        for token in ("query-saliency", "k4"):
            self.assertIn(token, t)


if __name__ == "__main__":
    unittest.main()
