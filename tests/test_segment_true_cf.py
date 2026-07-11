"""Tests for segment_true counterfactual-structure parsing (_cf_models_seq).

The trainer saves each sample's counterfactual as a NetList whose ``.models`` is
a ModuleList of per-epoch models. This regression-guards that structure (and the
older/alternate shapes) so segment_true_full -- and hence the true_* window LOO
oracle -- works on real trainer output.
"""

import logging
import os
import tempfile
import unittest

import torch
import torch.nn as nn

# Stub optional heavy deps before importing wie (lean-env discovery safety).
from tests.conftest import ensure_dummy_modules

ensure_dummy_modules()

from wie.infl.segment_true import (  # noqa: E402
    _cf_models_seq,
    SegmentTrueFullInfluenceCalculator,
)


class _NetListLike:
    def __init__(self, models):
        self.models = models


class TestCfModelsSeq(unittest.TestCase):
    def test_netlist_modulelist(self):
        # The actual trainer output: .models is a ModuleList of per-epoch models.
        entry = _NetListLike(nn.ModuleList([nn.Linear(3, 2) for _ in range(4)]))
        seq = _cf_models_seq(entry)
        self.assertEqual(len(seq), 4)
        self.assertTrue(hasattr(seq[0], "state_dict"))

    def test_netlist_plain_list(self):
        entry = _NetListLike([nn.Linear(3, 2) for _ in range(3)])
        self.assertEqual(len(_cf_models_seq(entry)), 3)

    def test_nested_models_models(self):
        # Older nesting: .models is an object whose .models is a list.
        entry = _NetListLike(_NetListLike([nn.Linear(3, 2), nn.Linear(3, 2)]))
        self.assertEqual(len(_cf_models_seq(entry)), 2)

    def test_dict_with_models_list(self):
        entry = {"models": [{"w": torch.zeros(2)} for _ in range(5)]}
        self.assertEqual(len(_cf_models_seq(entry)), 5)

    def test_bare_list(self):
        entry = [nn.Linear(3, 2) for _ in range(6)]
        self.assertEqual(len(_cf_models_seq(entry)), 6)

    def test_unrecognized_returns_none(self):
        self.assertIsNone(_cf_models_seq(42))
        self.assertIsNone(_cf_models_seq(_NetListLike(None)))


class TestPerSampleLoader(unittest.TestCase):
    """The per-sample per-epoch files are complete (incl. the LAST sample),
    unlike the consolidated archive whose final entry can be empty."""

    def _mk(self, dn, seed=0, num_epoch=None, n_tr=None):
        c = SegmentTrueFullInfluenceCalculator.__new__(
            SegmentTrueFullInfluenceCalculator
        )
        c.dn = dn
        c.seed = seed
        c.relabel_percentage = None
        c.logger = logging.getLogger("test_segtrue")
        if num_epoch is not None:
            c.num_epoch = num_epoch
        if n_tr is not None:
            c.n_tr = n_tr
        return c

    def _write(self, rec, i, e, seed=0):
        torch.save(
            {"model_state": nn.Linear(3, 2).state_dict()},
            os.path.join(rec, f"counterfactual_{i:04d}_epoch_{e}_{seed:03d}.pt"),
        )

    def test_builds_complete_list_including_last_sample(self):
        n_tr, n_ep, seed = 4, 3, 0
        with tempfile.TemporaryDirectory() as dn:
            rec = os.path.join(dn, "records")
            os.makedirs(rec)
            sd = nn.Linear(3, 2).state_dict()
            for i in range(n_tr):
                for e in range(n_ep):
                    torch.save(
                        {"model_state": sd, "epoch": e, "counterfactual_sample": i},
                        os.path.join(
                            rec, f"counterfactual_{i:04d}_epoch_{e}_{seed:03d}.pt"
                        ),
                    )
            cf = self._mk(dn, seed)._load_counterfactual_per_sample()
            self.assertEqual(len(cf), n_tr)
            # every sample, including the last, has n_ep per-epoch state-dicts
            for i in range(n_tr):
                self.assertEqual(len(cf[i]), n_ep)
                self.assertIn("weight", cf[i][0])

    def test_returns_none_when_absent(self):
        with tempfile.TemporaryDirectory() as dn:
            os.makedirs(os.path.join(dn, "records"))
            self.assertIsNone(self._mk(dn)._load_counterfactual_per_sample())

    def test_ignores_stale_higher_epochs(self):
        # num_epoch=3 but a reused dir left epoch 3,4 too -> load only 0..2.
        with tempfile.TemporaryDirectory() as dn:
            rec = os.path.join(dn, "records")
            os.makedirs(rec)
            for i in range(2):
                for e in range(5):  # extra stale epochs 3,4
                    self._write(rec, i, e)
            cf = self._mk(dn, num_epoch=3)._load_counterfactual_per_sample()
            self.assertEqual(len(cf), 2)
            self.assertTrue(all(len(s) == 3 for s in cf))

    def test_falls_back_when_missing_epoch(self):
        # num_epoch=4 but only 0..2 present -> incomplete -> None (fall back).
        with tempfile.TemporaryDirectory() as dn:
            rec = os.path.join(dn, "records")
            os.makedirs(rec)
            for i in range(2):
                for e in range(3):
                    self._write(rec, i, e)
            self.assertIsNone(
                self._mk(dn, num_epoch=4)._load_counterfactual_per_sample()
            )

    def test_resolve_cf_epochs(self):
        # exactly num_epoch snapshots -> offset 0
        self.assertEqual(self._mk("/x", num_epoch=10)._resolve_cf_epochs(10), (10, 0))
        # one extra (init) -> offset 1
        self.assertEqual(self._mk("/x", num_epoch=10)._resolve_cf_epochs(11), (10, 1))
        # metadata unavailable -> trust the archive
        self.assertEqual(self._mk("/x")._resolve_cf_epochs(7), (7, 0))
        # archive truncated (fewer than trained) -> raise, don't shorten
        with self.assertRaises(ValueError):
            self._mk("/x", num_epoch=10)._resolve_cf_epochs(5)
        # more than one extra snapshot (stale/longer archive) -> raise
        with self.assertRaises(ValueError):
            self._mk("/x", num_epoch=10)._resolve_cf_epochs(13)

    def test_assert_cf_complete_raises_on_short_entry(self):
        # An incomplete cf list (e.g. empty final NetList) must fail loudly.
        c = self._mk("/x", num_epoch=3)
        c.n_tr = 3
        good = [nn.Linear(3, 2) for _ in range(3)]
        cf = [good, good, []]  # sample 2 empty
        with self.assertRaises(ValueError):
            c._assert_cf_complete(cf, required=3)
        # a complete list passes
        c._assert_cf_complete([good, good, good], required=3)

    def test_falls_back_when_samples_missing(self):
        # Interrupted run: files for samples 0..2 but n_tr=5 -> None (fall back),
        # not a short list that would trip the caller's len<n_tr guard.
        with tempfile.TemporaryDirectory() as dn:
            rec = os.path.join(dn, "records")
            os.makedirs(rec)
            for i in range(3):  # only 3 of 5 expected samples
                for e in range(2):
                    self._write(rec, i, e)
            self.assertIsNone(
                self._mk(dn, num_epoch=2, n_tr=5)._load_counterfactual_per_sample()
            )


if __name__ == "__main__":
    unittest.main()
