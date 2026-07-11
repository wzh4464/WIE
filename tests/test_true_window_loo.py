"""Tests for the window-level LOO ground-truth oracles (true_first/middle/last).

These calculators difference the per-epoch full-LOO validation-loss trajectory
(``segment_true_full``) across a window's epoch endpoints:
``Q[i] = seg[hi, i] - seg[lo, i]`` (``seg[-1] = 0``), which is the paper's
Test-Loss window quantity. We test:

1. the pure differencing helper ``infl_diff_helper`` (values + guards),
2. each variant's epoch-window bounds match its ``wie_*`` counterpart,
3. ``_load_segment_full_matrix`` round-trips a ``segment_true_full`` CSV, and
4. a full ``calculate()`` (which only needs dn/seed/relabel/infl_type/length)
   writes the correct window difference CSV.
"""

import logging
import os
import tempfile
import unittest

import numpy as np

# Stub optional heavy deps (emnist/matplotlib) BEFORE importing wie, so unittest
# discovery does not fail at import time in a lean/pip environment where they are
# absent (wie.infl.core -> wie.data.modules -> emnist).
from tests.conftest import ensure_dummy_modules

ensure_dummy_modules()

from wie.infl.core import infl_diff_helper, save_results
from wie.infl.true_first import TrueFirstInfluenceCalculator
from wie.infl.true_middle import TrueMiddleInfluenceCalculator
from wie.infl.true_last import TrueLastInfluenceCalculator


def _mk(cls, length, dn=None, seed=7, relabel=None, num_epoch=None):
    calc = cls.__new__(cls)
    calc.length = length
    calc.seed = seed
    calc.relabel_percentage = relabel
    calc.dn = dn
    calc.infl_type = calc._get_infl_type()
    calc.logger = logging.getLogger("test_true_window")
    if num_epoch is not None:
        calc.num_epoch = num_epoch
    return calc


class TestInflDiffHelper(unittest.TestCase):
    def setUp(self):
        # 3 epochs, 2 samples
        self.seg = np.array([[1.0, 2.0], [3.0, 5.0], [6.0, 9.0]])

    def test_interior_window(self):
        got = infl_diff_helper(self.seg, lo=0, hi=2)
        np.testing.assert_allclose(got, [6.0 - 1.0, 9.0 - 2.0])

    def test_lo_minus_one_is_zero_start(self):
        got = infl_diff_helper(self.seg, lo=-1, hi=2)
        np.testing.assert_allclose(got, [6.0, 9.0])

    def test_single_epoch_window(self):
        got = infl_diff_helper(self.seg, lo=1, hi=2)
        np.testing.assert_allclose(got, [3.0, 4.0])

    def test_empty_window_raises(self):
        with self.assertRaises(ValueError):
            infl_diff_helper(self.seg, lo=2, hi=2)
        with self.assertRaises(ValueError):
            infl_diff_helper(self.seg, lo=2, hi=1)

    def test_hi_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            infl_diff_helper(self.seg, lo=0, hi=3)


class TestWindowEpochBounds(unittest.TestCase):
    def test_first(self):
        self.assertEqual(
            _mk(TrueFirstInfluenceCalculator, 2).get_window_epoch_bounds(6), (-1, 1)
        )
        # length beyond the trajectory clamps hi to the last epoch
        self.assertEqual(
            _mk(TrueFirstInfluenceCalculator, 10).get_window_epoch_bounds(6), (-1, 5)
        )

    def test_last(self):
        self.assertEqual(
            _mk(TrueLastInfluenceCalculator, 2).get_window_epoch_bounds(6), (3, 5)
        )
        # length covering the whole run -> lo clamps to -1 (start before training)
        self.assertEqual(
            _mk(TrueLastInfluenceCalculator, 10).get_window_epoch_bounds(6), (-1, 5)
        )

    def test_middle(self):
        # start = max(0, (6-2)//2) = 2 -> window epochs [2,4) -> lo=1, hi=3
        self.assertEqual(
            _mk(TrueMiddleInfluenceCalculator, 2).get_window_epoch_bounds(6), (1, 3)
        )
        # odd count: start = (5-2)//2 = 1 -> lo=0, hi=2
        self.assertEqual(
            _mk(TrueMiddleInfluenceCalculator, 2).get_window_epoch_bounds(5), (0, 2)
        )

    def test_bounds_match_wie_window_step_bounds(self):
        # true_* epoch bounds must localize the SAME first/mid/last region as the
        # wie_* step windows divided by steps_per_epoch.
        num_epoch, length, spe = 6, 2, 4
        total = num_epoch * spe
        # wie_first: [0, length*spe] -> epochs [0, length]; true hi = length-1, lo=-1
        self.assertEqual(
            _mk(TrueFirstInfluenceCalculator, length).get_window_epoch_bounds(
                num_epoch
            ),
            (-1, length - 1),
        )
        # wie_last: [total-length*spe, total] -> epochs [num_epoch-length, num_epoch]
        lo, hi = _mk(TrueLastInfluenceCalculator, length).get_window_epoch_bounds(
            num_epoch
        )
        self.assertEqual((lo, hi), (num_epoch - length - 1, num_epoch - 1))
        self.assertEqual(
            (lo + 1, hi + 1), ((total - length * spe) // spe, total // spe)
        )


class TestLoadAndCalculate(unittest.TestCase):
    def _write_segment_full(self, dn, seed, seg_list, relabel=None):
        # save_results writes infl_segment_true_full_{relabel}{seed:03d}.csv
        save_results(
            seg_list, dn, seed, "segment_true_full", logging.getLogger("w"), relabel
        )

    def test_load_segment_full_matrix_roundtrip(self):
        with tempfile.TemporaryDirectory() as dn:
            seg_list = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
            self._write_segment_full(dn, 7, seg_list)
            calc = _mk(TrueFirstInfluenceCalculator, 1, dn=dn, seed=7)
            seg = calc._load_segment_full_matrix("segment_true")
            self.assertEqual(seg.shape, (2, 3))
            np.testing.assert_allclose(seg, np.stack(seg_list))

    def test_calculate_writes_window_difference(self):
        with tempfile.TemporaryDirectory() as dn:
            # 4 epochs, 3 samples; cumulative full-LOO effect per epoch
            seg_list = [
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, -1.0, 0.5]),
                np.array([2.0, -2.0, 1.0]),
                np.array([3.0, -3.0, 1.5]),
            ]
            self._write_segment_full(dn, 7, seg_list)

            # true_last, length=1 -> lo=2, hi=3 -> seg[3]-seg[2]
            calc = _mk(TrueLastInfluenceCalculator, 1, dn=dn, seed=7)
            infl = calc.calculate()
            np.testing.assert_allclose(infl, [1.0, -1.0, 0.5])

            out_csv = os.path.join(dn, "infl_true_last_007.csv")
            self.assertTrue(os.path.isfile(out_csv), "true_last did not save its CSV")
            import pandas as pd

            saved = pd.read_csv(out_csv)
            col = [c for c in saved.columns if "influence" in c.lower()][0]
            np.testing.assert_allclose(saved[col].to_numpy(), [1.0, -1.0, 0.5])

    def test_calculate_true_first_from_zero(self):
        with tempfile.TemporaryDirectory() as dn:
            seg_list = [
                np.array([1.0, 2.0]),
                np.array([3.0, 5.0]),
                np.array([6.0, 9.0]),
            ]
            self._write_segment_full(dn, 7, seg_list)
            # true_first, length=2 -> lo=-1, hi=1 -> seg[1]-0
            calc = _mk(TrueFirstInfluenceCalculator, 2, dn=dn, seed=7)
            infl = calc.calculate()
            np.testing.assert_allclose(infl, [3.0, 5.0])

    def test_calculate_rejects_nonpositive_length(self):
        # Mirrors the wie_* guard: length < 1 must raise, not clamp to 1 epoch.
        with tempfile.TemporaryDirectory() as dn:
            seg_list = [np.array([1.0, 2.0]), np.array([3.0, 5.0])]
            self._write_segment_full(dn, 7, seg_list)
            for bad in (0, -1):
                calc = _mk(TrueLastInfluenceCalculator, bad, dn=dn, seed=7)
                with self.assertRaises(ValueError):
                    calc.calculate()

    def test_calculate_rejects_epoch_count_mismatch(self):
        # If training metadata (num_epoch) disagrees with the number of source
        # rows, the window cannot be aligned to the trained trajectory -> raise.
        with tempfile.TemporaryDirectory() as dn:
            seg_list = [
                np.array([1.0, 2.0]),
                np.array([3.0, 5.0]),
                np.array([6.0, 9.0]),
            ]  # 3 rows
            self._write_segment_full(dn, 7, seg_list)
            calc = _mk(TrueLastInfluenceCalculator, 1, dn=dn, seed=7, num_epoch=5)
            with self.assertRaises(ValueError):
                calc.calculate()

    def test_calculate_uses_metadata_epoch_count_when_aligned(self):
        # When num_epoch matches the row count, it is used for the bounds.
        with tempfile.TemporaryDirectory() as dn:
            seg_list = [
                np.array([0.0, 0.0]),
                np.array([1.0, -1.0]),
                np.array([2.0, -2.0]),
                np.array([3.0, -3.0]),
            ]
            self._write_segment_full(dn, 7, seg_list)
            calc = _mk(TrueLastInfluenceCalculator, 1, dn=dn, seed=7, num_epoch=4)
            infl = calc.calculate()  # lo=2, hi=3 -> seg[3]-seg[2]
            np.testing.assert_allclose(infl, [1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
