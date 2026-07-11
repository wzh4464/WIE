from typing import Tuple

from .core import InfluenceCalculatorFactory, BaseDifferenceCalculator


@InfluenceCalculatorFactory.register("true_first")
class TrueFirstInfluenceCalculator(BaseDifferenceCalculator):
    """Window-level LOO ground truth for the FIRST ``length`` epochs.

    Oracle for :class:`~wie.infl.wie_first.WieFirstInfluenceCalculator`: the
    change in the full-LOO validation-loss effect over the epoch window
    ``[0, length]``. Since the window starts before any training (zero
    deviation), this is ``seg[length-1] - 0`` -- i.e. ``lo = -1``, ``hi =
    min(length, num_epochs) - 1``.
    """

    def _get_infl_type(self) -> str:
        return "true_first"

    def get_source_prefix(self) -> str:
        return "segment_true"

    def get_window_epoch_bounds(self, num_epoch: int) -> Tuple[int, int]:
        length = int(self.length)
        hi = min(length, num_epoch) - 1
        lo = -1
        return lo, hi
