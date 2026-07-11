from typing import Tuple

from .core import InfluenceCalculatorFactory, BaseDifferenceCalculator


@InfluenceCalculatorFactory.register("true_middle")
class TrueMiddleInfluenceCalculator(BaseDifferenceCalculator):
    """Window-level LOO ground truth for the CENTERED ``length`` epochs.

    Oracle for :class:`~wie.infl.wie_middle.WieMiddleInfluenceCalculator`: with
    ``start = max(0, (num_epochs - length) // 2)`` the epoch window is
    ``[start, start + length]``, so the difference is ``seg[start+length-1] -
    seg[start-1]`` (``lo = start - 1``, ``= -1`` when ``start = 0``).
    """

    def _get_infl_type(self) -> str:
        return "true_middle"

    def get_source_prefix(self) -> str:
        return "segment_true"

    def get_window_epoch_bounds(self, num_epoch: int) -> Tuple[int, int]:
        length = int(self.length)
        start = max(0, (num_epoch - length) // 2)
        hi = min(start + length, num_epoch) - 1
        lo = start - 1
        return lo, hi
