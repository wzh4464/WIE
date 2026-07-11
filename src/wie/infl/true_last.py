from typing import Tuple

from .core import InfluenceCalculatorFactory, BaseDifferenceCalculator


@InfluenceCalculatorFactory.register("true_last")
class TrueLastInfluenceCalculator(BaseDifferenceCalculator):
    """Window-level LOO ground truth for the LAST ``length`` epochs.

    Oracle for :class:`~wie.infl.wie_last.WieLastInfluenceCalculator`: the change
    in the full-LOO validation-loss effect over the epoch window
    ``[num_epochs - length, num_epochs]``, i.e. ``seg[num_epochs-1] -
    seg[num_epochs-length-1]`` (``lo`` clamped to ``-1`` when the window reaches
    back before training).
    """

    def _get_infl_type(self) -> str:
        return "true_last"

    def get_source_prefix(self) -> str:
        return "segment_true"

    def get_window_epoch_bounds(self, num_epoch: int) -> Tuple[int, int]:
        length = int(self.length)
        hi = num_epoch - 1
        lo = max(-1, num_epoch - length - 1)
        return lo, hi
