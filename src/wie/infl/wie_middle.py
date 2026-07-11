import torch
from typing import List, Tuple

from .core import InfluenceCalculatorFactory
from .wie_window_base import _WieWindowInfluenceCalculator


@InfluenceCalculatorFactory.register("wie_middle")
class WieMiddleInfluenceCalculator(_WieWindowInfluenceCalculator):
    """Window-level influence over the CENTERED ``length`` epochs.

    ``start_epoch = max(0, (num_epoch - length) // 2)``; the window is
    ``[start_epoch * spe, min(start_epoch + length, num_epoch) * spe]``. The
    window ends at an intermediate checkpoint (NOT the final model), so the
    query ``u`` is the validation-loss gradient at the window-end model state,
    obtained via :meth:`_u_at_step`. The reverse-SGD sweep and all numeric
    helpers are shared UNCHANGED with ``wie_last`` (see
    :class:`_WieWindowInfluenceCalculator`).

    With ``length == num_epoch`` this degenerates to the full trajectory
    ``[0, total_steps]`` (assuming ``total_steps == num_epoch * spe``).
    """

    def _get_infl_type(self) -> str:
        return "wie_middle"

    def _window_step_bounds(self) -> Tuple[int, int]:
        start_epoch = max(0, (self.num_epoch - self.length) // 2)
        w_start = start_epoch * self.steps_per_epoch
        w_end = min(start_epoch + self.length, self.num_epoch) * self.steps_per_epoch
        return w_start, w_end

    def _init_query_u(self) -> List[torch.Tensor]:
        _, w_end = self._clamped_window_step_bounds()
        return self._u_at_step(w_end)
