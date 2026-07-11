import torch
from typing import List, Tuple

from ..models.networks import get_network
from .core import (
    InfluenceCalculatorFactory,
    load_final_model,
)
from .wie_window_base import _WieWindowInfluenceCalculator


@InfluenceCalculatorFactory.register("wie_last")
class WieLastInfluenceCalculator(_WieWindowInfluenceCalculator):
    """Computes influence by reversing SGD for the last 'length' epochs.

    Window = the LAST ``length`` epochs; the window ends at the FINAL model, so
    the second query ``q(t2)`` is the validation-loss gradient at the final
    model state. All numeric logic lives in
    :class:`_WieWindowInfluenceCalculator`; this class only pins the window
    bounds and ``q(t2)``.

    NOTE: with ``length < num_epoch`` the window start ``t1 > 0``, so the
    faithful two-query estimator (Algorithm 1) also sweeps the PRE-window steps
    ``[0, t1)`` to build the pre-window deviation term and subtract
    ``<q(t1), delta[t1]>``. This changes (and corrects) the scores relative to
    the earlier single-``u`` implementation, and the backward sweep now runs to
    step 0 rather than only over the last ``length`` epochs -- i.e. the same
    O(t2) cost as a full-trajectory sweep, per the paper's complexity analysis.
    Only ``length >= num_epoch`` (full trajectory, ``t1 = 0``) reproduces the
    old output exactly.
    """

    def _get_infl_type(self) -> str:
        return "wie_last"

    def _window_step_bounds(self) -> Tuple[int, int]:
        """Last ``length`` epochs: ``[total_steps - length*spe, total_steps]``.

        Matches the original ``start_step_incl``/``end_step_excl`` exactly:
        the sweep in the base clamps the lower bound with ``max(0, w_start)``.
        """
        w_start = max(0, self.total_steps - self.length * self.steps_per_epoch)
        w_end = self.total_steps
        return w_start, w_end

    def _init_query_u(self) -> List[torch.Tensor]:
        """Query ``u`` at the FINAL model (original ``wie_last`` behavior)."""
        model = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )
        model.load_state_dict(
            load_final_model(self.dn, self.seed, self.device, self.logger)
        )
        model.eval()
        return self._init_u(model)
