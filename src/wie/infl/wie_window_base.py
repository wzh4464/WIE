import os
import re
import numpy as np
import torch
from typing import List, Optional, Tuple

from ..models.networks import get_network
from ..io.naming import make_relabel_prefix
from .core import (
    InfluenceCalculator,
    compute_gradient,
    load_step_data,
    load_initial_model,
)
from .effective_lr import scalar_effective_lr
from .queries import prediction_query, saliency_query


class _WieWindowInfluenceCalculator(InfluenceCalculator):
    """Shared reverse-SGD, two-query window-level influence (paper Algorithm 1).

    Measures each training sample's window-level influence
    ``Q_{-j}^{[t1,t2]}(q) = <q(t2), delta[t2]> - <q(t1), delta[t1]>`` by a single
    backward sweep with two adjoint vectors ``u2`` (carries ``q(t2)``) and
    ``u1`` (carries ``q(t1)``), propagated by the influence propagator
    ``P^[t]=I-eta_t H^[t]`` via exact Pearlmutter HVPs (:meth:`_hvp`). See
    :meth:`calculate` for the full algorithm.

    Subclasses parametrize ONLY the window and the query-init point via two
    hooks:

    - :meth:`_window_step_bounds` -> ``(t1, t2)`` (``= (w_start, w_end)``) in SGD
      steps.
    - :meth:`_init_query_u` -> the query ``q(t2)`` (a list of tensors aligned
      with model parameters), evaluated at the window-end model state. The
      window-start query ``q(t1)`` is obtained via :meth:`_u_at_step`.

    For ``t1 = 0`` (``wie_first``, or any full-trajectory window) the second
    query and the pre-window term vanish, so the estimate reduces exactly to the
    earlier single-``u`` result; for ``t1 > 0`` (``wie_last``/``wie_middle``) the
    faithful two-query/two-term estimator adds the pre-window deviation the
    single-``u`` version dropped.
    """

    # -----------------------------
    # Hooks (subclasses override)
    # -----------------------------
    def _window_step_bounds(self) -> Tuple[int, int]:
        """Return ``(w_start, w_end)`` epoch-window bounds in SGD steps."""
        raise NotImplementedError

    def _last_recorded_step(self) -> Optional[int]:
        """Largest SGD step for which a usable checkpoint exists on disk.

        Inspects ``{self.dn}/records`` with a single ``listdir``. Prefers seeded
        step files (``...step_{N}_{seed:03d}.pt``, the format ``load_step_data``
        reads); if none exist, falls back to epoch files
        (``...epoch_{E}_{seed:03d}.pt``), whose end-of-epoch state serves step
        ``(E + 1) * steps_per_epoch`` via the loader's epoch fallback. Returns
        ``None`` when nothing is found (or ``dn``/``seed`` are unset), so full
        runs are left unclamped.

        Scoped to a SINGLE relabel namespace, mirroring ``resolve_step_file``/
        ``make_relabel_prefix``: when multiple relabel trajectories for one seed
        share the records dir, we pick ONE prefix for the WHOLE endpoint -- the
        first (in loader order ``[relabel_prefix, ""]``) that has ANY step OR
        epoch record -- and combine the step-max and epoch-endpoint WITHIN that
        same prefix. We never combine endpoints across prefixes, so a stale/
        longer/other-prefix trajectory can't set the endpoint for this run.

        ``load_step_data`` can serve a step from EITHER a seeded step file or the
        end-of-epoch fallback, so within the chosen namespace the true endpoint
        is the furthest of both formats -- a run may hold steps 1..N plus
        complete epoch checkpoints that serve steps > N; taking the step max
        alone would truncate wie_first/wie_middle (or collapse wie_last).

        Bare, unseeded ``step_NNNNNN.pt`` dumps (written under ``--steps_only``)
        are intentionally NOT matched: ``load_step_data`` cannot consume them.

        CAVEAT (documented, not fixed here): ``resolve_step_file`` tries an
        UNprefixed step file before the prefixed-epoch fallback, so if a stale
        unprefixed *seeded* step file shadows a step this window sweeps, the
        loader could still return it at load time. That is pre-existing shared
        behavior (``wie_last``/``wie_all_epochs`` also route through
        ``resolve_step_file``); rewriting it risks changing accepted results, and
        in practice the trainer never writes seeded step files (only unseeded
        ``step_NNNNNN.pt``), so the shadow can't arise from a normal run. This
        endpoint-discovery method is namespace-correct regardless.
        """
        dn = getattr(self, "dn", None)
        seed = getattr(self, "seed", None)
        if not dn or seed is None:
            return None
        records_dir = os.path.join(dn, "records")
        if not os.path.isdir(records_dir):
            return None
        try:
            names = os.listdir(records_dir)
        except OSError:
            return None

        seed_suffix = re.escape(f"_{int(seed):03d}.pt")
        # Prefix candidates in loader-preference order: the current relabel
        # prefix first, then unprefixed (exactly resolve_step_file's fallback).
        prefix = make_relabel_prefix(getattr(self, "relabel_percentage", None))
        prefixes = [prefix, ""] if prefix else [""]

        def _max_for(prefix_str: str, kind: str) -> Optional[int]:
            # Anchor at start so a longer relabel-prefixed name never leaks into
            # an unprefixed (or different-prefix) selection.
            pat = re.compile(
                r"^" + re.escape(prefix_str) + kind + r"_(\d+)" + seed_suffix + r"$"
            )
            best: Optional[int] = None
            for name in names:
                m = pat.match(name)
                if m:
                    v = int(m.group(1))
                    best = v if best is None else max(best, v)
            return best

        spe = int(self.steps_per_epoch)
        # Choose ONE namespace for the whole endpoint: the first prefix that has
        # ANY step OR epoch record. Combine both formats within that prefix only.
        for p in prefixes:
            endpoints = []
            step_max = _max_for(p, "step")
            if step_max is not None:
                endpoints.append(step_max)
            epoch_max = _max_for(p, "epoch")
            if epoch_max is not None and spe > 0:
                endpoints.append((epoch_max + 1) * spe)
            if endpoints:
                return max(endpoints)
        return None

    def _clamped_window_step_bounds(self) -> Tuple[int, int]:
        """Subclass window bounds clamped to the RECORDED trajectory.

        ``self.total_steps`` is often the *nominal* ``num_epoch * steps_per_epoch``
        synthesized in ``InfluenceCalculator.__init__`` from ``global_info``,
        which records the *planned* ``num_epoch`` even for a partial/early-stopped
        run. A nominal ``w_end`` would then ask ``_u_at_step`` (query init) and the
        backward sweep for a checkpoint that was never written -- and for the
        query init that fails before the per-step error handling starts. Clamp
        ``w_end`` to the ACTUAL last recorded step (from disk); fall back to the
        nominal ``total_steps`` when the records dir can't be inspected, so full
        runs are a strict no-op.

        ``total_steps`` is always an UPPER bound: the epoch-file fallback in
        ``_last_recorded_step`` rounds up to ``(epoch + 1) * steps_per_epoch``,
        which overshoots the real trajectory when the final epoch was partial, so
        we cap the recorded value by ``total_steps`` to avoid requesting
        nonexistent steps (which ``load_step_data`` would serve by reusing the
        final ``step_info``, double-counting).
        """
        w_start, w_end = self._window_step_bounds()
        recorded = self._last_recorded_step()
        endpoint = (
            int(self.total_steps)
            if recorded is None
            else min(int(self.total_steps), int(recorded))
        )
        w_end = max(0, min(int(w_end), endpoint))
        w_start = max(0, min(int(w_start), w_end))
        return w_start, w_end

    def _init_query_u(self) -> List[torch.Tensor]:
        """Return the query vector ``u`` at the window-end model state."""
        raise NotImplementedError

    # -----------------------------
    # Sweep (shared)
    # -----------------------------
    def calculate(self) -> np.ndarray:
        """Reverse-SGD influence over ``[t1, t2]`` via Algorithm 1 (two queries).

        Faithful implementation of the paper's backward-sweep estimator
        (Algorithm 1 / Eq. window-est / Def. 2):

            Q_{-j}^{[t1,t2]}(q) = <q(t2), delta[t2]> - <q(t1), delta[t1]>.

        Two adjoint vectors are propagated backward with the influence
        propagator ``P^[t] = I - eta_t H^[t]``: ``u2`` carries ``q(t2)`` from the
        window end and is propagated at EVERY step; ``u1`` stays zero until the
        sweep reaches the window start ``t1 = w_start``, then carries ``q(t1)``
        and is propagated for all ``t < t1``. Whenever ``z_j`` is in the step's
        batch we accumulate ``<u2 - u1, 1̃_j>`` with the instantaneous influence
        ``1̃_j = (eta_t/|S_t|) g(z_j; theta^[t])`` (:meth:`_accumulate_influence`).

        The sweep runs down to step 0 (not merely to ``w_start``). The extra
        pre-window steps ``[0, t1)`` build the "evolution of pre-window
        deviation" term ``(prod P - I) delta[t1]`` of Eq. window-est that the
        earlier single-``u`` implementation dropped; that term is exactly zero
        only when ``t1 = 0``, so ``wie_first`` (and any full-trajectory window)
        is byte-identical to before, while ``wie_last``/``wie_middle`` gain the
        second query and the pre-window contribution.

        Subclass hooks are unchanged: :meth:`_window_step_bounds` gives
        ``(t1, t2)`` and :meth:`_init_query_u` gives ``q(t2)``; ``q(t1)`` is the
        window-start val-gradient via :meth:`_u_at_step`.
        """
        # Guard for DIRECT callers of wie.infl (the pipeline parser also rejects
        # this up front). A non-positive window length collapses every WIE
        # window to an empty interval, so calculate() would otherwise emit an
        # n_tr-row all-zero CSV that cleansing silently treats as success.
        if int(self.length) < 1:
            raise ValueError(
                f"{self._get_infl_type()}: window length must be >= 1 epoch, got "
                f"{self.length}. A non-positive length collapses the window to an "
                "empty interval; refusing to produce an all-zero score CSV."
            )
        w_start, w_end = self._clamped_window_step_bounds()
        # An empty window after clamping means the recorded trajectory stops
        # BEFORE the requested (late/middle/last) window -- e.g. 50 planned steps
        # but only 10 recorded, so wie_last's [35, 50] collapses to [10, 10].
        # Sweeping zero steps would silently save an n_tr-row all-zero CSV that
        # cleansing treats as success. Raise instead (honest failure); do NOT
        # recompute the window from the recorded endpoint, which would relabel a
        # different quantity as "last N epochs". wie_first ([0, recorded]) stays
        # non-empty on a partial run, so only the late-window case reaches here.
        if w_end <= w_start:
            recorded = self._last_recorded_step()
            raise ValueError(
                f"{self._get_infl_type()} window [{w_start}, {w_end}] is empty "
                f"after clamping to the recorded endpoint ({recorded}): the "
                "trajectory stops before the requested window (partial/"
                "early-stopped run); cannot compute this window."
            )
        self.logger.info(
            f"Calculating {self._get_infl_type()} influence over window steps "
            f"[{w_start}, {w_end}] (length={self.length} epochs)."
        )

        # 1) Initialize the two query adjoints.
        #    u2 = q(t2) at the window-end model; propagated at every step.
        #    u1 = 0 until the sweep reaches t1, then set to q(t1). Load q(t1)
        #    UP FRONT (not lazily inside the loop) so a missing window-start
        #    checkpoint fails loudly here rather than silently degrading the
        #    estimate to single-u for the pre-window steps.
        u2 = self._init_query_u()
        u1 = [torch.zeros_like(uu) for uu in u2]
        q_t1: Optional[List[torch.Tensor]] = (
            self._u_at_step(w_start) if w_start > 0 else None
        )
        u1_active = False

        # 2) Iterate backwards from the window end down to step 0. In the code's
        #    step convention the in-window steps are (t1, t2] (u1 == 0 there, so
        #    the accumulation is <u2, 1̃_j>, byte-identical to the previous
        #    single-u sweep); steps <= t1 are pre-window, where u1 = q(t1) is
        #    active and the accumulation becomes <u2 - u1, 1̃_j>.
        infl = np.zeros(self.n_tr, dtype=np.float64)
        self.logger.info(
            f"Reversing SGD from step {w_end} back to step 1 "
            f"(t1={w_start}: second query q(t1) "
            f"{'active for pre-window steps' if q_t1 is not None else 'disabled, t1=0'})"
        )

        for t in range(w_end, 0, -1):
            step_log_prefix = f"Step {t}/{self.total_steps}"
            try:
                # Activate the second query exactly at the window start t1.
                if q_t1 is not None and not u1_active and t == w_start:
                    u1 = q_t1
                    u1_active = True

                m_step, idx, lr, x_batch, y_batch = self._load_step_model_and_batch(t)
                # Effective per-step learning rate. Default (SGD): the recorded
                # eta_t. With use_effective_lr (Appendix-E checkpoint heuristic),
                # the observed step scale ||theta[t]-theta[t+1]||/||g_bar[t]||,
                # which equals eta_t for a true SGD step. Used everywhere eta_t
                # enters the recurrence (both the injection and the propagator).
                step_lr = self._step_lr(t, m_step, x_batch, y_batch, lr)
                if idx.numel() == 0:
                    # No target indices this step, but both adjoints must still
                    # propagate through P^[t] so earlier (lower-t) steps see the
                    # correct propagated state.
                    u2 = self._safe_update_u(
                        m_step, x_batch, y_batch, u2, step_lr, step_log_prefix
                    )
                    if u1_active:
                        u1 = self._safe_update_u(
                            m_step, x_batch, y_batch, u1, step_lr, step_log_prefix
                        )
                    continue
                # 2a) Per-sample gradients for the batch
                param_grads_list = self._compute_param_grads_list(
                    m_step, x_batch, y_batch, u2[0].dtype
                )
                # 2b) Accumulate influence: <u2 - u1, 1̃_j> (u1 == 0 in-window).
                u_eff = [a - b for a, b in zip(u2, u1)] if u1_active else u2
                self._accumulate_influence(infl, idx, param_grads_list, u_eff, step_lr)
                # 2c) Propagate both adjoints via P^[t] = I - eta_eff H^[t].
                u2 = self._safe_update_u(
                    m_step, x_batch, y_batch, u2, step_lr, step_log_prefix
                )
                if u1_active:
                    u1 = self._safe_update_u(
                        m_step, x_batch, y_batch, u1, step_lr, step_log_prefix
                    )
            except Exception as e:
                self.logger.error(
                    f"{step_log_prefix}: Error processing step: {e}", exc_info=True
                )
                continue

            if t % 100 == 0:
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return infl

    # -----------------------------
    # Helper methods (private) -- moved UNCHANGED from wie_last
    # -----------------------------
    def _query(self, model: torch.nn.Module) -> List[torch.Tensor]:
        """Query vector ``q`` evaluated at ``model``'s parameters (paper Def. 2).

        Selected by ``self.query_type`` (default ``"loss"``):

        - ``"loss"``: ``grad_theta l`` over the validation set (original behavior).
        - ``"prediction"``: ``grad_theta f(x_test)`` -- the ``query_class`` logit
          (or argmax) at the ``query_index``-th validation input.
        - ``"saliency"``: ``grad_theta ( d l / d x_test[query_coord] )`` -- one
          component of the input-gradient saliency at that input.

        Returns a per-parameter list; ``"loss"`` reproduces the previous query
        exactly, so default WIE results are unchanged.
        """
        qtype = getattr(self, "query_type", "loss")
        if qtype == "loss":
            return compute_gradient(self.x_val, self.y_val, model, self.loss_fn)
        if qtype == "prediction":
            return prediction_query(
                self._query_test_input(), model, getattr(self, "query_class", None)
            )
        if qtype == "saliency":
            return saliency_query(
                self._query_test_input(),
                self._query_test_target(),
                model,
                self.loss_fn,
                int(getattr(self, "query_coord", 0)),
            )
        raise ValueError(
            f"Unknown query_type {qtype!r}; expected 'loss', 'prediction', or "
            "'saliency'."
        )

    def _output_infl_type(self) -> str:
        """Output identity, suffixed with the query settings for a non-loss query.

        Prediction/saliency scores are saved under a DISTINCT name
        (``infl_{type}_query-..._{seed}.csv``) so they never overwrite -- or get
        resolved as -- the default loss-query WIE scores. The (validated, in-range)
        ``query_index`` is used verbatim, so the filename identifies the actual
        validation example. Loss (default) returns ``self.infl_type`` unchanged.
        """
        qtype = getattr(self, "query_type", "loss")
        if qtype == "loss":
            return self.infl_type
        parts = [
            self.infl_type,
            f"query-{qtype}",
            f"i{int(getattr(self, 'query_index', 0))}",
        ]
        if qtype == "prediction":
            qc = getattr(self, "query_class", None)
            parts.append("cargmax" if qc is None else f"c{int(qc)}")
        elif qtype == "saliency":
            parts.append(f"k{int(getattr(self, 'query_coord', 0))}")
        return "_".join(parts)

    def _query_test_index(self) -> int:
        """Validated ``query_index`` into the validation set (rejects out-of-range).

        Rejecting (rather than clamping) keeps the output filename honest: the
        saved ``..._i{idx}...`` name always identifies the example actually used.
        """
        idx = int(getattr(self, "query_index", 0))
        n = int(self.x_val.shape[0])
        if idx < 0 or idx >= n:
            raise ValueError(
                f"query_index {idx} out of range for {n} validation examples "
                f"(valid 0..{n - 1})."
            )
        return idx

    def _query_test_input(self) -> torch.Tensor:
        idx = self._query_test_index()
        return self.x_val[idx : idx + 1]

    def _query_test_target(self) -> torch.Tensor:
        idx = self._query_test_index()
        return self.y_val[idx : idx + 1]

    def _init_u(self, model: torch.nn.Module) -> List[torch.Tensor]:
        u = self._query(model)
        u = [uu.to(self.device) for uu in u]
        try:
            return [uu.to(torch.float64) for uu in u]
        except TypeError:
            self.logger.warning(
                "float64 not supported for u vector, falling back to float32"
            )
            return [uu.to(torch.float32) for uu in u]

    def _u_dtype(self) -> torch.dtype:
        """Select a safe dtype for u on the current device.

        On MPS, float64 is not supported, so we use float32. Otherwise, prefer
        float64 for numerical stability. (Ported from ``wie_all_epochs`` so that
        the window-end query init used by ``wie_first``/``wie_middle`` matches
        the per-epoch calculator's convention.)
        """
        try:
            device_type = (
                self.device.type
                if isinstance(self.device, torch.device)
                else str(self.device)
            )
        except Exception:
            device_type = "cpu"
        if device_type == "mps":
            return torch.float32
        return torch.float64

    def _u_at_step(self, step: int) -> List[torch.Tensor]:
        """Return the query ``u`` at the model state at ``step``.

        Used by windows that do NOT end at the final model (``wie_first`` /
        ``wie_middle``) and for the second query ``q(t1)``. The query is
        :meth:`_query` evaluated at the checkpoint (validation-loss gradient by
        default; prediction/saliency when ``query_type`` is set).
        """
        step_data_end = load_step_data(
            self.dn, step, self.seed, self.relabel_percentage, self.logger
        )
        if (
            not isinstance(step_data_end, dict)
            or "model_state" not in step_data_end
            or step_data_end["model_state"] is None
        ):
            raise FileNotFoundError(
                f"Step file {step} has no model_state; cannot initialize u."
            )
        m = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )
        m.load_state_dict(step_data_end["model_state"])
        m.eval()
        u = self._query(m)
        u_dtype = self._u_dtype()
        u = [uu.to(self.device).to(u_dtype) for uu in u]
        del m
        return u

    def _load_step_model_and_batch(
        self, t: int
    ) -> Tuple[torch.nn.Module, torch.Tensor, float, torch.Tensor, torch.Tensor]:
        step_data = load_step_data(
            self.dn, t, self.seed, self.relabel_percentage, self.logger
        )
        current_model_state = step_data["model_state"]
        idx_raw, lr = step_data["idx"], float(step_data["lr"])  # type: ignore

        m_step = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )
        m_step.load_state_dict(current_model_state)
        m_step.eval()

        if not isinstance(idx_raw, (list, np.ndarray, torch.Tensor)):
            idx_raw = [idx_raw]
        idx = torch.as_tensor(idx_raw, device=self.device)
        valid_idx = (idx >= 0) & (idx < self.n_tr)
        idx = idx[valid_idx]
        x_batch, y_batch = self.x_tr[idx], self.y_tr[idx]
        return m_step, idx, lr, x_batch, y_batch

    # -----------------------------
    # Appendix-E checkpoint effective-lr heuristic (opt-in via use_effective_lr)
    # -----------------------------
    def _batch_mean_grad(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Batch-mean (regularized) loss gradient ``g_bar^[t]`` at the model state.

        Matches the loss the recorded SGD update used (incl. the ``alpha`` L2
        term), so on a genuine SGD trajectory
        ``||theta[t]-theta[t+1]|| / ||g_bar|| == eta_t``.
        """
        model.zero_grad()
        loss = self.loss_fn(model(x), y)
        if self.alpha > 0:
            for p in model.parameters():
                loss = loss + 0.5 * self.alpha * (p * p).sum()
        loss.backward()
        g = [
            (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
            for p in model.parameters()
        ]
        model.zero_grad()
        return g

    def _has_step_checkpoint(self, step: int) -> bool:
        """True iff a genuine per-step (seeded) checkpoint file exists for ``step``.

        Only seeded ``{prefix}step_{step}_{seed:03d}.pt`` files are real per-step
        states that ``load_step_data`` consumes as ONE SGD step. When they are
        absent (epoch-only runs, or the unseeded ``step_NNNNNN.pt`` dumps the
        loader cannot read), ``load_step_data`` falls back to an epoch checkpoint,
        so ``theta[t]`` and ``theta[t-1]`` for intra-epoch ``t`` resolve to the
        SAME epoch file (delta 0) or, at boundaries, a whole epoch's movement --
        neither is a single SGD step, so the effective-lr delta is invalid. The
        caller must fall back to the nominal lr in that case.
        """
        records_dir = os.path.join(self.dn, "records")
        if step == 0:
            # theta[0] is the INITIAL model, saved as init_{seed:03d}.pt (not a
            # step file); used as theta[t-1] for the first update (t == 1).
            return os.path.isfile(
                os.path.join(records_dir, f"init_{int(self.seed):03d}.pt")
            )
        prefix = make_relabel_prefix(getattr(self, "relabel_percentage", None))
        for name in (
            f"{prefix}step_{step}_{int(self.seed):03d}.pt",
            f"step_{step}_{int(self.seed):03d}.pt",
        ):
            if os.path.isfile(os.path.join(records_dir, name)):
                return True
        return False

    def _load_prev_step_model(self, t: int) -> Optional[torch.nn.Module]:
        """Model at ``theta^[t-1]`` (state BEFORE update t), or ``None`` if absent.

        In this repo's convention (``TrainManager._save_each_step``:
        ``optimizer.step()`` then save ``step_{k}`` = state after update k), the
        batch/gradient recorded at index ``t`` drove the update
        ``theta[t-1] -> theta[t]`` using the gradient evaluated at the PRE-update
        state ``theta[t-1]``. The full model is returned (not just its params) so
        the effective-lr can recompute ``g_bar`` at that pre-update state -- which
        makes ``||theta[t]-theta[t-1]|| / ||g_bar(theta[t-1])||`` exactly the
        recorded lr for a genuine SGD step (a post-update gradient would not).

        The returned model is left in the mode :meth:`_step_lr` sets for the
        gradient recomputation (train mode, to match the recorded update for
        BatchNorm/dropout models); it is NOT forced to eval here.
        """
        try:
            if t - 1 == 0:
                # theta[0] is the initial model (init_{seed}.pt), not a step file.
                state = load_initial_model(self.dn, self.seed, self.device, self.logger)
            else:
                sd = load_step_data(
                    self.dn, t - 1, self.seed, self.relabel_percentage, self.logger
                )
                state = sd.get("model_state") if isinstance(sd, dict) else None
            if state is None:
                return None
            m = get_network(self.model_type, self.input_dim, logger=self.logger).to(
                self.device
            )
            m.load_state_dict(state)
            return m
        except Exception as e:  # missing/first step -> fall back to nominal lr
            self.logger.debug(f"No theta[{t - 1}] model for effective-lr: {e}")
            return None

    def _step_lr(
        self,
        t: int,
        m_step: torch.nn.Module,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        nominal_lr: float,
    ) -> float:
        """Effective per-step learning rate used throughout the recurrence.

        Default (SGD): the recorded ``nominal_lr`` (``eta_t``). With
        ``use_effective_lr`` (Appendix-E checkpoint heuristic), the scalar
        observed step scale ``||theta[t]-theta[t-1]|| / ||g_bar(theta[t-1])||``
        -- which equals ``eta_t`` EXACTLY for a true SGD step and recovers the
        effective step magnitude of an Adam/AdamW step. Both the delta and the
        gradient use the PRE-update state ``theta[t-1]`` (the batch recorded at
        index ``t`` drove ``theta[t-1] -> theta[t]`` with the gradient at
        ``theta[t-1]``; see :meth:`_load_prev_step_model`). Falls back to
        ``nominal_lr`` when the adjacent checkpoint or a usable gradient norm is
        unavailable.
        """
        if not getattr(self, "use_effective_lr", False):
            return nominal_lr
        # Require genuine per-step checkpoints for BOTH t and t-1; otherwise the
        # loaded states come from the epoch fallback and their delta is not one
        # SGD step (zero for intra-epoch, whole-epoch at boundaries).
        if not (self._has_step_checkpoint(t) and self._has_step_checkpoint(t - 1)):
            # Make the fallback LOUD (once): otherwise --effective-lr would
            # silently reproduce the default scores. The bundled trainer writes
            # UNSEEDED 'step_NNNNNN.pt' (which load_step_data cannot consume) or
            # epoch-only checkpoints, so the sweep already runs on epoch-fallback
            # states and the per-step delta is unavailable. Seeded per-step
            # 'step_{t}_{seed:03d}.pt' records are required to activate it.
            if not getattr(self, "_eff_lr_warned", False):
                self.logger.warning(
                    "use_effective_lr is set but genuine per-step checkpoints "
                    "(seeded 'step_{t}_{seed:03d}.pt', consumable by "
                    "load_step_data) are unavailable for this trajectory; the "
                    "Appendix-E effective-lr heuristic is INACTIVE and the "
                    "nominal recorded lr is used for the whole run (results equal "
                    "the default). Re-record with seeded per-step checkpoints to "
                    "enable it."
                )
                self._eff_lr_warned = True
            return nominal_lr
        prev_model = self._load_prev_step_model(t)
        if prev_model is None:
            return nominal_lr
        theta_t = [p.detach() for p in m_step.parameters()]
        theta_prev = [p.detach() for p in prev_model.parameters()]
        # g_bar at the PRE-update state theta[t-1], in TRAIN mode to match the
        # recorded optimizer step for models with training-only layers (BatchNorm
        # uses batch stats, dropout is active); an eval-mode gradient would
        # mismatch the update and break the SGD no-op invariant for BN/dropout
        # models. (Dropout's randomness still makes it approximate, but train mode
        # avoids the systematic eval-mode error.)
        prev_model.train()
        grad_prev = self._batch_mean_grad(prev_model, x_batch, y_batch)
        eff = scalar_effective_lr(theta_t, theta_prev, grad_prev, nominal_lr)
        if eff != nominal_lr:
            self.logger.debug(
                f"Step {t}: effective lr {eff:.3e} (nominal {nominal_lr:.3e})"
            )
        return eff

    def _compute_param_grads_list(
        self,
        m_step: torch.nn.Module,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        dtype: torch.dtype,
    ) -> List[List[torch.Tensor]]:
        batch_size = x_batch.shape[0]
        param_grads_list: List[List[torch.Tensor]] = []
        for i_local in range(batch_size):
            m_step.zero_grad()
            z_i = m_step(x_batch[[i_local]])
            loss_i = self.loss_fn(z_i, y_batch[[i_local]])
            if self.alpha > 0:
                for p in m_step.parameters():
                    loss_i += 0.5 * self.alpha * (p * p).sum()
            loss_i.backward()
            grad_i = [
                (
                    p.grad.data.clone().to(dtype=dtype)
                    if p.grad is not None
                    else torch.zeros_like(p, dtype=dtype)
                )
                for p in m_step.parameters()
            ]
            param_grads_list.append(grad_i)
        m_step.zero_grad()
        return param_grads_list

    def _accumulate_influence(
        self,
        infl: np.ndarray,
        idx: torch.Tensor,
        param_grads_list: List[List[torch.Tensor]],
        u: List[torch.Tensor],
        lr: float,
    ) -> None:
        batch_size = len(idx)
        for i_local, sample_idx in enumerate(idx.tolist()):
            grad_i = param_grads_list[i_local]
            grad_sum = sum(
                torch.sum(u[j].data * param_grad).item()
                for j, param_grad in enumerate(grad_i)
                if j < len(u)
            )
            infl[sample_idx] += lr * grad_sum / max(batch_size, 1)

    def _hvp(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        v: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Exact Hessian-vector product ``H^[t] v`` via Pearlmutter double-backprop.

        Implements Eq.(hvp): ``H^[t] v = (1/|S_t|) sum_i grad(grad l_i^T v)`` in a
        single reverse-over-reverse autograd pass on the mean batch loss (its
        gradient is the batch-mean, so the second grad is exactly the batch-mean
        Hessian applied to ``v``). When L2 regularization ``alpha > 0`` is used in
        training, the loss carries ``0.5*alpha*||theta||^2`` whose Hessian is
        ``alpha*I``; we add ``alpha*v`` so the propagator ``P=I-eta(H+alpha I)``
        matches the actual (regularized) SGD dynamics.

        This replaces the previous central finite-difference approximation plus
        the undocumented adaptive Tikhonov damping ``0.05*||u||/||Hu||``: that
        damping made the propagator NONLINEAR in the propagated vector, which is
        both inconsistent with the paper's clean ``P=I-eta H`` (the operator
        Theorem 1 bounds) and incompatible with the two-query estimator, whose
        ``<u2-u1, .>`` accounting requires a LINEAR propagator shared by both
        adjoints.

        ``v`` may be float64 while the model runs in float32; ``grad_outputs``
        must match the (param-dtype) gradients, so ``v`` is cast down for the
        product and the result cast back to ``v``'s dtype -- identical to the old
        finite-difference helper's dtype handling.
        """
        params = [p for p in model.parameters()]
        assert len(params) == len(v), "v must align with model parameters"
        out_dtype, out_device = v[0].dtype, v[0].device

        v_cast = [vv.to(dtype=p.dtype, device=p.device) for vv, p in zip(v, params)]
        model.zero_grad()
        out = model(x)
        loss = self.loss_fn(out, y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        hv = torch.autograd.grad(grads, params, grad_outputs=v_cast, retain_graph=False)
        hv = [h.detach() for h in hv]
        if self.alpha > 0:
            hv = [h + self.alpha * vc for h, vc in zip(hv, v_cast)]
        model.zero_grad()
        return [h.to(dtype=out_dtype, device=out_device) for h in hv]

    def _safe_update_u(
        self,
        m_step: torch.nn.Module,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        u: List[torch.Tensor],
        lr: float,
        step_log_prefix: str,
    ) -> List[torch.Tensor]:
        """Propagate an adjoint one step backward: ``u <- P^[t] u = u - eta_t H^[t] u``.

        Clean linear propagator (no damping). A NaN/inf guard falls back to the
        pre-update ``u`` for that tensor so a single unstable step cannot poison
        the whole sweep, and any HVP exception logs and skips the update.
        """
        u_prev = [uu.clone() for uu in u]
        try:
            hvp = self._hvp(m_step, x_batch, y_batch, u)
            new_u: List[torch.Tensor] = []
            for j in range(len(u)):
                new_u_val = u[j] - lr * hvp[j]
                if torch.isnan(new_u_val).any() or torch.isinf(new_u_val).any():
                    new_u.append(u_prev[j])
                else:
                    new_u.append(new_u_val)
            return new_u
        except Exception as e:
            self.logger.warning(
                f"{step_log_prefix}: Error during HVP update: {e}. Skipping u update."
            )
            return u_prev
