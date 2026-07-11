import os
import re
import gc
import numpy as np
import torch

from ..models.networks import get_network
from ..io.naming import make_relabel_prefix
from .core import (
    InfluenceCalculator,
    InfluenceCalculatorFactory,
    load_epoch_data,
    save_results,
)


def _cf_models_seq(cf_entry):
    """Per-epoch sequence of counterfactual models for one training sample.

    Handles the trainer's actual output -- a ``NetList`` whose ``.models`` is a
    ``ModuleList`` (or list) of per-epoch models -- as well as the older
    ``.models.models`` nesting, a ``dict`` with a ``"models"`` list, or a bare
    list of models/state-dicts. Returns ``None`` if the structure is
    unrecognized. Elements may be model objects (``.state_dict()``) or raw
    state-dicts; callers handle both.
    """
    seq_types = (list, tuple, torch.nn.ModuleList)
    models_attr = getattr(cf_entry, "models", None)
    if isinstance(models_attr, seq_types):
        return models_attr
    if hasattr(models_attr, "models") and isinstance(models_attr.models, seq_types):
        return models_attr.models
    if isinstance(cf_entry, dict) and isinstance(cf_entry.get("models"), seq_types):
        return cf_entry["models"]
    if isinstance(cf_entry, seq_types):
        return cf_entry
    return None


@InfluenceCalculatorFactory.register("segment_true_full")
class SegmentTrueFullInfluenceCalculator(InfluenceCalculator):
    """
    Computes cumulative 'true' influence state at each epoch boundary using counterfactual models.
    For epoch e, compares validation loss between the baseline model at epoch e and the
    counterfactual model (leave-one-out) at epoch e for every training sample.
    Saves a list with length equal to number of epochs; element e stores an array of size n_tr.
    """

    def _get_infl_type(self) -> str:
        return "segment_true_full"

    def _load_counterfactual_per_sample(self):
        """Build the cf list from per-sample per-epoch files (complete + reliable).

        The trainer writes one file per (sample, epoch)
        ``{prefix}counterfactual_{i:04d}_epoch_{e}_{seed:03d}.pt`` -- a dict with
        a ``model_state``. Unlike the consolidated archive (whose LAST sample's
        NetList can be empty: it is saved before the final counterfactual run
        appends), these are always complete and aligned 1:1 with the base epoch
        checkpoints. Returns a list, indexed by sample, of per-epoch state-dict
        lists; or ``None`` if no such files exist (fall back to the archive).
        """
        records_dir = os.path.join(self.dn, "records")
        if not os.path.isdir(records_dir):
            return None
        prefix = make_relabel_prefix(self.relabel_percentage)
        pat = re.compile(
            r"^"
            + re.escape(prefix)
            + r"counterfactual_(\d+)_epoch_(\d+)_"
            + f"{int(self.seed):03d}"
            + r"\.pt$"
        )
        by_sample: dict[int, dict[int, str]] = {}
        for name in os.listdir(records_dir):
            m = pat.match(name)
            if m:
                i, e = int(m.group(1)), int(m.group(2))
                by_sample.setdefault(i, {})[e] = os.path.join(records_dir, name)
        if not by_sample:
            return None
        # Load EXACTLY epochs 0..num_epoch-1: ignore stale higher-epoch files left
        # by a reused/longer prior run (they would shift the epoch alignment). If
        # the training epoch count is unknown, fall back to the max contiguous run
        # 0..k present in every sample.
        n_want = int(getattr(self, "num_epoch", 0) or 0)
        if n_want <= 0:
            common = set.intersection(*(set(v) for v in by_sample.values()))
            n_want = 0
            while n_want in common:
                n_want += 1
            if n_want <= 0:
                return None
        want = list(range(n_want))
        # Require every EXPECTED sample (0..n_tr-1), not just those present: an
        # interrupted run may have files for only samples 0..k, and returning a
        # short list would trip the caller's len<n_tr guard WITHOUT falling back
        # to the consolidated archive. Missing IDs above max(by_sample) count as
        # incomplete -> None.
        n_samples = int(getattr(self, "n_tr", 0) or 0) or (max(by_sample) + 1)
        cf_list = []
        for i in range(n_samples):
            epochs = by_sample.get(i, {})
            if not all(e in epochs for e in want):
                # This run's per-sample files are incomplete -> don't guess;
                # fall back to the consolidated archive path.
                self.logger.warning(
                    f"Per-sample counterfactual files incomplete for sample {i} "
                    f"(need epochs 0..{n_want - 1}, have {sorted(epochs)}); "
                    "falling back to the consolidated archive."
                )
                return None
            seq = []
            for e in want:
                d = torch.load(epochs[e], map_location="cpu", weights_only=False)
                seq.append(d.get("model_state") if isinstance(d, dict) else d)
            cf_list.append(seq)
        self.logger.info(
            f"Loaded {len(cf_list)} counterfactual trajectories x {n_want} epochs "
            "from per-sample files (avoids the possibly-stale consolidated archive)."
        )
        return cf_list

    def _resolve_cf_epochs(self, num_epochs_cf):
        """Return ``(n_epochs, cf_offset)`` for aligning cf snapshots to base epochs.

        segment_true_full emits one row per TRAINED epoch and the true_* window
        oracle depends on that alignment. The cf trajectory may carry exactly ONE
        extra leading snapshot (the shared initial model, ``cf[0] == init``), so we
        take the cf TAIL: base epoch ``e`` <-> ``cf[cf_offset + e]``.

        Fails loudly rather than silently mis-aligning:
        - ``num_epoch > num_epochs_cf`` (archive has FEWER snapshots than trained
          epochs -- partial/interrupted counterfactual run) -> raise, don't
          truncate the output.
        - ``cf_offset > 1`` (more than the one allowed extra snapshot -- a
          longer/other run's archive) -> raise, don't guess the alignment.
        Only when the epoch metadata is unavailable (``num_epoch <= 0``) do we
        trust the archive's own snapshot count.
        """
        n_epochs = int(getattr(self, "num_epoch", 0) or 0)
        if n_epochs <= 0:
            n_epochs = num_epochs_cf  # metadata unavailable: trust the archive
        elif n_epochs > num_epochs_cf:
            raise ValueError(
                f"segment_true_full: training recorded {n_epochs} epochs but the "
                f"counterfactual archive has only {num_epochs_cf} snapshots -- the "
                "counterfactual run is partial/interrupted. Refusing to emit a "
                "truncated segment_true_full (the true_* oracle needs one row per "
                "trained epoch); regenerate the counterfactual data."
            )
        cf_offset = num_epochs_cf - n_epochs
        if cf_offset > 1:
            raise ValueError(
                f"segment_true_full: counterfactual trajectory has {num_epochs_cf} "
                f"snapshots but only {n_epochs} trained epochs (at most +1 for the "
                "initial model is allowed); this looks like a stale/longer archive. "
                "Refusing to guess the epoch alignment -- regenerate the "
                "counterfactual data for this run."
            )
        return n_epochs, cf_offset

    def _assert_cf_complete(self, cf_list, required):
        """Raise unless every expected sample has >= ``required`` epoch snapshots.

        An incomplete counterfactual set -- the consolidated archive whose final
        sample's NetList is empty (saved before the last run appends), or an
        interrupted run -- would otherwise hit a per-sample IndexError that
        ``calculate`` catches and records as ``0.0``, silently producing a
        bogus-but-"successful" ``segment_true_full`` CSV.
        """
        short = []
        for i in range(min(int(self.n_tr), len(cf_list))):
            seq_i = _cf_models_seq(cf_list[i])
            have = 0 if seq_i is None else len(seq_i)
            if have < required:
                short.append((i, have))
        if short:
            raise ValueError(
                f"segment_true_full: {len(short)} counterfactual entries have "
                f"fewer than {required} epoch snapshots (e.g. sample->count "
                f"{short[:5]}); the counterfactual data is incomplete (interrupted "
                "run, or a stale/partial archive written before the final run "
                "appended). Refusing to emit a silently-zeroed segment_true_full; "
                "re-run training with complete --compute_counterfactual output."
            )

    def _load_counterfactual_list(self):
        """Load the list of counterfactual models (one entry per training sample)."""
        # Prefer per-sample per-epoch files -- always complete. The consolidated
        # archive can carry an empty final-sample NetList (saved before the last
        # run appends), which would silently record 0.0 for that sample.
        per_sample = self._load_counterfactual_per_sample()
        if per_sample is not None:
            return per_sample
        # First try the main results fallback (legacy behavior may include 'counterfactual')
        try:
            res = torch.load(self.fn_fallback, map_location="cpu", weights_only=False)
            if isinstance(res, dict) and "counterfactual" in res:
                return res["counterfactual"]
        except Exception:
            pass

        # Then try dedicated counterfactual file saved by train.py
        relabel_prefix = (
            f"relabel_{int(self.relabel_percentage):03d}_pct_"
            if self.relabel_percentage is not None
            else ""
        )
        cf_filename = f"{relabel_prefix}counterfactual_models_{self.seed:03d}.pt"
        # The trainer writes the consolidated file under records/; older layouts
        # kept it at the run root. Try both.
        candidates = [
            os.path.join(self.dn, "records", cf_filename),
            os.path.join(self.dn, cf_filename),
        ]
        cf_path = next((p for p in candidates if os.path.isfile(p)), candidates[0])
        cf_list = torch.load(cf_path, map_location="cpu", weights_only=False)
        return cf_list

    def calculate(self) -> list[np.ndarray]:
        self.logger.info("Starting Segment-True (full) influence calculation...")

        # Load counterfactual models list
        cf_list = self._load_counterfactual_list()
        if not isinstance(cf_list, (list, tuple)) or len(cf_list) < self.n_tr:
            raise ValueError(
                "Counterfactual models list invalid or shorter than n_tr; cannot compute segment_true_full."
            )

        # Prepare reusable models
        m_base = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )
        m_cf = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )

        out: list[np.ndarray] = []

        # Determine number of epochs recorded for counterfactuals from the first entry
        first_seq = _cf_models_seq(cf_list[0])
        if first_seq is None:
            raise ValueError(
                "Unrecognized counterfactual entry structure for epoch count."
            )
        num_epochs_cf = len(first_seq)
        if num_epochs_cf <= 0:
            raise ValueError("Counterfactual epoch count is zero.")

        n_epochs, cf_offset = self._resolve_cf_epochs(num_epochs_cf)

        # Validate EVERY counterfactual entry BEFORE computing (fail loudly on an
        # incomplete set instead of silently recording 0.0 -> bogus CSV).
        self._assert_cf_complete(cf_list, cf_offset + n_epochs)

        for e in range(n_epochs):
            self.logger.info(f"--- Segment-True: processing epoch {e} ---")

            # Load baseline model at epoch e
            try:
                epoch_data = load_epoch_data(
                    self.dn, e, self.seed, self.relabel_percentage, self.logger
                )
                m_base.load_state_dict(epoch_data["model_state"])
                m_base.eval()
            except Exception as ex:
                self.logger.error(
                    f"Failed to load baseline model for epoch {e}: {ex}", exc_info=True
                )
                # Skip this epoch; append zeros to keep alignment
                out.append(np.zeros(self.n_tr, dtype=np.float64))
                continue

            # Baseline val loss
            with torch.no_grad():
                z_base = m_base(self.x_val)
                base_loss = self.loss_fn(z_base, self.y_val).item()

            infl_epoch = np.zeros(self.n_tr, dtype=np.float64)

            for i in range(self.n_tr):
                cf_entry = cf_list[i]
                try:
                    # Extract the model (or state-dict) at epoch e for sample i.
                    seq = _cf_models_seq(cf_entry)
                    if seq is None:
                        raise ValueError(
                            f"Unsupported counterfactual entry type at index {i}: "
                            f"{type(cf_entry)}"
                        )
                    cf_model_obj = seq[cf_offset + e]
                    if hasattr(cf_model_obj, "state_dict") and callable(
                        cf_model_obj.state_dict
                    ):
                        # Do NOT move the stored checkpoint object to the device:
                        # m_cf already lives on self.device and load_state_dict
                        # copies the (CPU) tensors in. Moving cf_model_obj would
                        # leave every processed counterfactual resident on the GPU
                        # for the rest of the run (OOM for large n_tr/epochs).
                        m_cf.load_state_dict(cf_model_obj.state_dict())
                    else:
                        m_cf.load_state_dict(cf_model_obj)  # raw state-dict

                    m_cf.eval()
                    with torch.no_grad():
                        zi = m_cf(self.x_val)
                        lossi = self.loss_fn(zi, self.y_val)
                    infl_epoch[i] = (lossi - base_loss).item()
                except Exception as e_inner:
                    self.logger.warning(
                        f"Epoch {e}: error processing counterfactual for sample {i}: {e_inner}. Set 0."
                    )
                    infl_epoch[i] = 0.0

                if (i + 1) % 200 == 0:
                    self.logger.info(
                        f"Epoch {e}: processed {i + 1}/{self.n_tr} counterfactuals"
                    )

            out.append(infl_epoch)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save as 'segment_true_full'
        save_results(
            out,
            self.dn,
            self.seed,
            self._get_infl_type(),
            self.logger,
            self.relabel_percentage,
        )
        return out
