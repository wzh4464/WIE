import gc
import numpy as np
import torch
from typing import Any, Dict, Tuple

from ..models.networks import get_network
from .core import InfluenceCalculator, InfluenceCalculatorFactory


@InfluenceCalculatorFactory.register("true")
class TrueInfluenceCalculator(InfluenceCalculator):
    """Computes 'true' influence by comparing val loss with counterfactual models."""

    def _get_infl_type(self) -> str:
        return "true"

    def calculate(self) -> np.ndarray:
        """Compute 'true' influence by comparing base vs counterfactual losses."""
        res, model = self._load_results_and_final_model()

        counterfactuals = res["counterfactual"]
        num_counterfactuals = len(counterfactuals)
        if num_counterfactuals < self.n_tr:
            self.logger.warning(
                f"Found {num_counterfactuals} counterfactuals, expected {self.n_tr}. Missing indices set to 0."
            )

        infl = np.zeros(self.n_tr, dtype=np.float64)
        base_loss = self._compute_base_loss(model)

        # Template model for counterfactual evaluation
        m_cf = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )
        m_cf.eval()

        for i in range(self.n_tr):
            if i >= num_counterfactuals:
                infl[i] = 0.0
                continue
            cf_item = counterfactuals[i]
            if cf_item is None:
                self.logger.warning(
                    f"Counterfactual result for index {i} is None. Setting influence to 0."
                )
                infl[i] = 0.0
                continue
            try:
                self._load_counterfactual_into_model(cf_item, m_cf)
                with torch.no_grad():
                    lossi = self.loss_fn(m_cf(self.x_val), self.y_val).item()
                infl[i] = lossi - base_loss
            except (AttributeError, IndexError, KeyError, TypeError) as e_inner:
                self.logger.warning(
                    f"Error processing counterfactual for index {i}: {e_inner}. Setting influence to 0."
                )
                infl[i] = 0.0

            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i + 1}/{self.n_tr} counterfactuals.")

        del res, counterfactuals, model, m_cf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return infl

    # -----------------------------
    # Helpers (private)
    # -----------------------------
    def _load_results_and_final_model(self) -> Tuple[Dict[str, Any], torch.nn.Module]:
        self.logger.info(
            f"Loading full results object from fallback file: {self.fn_fallback}"
        )
        try:
            res = torch.load(self.fn_fallback, map_location="cpu", weights_only=False)
            self.logger.info("Successfully loaded main results object.")
        except FileNotFoundError:
            self.logger.error(
                f"Required results file {self.fn_fallback} not found for 'true' influence."
            )
            raise
        except Exception as e:
            self.logger.error(f"Error loading results file {self.fn_fallback}: {e}")
            raise

        if "counterfactual" not in res or not isinstance(res["counterfactual"], list):
            self.logger.error(
                f"Counterfactual models not found or invalid in {self.fn_fallback}."
            )
            raise ValueError("Missing or invalid counterfactual data.")

        model = get_network(self.model_type, self.input_dim, logger=self.logger).to(
            self.device
        )
        state = self._extract_final_model_state(res)
        model.load_state_dict(state)
        model.eval()
        self.logger.info("Loaded final model state from results object.")
        return res, model

    def _extract_final_model_state(self, res: Dict[str, Any]) -> Dict:
        try:
            models_attr = res.get("models")
            is_netlist_like = hasattr(models_attr, "models") and isinstance(
                getattr(models_attr, "models", None), list
            )
            if is_netlist_like and models_attr.models:
                return models_attr.models[-1].state_dict()
            if "final_model_state" in res:
                return res["final_model_state"]
            raise KeyError("Final model state not found in results object")
        except (KeyError, AttributeError, IndexError) as e:
            self.logger.error(
                f"Could not extract final model state from {self.fn_fallback}: {e}"
            )
            raise ValueError(f"Cannot load final model from {self.fn_fallback}")

    def _compute_base_loss(self, model: torch.nn.Module) -> float:
        with torch.no_grad():
            z_base = model(self.x_val)
            base_loss = self.loss_fn(z_base, self.y_val).item()
            self.logger.info(f"Base loss (final model): {base_loss:.6f}")
        return float(base_loss)

    def _load_counterfactual_into_model(
        self, cf_item: Any, model: torch.nn.Module
    ) -> None:
        cf_models_attr = getattr(cf_item, "models", None)
        is_cf_netlist_like = hasattr(cf_models_attr, "models") and isinstance(
            getattr(cf_models_attr, "models", None), list
        )
        if is_cf_netlist_like and cf_models_attr.models:
            counterfactual_model_obj = cf_models_attr.models[-1]
            counterfactual_model_obj.to(self.device)
            model.load_state_dict(counterfactual_model_obj.state_dict())
        elif isinstance(cf_item, dict) and "final_model_state" in cf_item:
            model.load_state_dict(cf_item["final_model_state"])
        elif hasattr(cf_item, "state_dict") and callable(cf_item.state_dict):
            cf_item.to(self.device)
            model.load_state_dict(cf_item.state_dict())
        else:
            raise AttributeError("Unrecognized counterfactual model structure")
