import logging

import numpy as np
import torch

from .core import InfluenceCalculator, InfluenceCalculatorFactory


def _flatten_features(x: torch.Tensor) -> torch.Tensor:
    if x.dim() <= 2:
        return x.view(x.size(0), -1)
    return x.reshape(x.size(0), -1)


def _simple_centroid_fallback(
    x_tr: torch.Tensor, y_tr: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor
) -> np.ndarray:
    """
    Native, dependency-free LAVA-like valuation.

    Heuristic: for each training sample, compute the negative Euclidean
    distance to its class-conditional validation centroid; if that class is
    absent in validation, fall back to the global validation centroid.

    Lower distances => higher scores. Outputs a single static vector aligned
    with the repository's CSV schema (sample_idx,influence).
    """
    device = x_tr.device
    x_tr_f = _flatten_features(x_tr).to(device)
    x_val_f = _flatten_features(x_val).to(device)

    # Normalize dtype
    x_tr_f = x_tr_f.to(torch.float32)
    x_val_f = x_val_f.to(torch.float32)

    # Prepare labels (int)
    y_tr_i = y_tr.detach()
    y_val_i = y_val.detach()
    if y_tr_i.dim() > 1:
        y_tr_i = (y_tr_i > 0.5).long().view(-1)
    else:
        y_tr_i = y_tr_i.long().view(-1)
    if y_val_i.dim() > 1:
        y_val_i = (y_val_i > 0.5).long().view(-1)
    else:
        y_val_i = y_val_i.long().view(-1)

    # Compute class-wise centroids on validation
    unique_classes = torch.unique(y_tr_i)
    centroids = {}
    for c in unique_classes.tolist():
        mask = y_val_i == int(c)
        if mask.any():
            centroids[int(c)] = x_val_f[mask].mean(dim=0)

    # Global centroid fallback
    if len(centroids) == 0:
        global_centroid = x_val_f.mean(dim=0)
        dists = torch.cdist(x_tr_f, global_centroid.view(1, -1)).view(-1)
        return (-dists).to("cpu").numpy().astype(np.float32)

    # Per-sample distance to its class centroid (or global if missing)
    global_centroid = x_val_f.mean(dim=0)
    scores = torch.empty(x_tr_f.size(0), device=device)
    for i in range(x_tr_f.size(0)):
        cls = int(y_tr_i[i].item())
        ctr = centroids.get(cls, global_centroid)
        d = torch.norm(x_tr_f[i] - ctr, p=2)
        scores[i] = -d
    return scores.to("cpu").numpy().astype(np.float32)


@InfluenceCalculatorFactory.register("lava")
class LavaInfluenceCalculator(InfluenceCalculator):
    """
    LAVA-style data valuation (model-less), implemented natively without
    external dependencies. Produces a single static influence vector that is
    compatible with the repository's cleansing and plotting pipelines.
    """

    def __init__(self, infl_type: str, **kwargs):
        super().__init__(infl_type, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_infl_type(self) -> str:
        return "lava"

    def calculate(self) -> np.ndarray:  # type: ignore[override]
        # Fully native, model-less valuation
        vals = _simple_centroid_fallback(self.x_tr, self.y_tr, self.x_val, self.y_val)
        return np.asarray(vals, dtype=np.float32)


__all__ = ["LavaInfluenceCalculator"]
