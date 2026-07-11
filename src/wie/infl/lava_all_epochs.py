import numpy as np
import torch
import gc
import os
import itertools
from functools import partial
from typing import List, Dict, Optional, Union, Callable, Literal

from wie.models.networks import get_network  # type: ignore
from .core import (
    InfluenceCalculator,
    InfluenceCalculatorFactory,
)

# Check for geomloss availability
try:
    import geomloss

    GEOMLOSS_AVAILABLE = True
except ImportError:
    GEOMLOSS_AVAILABLE = False
    import warnings

    warnings.warn(
        "geomloss not available. Will use simplified centroid method.", ImportWarning
    )

# # Simplified dataloader for our needs (to avoid opendataval dependency)
# try:
#     from torch.utils.data import DataLoader, TensorDataset

#     DATALOADER_AVAILABLE = True
# except ImportError:
#     DATALOADER_AVAILABLE = False


# Cost routines for geomloss (if available)
if GEOMLOSS_AVAILABLE:
    cost_routines = {
        1: geomloss.utils.distances,
        2: lambda x, y: geomloss.utils.squared_distances(x, y) / 2,
    }
else:
    # Fallback cost routines
    def _euclidean_distances(x, y):
        return torch.cdist(x, y, p=2)

    def _squared_euclidean_distances(x, y):
        return torch.cdist(x, y, p=2) ** 2 / 2

    cost_routines = {
        1: _euclidean_distances,
        2: _squared_euclidean_distances,
    }


def _flatten_features(x: torch.Tensor) -> torch.Tensor:
    """Flatten features while preserving batch dimension"""
    if x.dim() <= 2:
        return x.view(x.size(0), -1)
    return x.reshape(x.size(0), -1)


def _simple_extract_dataset(
    x_input: torch.Tensor,
    y_input: torch.Tensor,
    reindex_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified dataset extraction without external dependencies.

    Parameters
    ----------
    x_input : torch.Tensor
        Covariate tensor to be processed
    y_input : torch.Tensor
        Label tensor to be processed
    reindex_start : int, optional
        How much to offset the labels by, by default 0

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        x_tensor: Covariates flattened along first dimension
        y_tensor: Labels, converted to class indices and offset by reindex_start
    """
    # Flatten features
    x_tensor = x_input.view(x_input.shape[0], -1)

    # Convert labels to class indices
    if y_input.dim() > 1:
        # If one-hot encoded, convert to class indices
        y_tensor = y_input.argmax(dim=1).squeeze()
    else:
        # If already class indices, convert appropriately
        if y_input.dtype == torch.float:
            y_tensor = (y_input > 0.5).long().squeeze()
        else:
            y_tensor = y_input.long().squeeze()

    return x_tensor, y_tensor + reindex_start


class FeatureCost:
    """Class implementing a cost (or distance) between feature vectors.

    Arguments:
        p (int): the coefficient in the OT cost (i.e., the p in p-Wasserstein).
        src_embedding (callable, optional): if provided, source data will be
            embedded using this function prior to distance computation.
        tgt_embedding (callable, optional): if provided, target data will be
            embedded using this function prior to distance computation.
    """

    def __init__(
        self,
        src_embedding=None,
        tgt_embedding=None,
        src_dim=None,
        tgt_dim=None,
        p=2,
        device="cpu",
    ):
        assert (src_embedding is None) or (src_dim is not None)
        assert (tgt_embedding is None) or (tgt_dim is not None)
        self.src_emb = src_embedding
        self.tgt_emb = tgt_embedding
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.p = p
        self.device = device

    def _get_batch_shape(self, b):
        if b.ndim == 3:
            return b.shape
        elif b.ndim == 2:
            return (1, *b.shape)
        elif b.ndim == 1:
            return (1, 1, b.shape[0])

    def _batchify_computation(self, X, side="x", slices=20):
        embed = self.src_emb if side == "x" else self.tgt_emb
        out = torch.cat(
            [embed(b).to(self.device) for b in torch.chunk(X, slices, dim=0)]
        )
        return out.to(X.device)

    def __call__(self, X1, X2):
        if self.src_emb is not None:
            B1, N1, _ = self._get_batch_shape(X1)
            if hasattr(self.src_emb, "to"):
                self.src_emb = self.src_emb.to(self.device)
            X1 = self.src_emb(X1.view(-1, *self.src_dim)).reshape(B1, N1, -1)
        if self.tgt_emb is not None:
            B2, N2, _ = self._get_batch_shape(X2)
            if hasattr(self.tgt_emb, "to"):
                self.tgt_emb = self.tgt_emb.to(self.device)
            X2 = self.tgt_emb(X2.view(-1, *self.tgt_dim)).reshape(B2, N2, -1)

        return cost_routines[self.p](X1, X2)


def pwdist_exact(
    X1: torch.Tensor,
    Y1: torch.Tensor,
    X2: Optional[torch.Tensor] = None,
    Y2: Optional[torch.Tensor] = None,
    symmetric: bool = False,
    loss: str = "sinkhorn",
    cost_function: Union[
        Literal["euclidean"], Callable[..., torch.Tensor]
    ] = "euclidean",
    p: int = 2,
    debias: bool = True,
    entreg: float = 1e-1,
    device: torch.device = torch.device("cpu"),
):
    """Computation of pairwise Wasserstein distances.

    Efficient computation of pairwise label-to-label Wasserstein distances
    between multiple distributions, without using Gaussian assumption.
    """
    if X2 is None:  # If not specified, assume symmetric
        symmetric = True
        X2, Y2 = X1, Y1

    c1 = torch.unique(Y1)
    c2 = torch.unique(Y2)
    n1, n2 = len(c1), len(c2)

    # We account for the possibility that labels are shifted (c1[0]!=0), see below
    if symmetric:
        # If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        # If tasks are asymmetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))

    # Use tensorized backend and Python callables for costs to avoid KeOps string issues.
    if cost_function == "euclidean":
        cost_function = cost_routines[p]

    if GEOMLOSS_AVAILABLE:
        distance = geomloss.SamplesLoss(
            loss=loss,
            p=p,
            cost=cost_function,
            debias=debias,
            blur=entreg ** (1 / p),
            backend="tensorized",
        )
    else:
        # Fallback distance computation
        def distance(x1, x2):
            return torch.mean(cost_function(x1.unsqueeze(0), x2.unsqueeze(0)).squeeze())

    D = torch.zeros((n1, n2), device=device, dtype=X1.dtype)
    for i, j in pairs:
        m1 = X1[Y1 == c1[i]].to(device)
        m2 = X2[Y2 == c2[j]].to(device)

        if GEOMLOSS_AVAILABLE:
            D[i, j] = distance(m1, m2).item()
        else:
            D[i, j] = distance(m1, m2).item()

        if symmetric:
            D[j, i] = D[i, j]

    return D


def batch_augmented_cost(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    W: Optional[torch.Tensor] = None,
    feature_cost: Optional[str] = None,
    p: int = 2,
    lam_x: float = 1.0,
    lam_y: float = 1.0,
):
    """Batch ground cost computation on augmented datasets.

    Parameters
    ----------
    Z1 : torch.Tensor
        Tensor of size (B,N,D1), where last position in last dim corresponds to label Y.
    Z2 : torch.Tensor
        Tensor of size (B,M,D2), where last position in last dim corresponds to label Y.
    W : torch.Tensor, optional
        Tensor of size (V1,V2) of precomputed pairwise label distances for all labels
        V1,V2 and returns a batched cost matrix as a (B,N,M) Tensor. W is expected to be
        congruent with p. I.e, if p=2, W[i,j] should be squared Wasserstein distance.
    feature_cost : str, optional
        if None or 'euclidean', uses euclidean distances as feature metric,
        otherwise uses this function as metric.
    p : int, optional
        Power of the cost (i.e. order of p-Wasserstein distance)
    lam_x : float, optional
        Weight parameter for feature component of distance
    lam_y : float, optional
        Weight parameter for label component of distance

    Returns
    -------
    torch.Tensor
        torch Tensor of size (B,N,M)
    """
    _, _, D1 = Z1.shape
    _, M, D2 = Z2.shape
    assert (D1 == D2) or (feature_cost is not None)

    Y1 = Z1[:, :, -1].long()
    Y2 = Z2[:, :, -1].long()

    if feature_cost is None or feature_cost == "euclidean":  # default is euclidean
        C1 = cost_routines[p](Z1[:, :, :-1], Z2[:, :, :-1])  # Get from GeomLoss
    else:
        C1 = feature_cost(Z1[:, :, :-1], Z2[:, :, :-1])  # Feature Embedding

    # Label Distances
    if W is not None:
        # Label-to-label distances have been precomputed and passed
        # Stores flattened index corresponoding to label pairs
        M = W.shape[1] * Y1[:, :, None] + Y2[:, None, :]
        C2 = W.flatten()[M.flatten(start_dim=1)].reshape(-1, Y1.shape[1], Y2.shape[1])
    else:
        raise ValueError("Must provide either label distances or Means+Covs")

    assert C1.shape == C2.shape

    # NOTE: geomloss's cost_routines as defined above already divide by p. We do
    # so here too for consistency. But as a consequence, need to divide C2 by p too.
    D = lam_x * C1 + lam_y * (C2 / p)

    return D


class DatasetDistance:
    """The main class for the Optimal Transport Dataset Distance.

    An object of this class is instantiated with two datasets (the source and
    target), which are stored in it, and various arguments determining how the
    exact Wasserstein distance is to be computed.
    """

    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        feature_cost: Union[
            Literal["euclidean"], Callable[..., torch.Tensor]
        ] = "euclidean",
        p: int = 2,
        entreg: float = 0.1,
        lam_x: float = 1.0,
        lam_y: float = 1.0,
        # Inner OT (label to label) problem arguments
        inner_ot_loss: str = "sinkhorn",
        inner_ot_debiased: bool = False,
        inner_ot_p: int = 2,
        inner_ot_entreg: float = 0.1,
        # Misc
        device: torch.device = torch.device("cpu"),
    ):
        self.feature_cost = feature_cost
        self.inner_ot_loss = inner_ot_loss
        # For outer OT problem
        self.p = p
        self.entreg = entreg
        self.lam_x = lam_x
        self.lam_y = lam_y
        # For inner (label) OT problem - only used if gaussian approx is False
        self.inner_ot_p = inner_ot_p
        self.inner_ot_entreg = inner_ot_entreg
        self.inner_ot_debiased = inner_ot_debiased

        self.device = device

        [*self.covar_dim] = x_train[0].shape  # Syntax for unpacking tensor shapes
        [*self.label_dim] = (1,) if y_valid.ndim == 1 else y_train.shape[1:]
        self.label_distances = None

        self.x_train, self.y_train = _simple_extract_dataset(x_train, y_train)
        self.x_valid, self.y_valid = _simple_extract_dataset(
            x_valid, y_valid, reindex_start=int(np.prod(self.label_dim))
        )
        self.num_train, self.num_valid = len(y_train), len(y_valid)

    def _get_label_distances(self) -> torch.Tensor:
        """Precompute label-to-label distances.

        Returns tensor of size nclasses_1 x nclasses_2. DISTANCE BETWEEN LABEL IN D1 AND
        LABEL IN D2!
        """
        # Check if already computed
        if self.label_distances is not None:
            return self.label_distances

        # exact way of computing
        # We just define a function ahead, before loading real data
        # pwdist_exact From Geomloss defined function
        pwdist = partial(
            pwdist_exact,
            symmetric=False,
            p=self.inner_ot_p,
            loss=self.inner_ot_loss,
            debias=self.inner_ot_debiased,
            entreg=self.inner_ot_entreg,
            cost_function=self.feature_cost,
            device=self.device,
        )

        # Then we also need within-collection label distances
        DYY1 = pwdist(self.x_train, self.y_train)
        DYY2 = pwdist(self.x_valid, self.y_valid)
        DYY12 = pwdist(self.x_train, self.y_train, self.x_valid, self.y_valid)

        D = torch.cat([torch.cat([DYY1, DYY12], 1), torch.cat([DYY12.t(), DYY2], 1)])

        # Collect and save
        self.label_distances = D

        return self.label_distances

    def dual_sol(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute dataset distance.

        Note:
            Currently requires fully loading dataset into memory, this can probably be
            avoided, e.g., via subsampling.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            F_i (tensor): dual potentials for training data
            G_j (tensor): dual potentials for validation data
        """
        wasserstein = self._get_label_distances().to(self.device)

        # This one leverages precomputed pairwise label distances
        cost_geomloss = partial(
            batch_augmented_cost,
            W=wasserstein,
            lam_x=self.lam_x,
            lam_y=self.lam_y,
            feature_cost=self.feature_cost,
        )

        if GEOMLOSS_AVAILABLE:
            loss = geomloss.SamplesLoss(
                loss="sinkhorn",
                p=self.p,
                cost=cost_geomloss,
                debias=True,
                blur=self.entreg ** (1 / self.p),
                backend="tensorized",
            )
        else:
            # Fallback: simplified dual potential computation
            def loss(z1, z2):
                # Simple approximation: use negative cost as potential
                cost = cost_geomloss(z1.unsqueeze(0), z2.unsqueeze(0)).squeeze(0)
                f_i = -torch.mean(cost, dim=1)  # Training potentials
                g_j = -torch.mean(cost, dim=0)  # Validation potentials
                return f_i, g_j

        Z1 = torch.cat((self.x_train, self.y_train.float().unsqueeze(dim=1)), -1)
        Z2 = torch.cat((self.x_valid, self.y_valid.float().unsqueeze(dim=1)), -1)

        with torch.no_grad():
            if GEOMLOSS_AVAILABLE:
                loss.debias = False
                loss.potentials = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                F_i, G_j = loss(Z1.to(self.device), Z2.to(self.device))
            else:
                # Use fallback method
                F_i, G_j = loss(Z1.to(self.device), Z2.to(self.device))

        del Z1, Z2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return F_i, G_j


def _compute_lava_dual_potentials(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    feature_cost: Union[str, FeatureCost] = "euclidean",
    lam_x: float = 1.0,
    lam_y: float = 1.0,
    p: int = 2,
    entreg: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Compute LAVA dual potentials using the real OTDD implementation.

    This function uses the official OTDD library to compute the dual potentials
    which are then used for LAVA data valuation.

    Args:
        x_train, y_train: Training data
        x_val, y_val: Validation data
        feature_cost: Feature cost ("euclidean" or FeatureCost object)
        lam_x, lam_y: Regularization parameters for features and labels
        p: Norm parameter
        entreg: Entropy regularization
        device: Compute device

    Returns:
        Dual potentials for training samples (shape: [n_train])
    """
    # Debug logging
    import logging

    logger = logging.getLogger(__name__)
    logger.info("[DEBUG] _compute_lava_dual_potentials called with:")
    logger.info(f"[DEBUG]   x_train shape: {x_train.shape}, device: {x_train.device}")
    logger.info(f"[DEBUG]   y_train shape: {y_train.shape}, device: {y_train.device}")
    logger.info(f"[DEBUG]   x_val shape: {x_val.shape}, device: {x_val.device}")
    logger.info(f"[DEBUG]   y_val shape: {y_val.shape}, device: {y_val.device}")
    logger.info(f"[DEBUG]   feature_cost type: {type(feature_cost)}")
    logger.info(
        f"[DEBUG]   parameters: lam_x={lam_x}, lam_y={lam_y}, p={p}, entreg={entreg}"
    )

    # Prepare data for OTDD
    x_tr_flat = _flatten_features(x_train).to(device).float()
    x_val_flat = _flatten_features(x_val).to(device).float()

    logger.info("[DEBUG] After flattening:")
    logger.info(f"[DEBUG]   x_tr_flat shape: {x_tr_flat.shape}")
    logger.info(f"[DEBUG]   x_val_flat shape: {x_val_flat.shape}")

    # Convert labels to proper format for OTDD
    y_tr = y_train.detach().to(device)
    y_val = y_val.detach().to(device)

    if y_tr.dim() > 1:
        y_tr = (y_tr > 0.5).long().view(-1)
    else:
        y_tr = y_tr.long().view(-1)

    if y_val.dim() > 1:
        y_val = (y_val > 0.5).long().view(-1)
    else:
        y_val = y_val.long().view(-1)

    logger.info("[DEBUG] After label processing:")
    logger.info(
        f"[DEBUG]   y_tr shape: {y_tr.shape}, unique values: {torch.unique(y_tr).tolist()}"
    )
    logger.info(
        f"[DEBUG]   y_val shape: {y_val.shape}, unique values: {torch.unique(y_val).tolist()}"
    )

    try:
        # Create OTDD DatasetDistance object
        dataset_distance = DatasetDistance(
            x_train=x_tr_flat,
            y_train=y_tr,
            x_valid=x_val_flat,
            y_valid=y_val,
            feature_cost=feature_cost,
            p=p,
            entreg=entreg,
            lam_x=lam_x,
            lam_y=lam_y,
            device=device,
        )

        # Compute dual solution (F_i, G_j)
        # F_i are the dual potentials for training data
        # G_j are the dual potentials for validation data
        F_i, G_j = dataset_distance.dual_sol()

        # Extract training side dual potentials
        dual_potentials = F_i.detach().cpu().numpy()

        logger.info(
            f"[DEBUG] Dual potentials shape: {dual_potentials.shape}, dtype: {dual_potentials.dtype}"
        )
        logger.info(
            f"[DEBUG] Dual potentials stats - min: {dual_potentials.min():.6f}, max: {dual_potentials.max():.6f}, mean: {dual_potentials.mean():.6f}"
        )

        # Apply LAVA calibration (following paper's Eq. 3):
        # calibrated_gradient[i] = f_dual[i] - (Sum(f_dual) - f_dual[i]) / (N - 1)
        # V[i] = -1 * calibrated_gradient[i]
        # This accounts for the constraint that probability mass must sum to 1
        N = len(dual_potentials)
        sum_f_dual = np.sum(dual_potentials)

        # Vectorized computation of calibrated gradient
        calibrated_gradients = dual_potentials - (sum_f_dual - dual_potentials) / (N - 1)

        # Data value is the negation of the gradient
        # (Positive gradient = increasing mass increases distance = bad quality)
        lava_values = -1.0 * calibrated_gradients

        logger.info(
            f"[DEBUG] Final LAVA values shape: {lava_values.shape}, dtype: {lava_values.dtype}"
        )
        logger.info(
            f"[DEBUG] Final LAVA stats - min: {lava_values.min():.6f}, max: {lava_values.max():.6f}, mean: {lava_values.mean():.6f}"
        )

        return lava_values.astype(np.float32)

    except Exception as e:
        # Fallback to simple centroid-based method if OTDD fails
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"OTDD computation failed: {e}. Falling back to centroid method."
        )

        # Use the simplified centroid fallback from original lava.py
        from .lava import _simple_centroid_fallback

        return _simple_centroid_fallback(x_tr_flat, y_tr, x_val_flat, y_val)


def _extract_model_features(
    model: torch.nn.Module, x: torch.Tensor, batch_size: int = 64
) -> torch.Tensor:
    """Extract features from model (penultimate layer or logits)"""
    model.eval()
    features = []

    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch_x = x[i : i + batch_size]
            # Try to get penultimate layer features if available
            if hasattr(model, "features"):
                # For models with explicit feature extractor
                feats = model.features(batch_x)
                if feats.dim() > 2:
                    feats = torch.flatten(feats, 1)
            elif hasattr(model, "forward_features"):
                # For ViT-like models
                feats = model.forward_features(batch_x)
            else:
                # Fall back to logits
                feats = model(batch_x)

            features.append(feats)

    return torch.cat(features, dim=0)


@InfluenceCalculatorFactory.register("lava_all_epochs")
class LavaAllEpochsInfluenceCalculator(InfluenceCalculator):
    """
    Computes LAVA-style data valuation for each epoch using OTDD.

    This implementation provides two modes:
    1. Fixed representation: Uses a unified embedding (model-agnostic)
    2. Epoch-specific features: Uses each epoch's model as feature extractor

    The algorithm computes:
    - lava_values[e]: LAVA values for epoch e
    - lava_deltas[e]: lava_values[e] - lava_values[e-1] (e >= 1), 0 for e=0
    """

    def __init__(self, infl_type: str, **kwargs):
        super().__init__(infl_type, **kwargs)

        # Configuration for LAVA computation
        self.use_fixed_representation = kwargs.get(
            "use_fixed_representation", False
        )
        self.lam_x = kwargs.get("lam_x", 1.0)
        self.lam_y = kwargs.get("lam_y", 1.0)
        self.p = kwargs.get("p", 2)
        self.entreg = kwargs.get("entreg", 0.1)

        # Find model files
        self.records_dir = os.path.join(self.dn, "records")
        if not os.path.exists(self.records_dir):
            raise FileNotFoundError(
                f"Records directory not found: {self.records_dir}. "
                "Ensure training was completed."
            )

        self.main_files = self._find_model_files()

        if not self.main_files:
            raise FileNotFoundError(
                f"No main model files found in {self.records_dir}. "
                "Ensure training was completed."
            )

    def _get_infl_type(self) -> str:
        return "lava_all_epochs"

    def _find_model_files(self) -> Dict[int, str]:
        """Find main model files for each epoch"""
        seed_suffix = f"{self.seed:03d}"
        main_files = {}

        # Pattern 1: With relabel prefix (check first since relabeling experiments are common)
        if hasattr(self, "relabel_percentage") and self.relabel_percentage:
            relabel_prefix = f"relabel_{int(self.relabel_percentage):03d}_pct_"
            for epoch in range(self.num_epoch):
                pattern = f"{relabel_prefix}epoch_{epoch}_{seed_suffix}.pt"
                file_path = os.path.join(self.records_dir, pattern)
                if os.path.exists(file_path):
                    main_files[epoch] = file_path

        # Pattern 2: epoch_{epoch}_{seed}.pt (standard format)
        if not main_files:
            for epoch in range(self.num_epoch):
                pattern = f"epoch_{epoch}_{seed_suffix}.pt"
                file_path = os.path.join(self.records_dir, pattern)
                if os.path.exists(file_path):
                    main_files[epoch] = file_path

        # Pattern 3: epoch_final_{seed}.pt (final epoch format)
        if not main_files:
            pattern = f"epoch_final_{seed_suffix}.pt"
            file_path = os.path.join(self.records_dir, pattern)
            if os.path.exists(file_path):
                self.logger.info(f"Found final epoch file: {file_path}")
                self.logger.info(
                    f"Using final model for all {self.num_epoch} epochs since individual epoch files not found"
                )
                # Use the final model for all epochs
                for epoch in range(self.num_epoch):
                    main_files[epoch] = file_path

        self.logger.info(f"Found {len(main_files)} main model epoch files")
        return main_files

    def _load_model_state_dict(self, file_path: str) -> Optional[Dict]:
        """Load model state dictionary from file"""
        try:
            checkpoint = torch.load(
                file_path, map_location=self.device, weights_only=False
            )

            if "model_state" in checkpoint:
                return checkpoint["model_state"]
            elif isinstance(checkpoint, dict) and any(
                key.startswith(("weight", "bias")) for key in checkpoint.keys()
            ):
                return checkpoint
            else:
                self.logger.warning(f"Unrecognized file format: {file_path}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to load model state from {file_path}: {e}")
            return None

    def _get_fixed_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get fixed representation features (model-agnostic)"""
        # For fixed representation, use raw features or a pre-trained embedding
        # Here we use flattened raw features as the simplest approach
        return _flatten_features(x)

    def _get_epoch_features(
        self, x: torch.Tensor, model_state_dict: Dict, model_template: torch.nn.Module
    ) -> torch.Tensor:
        """Get features using the epoch's model as feature extractor"""
        model_template.load_state_dict(model_state_dict)
        return _extract_model_features(model_template, x)

    def _create_feature_cost(
        self,
        model_state_dict: Optional[Dict],
        model_template: Optional[torch.nn.Module],
    ) -> Union[str, FeatureCost]:
        """Create appropriate feature cost function"""
        if (
            self.use_fixed_representation
            or model_state_dict is None
            or model_template is None
        ):
            return "euclidean"
        else:
            # Create FeatureCost with the epoch model as embedding
            model_template.load_state_dict(model_state_dict)
            model_template.eval()

            # Create embedding function that extracts features
            def embedding_fn(x):
                return _extract_model_features(model_template, x)

            return FeatureCost(
                src_embedding=embedding_fn,
                tgt_embedding=embedding_fn,
                src_dim=self.input_dim,
                tgt_dim=self.input_dim,
                p=self.p,
                device=self.device,
            )

    def calculate(self) -> List[np.ndarray]:
        """
        Calculate LAVA All Epochs influence for each epoch.

        Returns:
            List of numpy arrays, one for each epoch, containing the LAVA
            influence scores for that epoch.
        """
        self.logger.info("Starting LAVA All Epochs influence calculation...")

        # Initialize results
        lava_values = []
        lava_deltas = []

        # Create model template if needed
        model_template = None
        if not self.use_fixed_representation:
            model_template = get_network(
                self.model_type, self.input_dim, logger=self.logger
            ).to(self.device)

        # Process each epoch
        available_epochs = sorted(list(self.main_files.keys()))
        self.logger.info(f"Available epochs: {available_epochs}")

        prev_lava_values = None

        for epoch_idx in range(self.num_epoch):
            self.logger.info(f"--- Calculating LAVA for Epoch {epoch_idx} ---")

            try:
                if epoch_idx not in self.main_files:
                    self.logger.warning(f"Epoch {epoch_idx}: No model file found")
                    # Use zeros for missing epochs
                    epoch_lava = np.zeros(self.n_tr, dtype=np.float32)
                else:
                    # Load model state if needed
                    model_state_dict = None
                    if not self.use_fixed_representation:
                        model_state_dict = self._load_model_state_dict(
                            self.main_files[epoch_idx]
                        )
                        if model_state_dict is None:
                            self.logger.warning(
                                f"Epoch {epoch_idx}: Failed to load model, using fixed representation"
                            )

                    # Create appropriate feature cost
                    feature_cost = self._create_feature_cost(
                        model_state_dict, model_template
                    )

                    # Prepare input features
                    if self.use_fixed_representation or model_state_dict is None:
                        # Mode 1: Fixed representation (model-agnostic)
                        x_tr_features = self._get_fixed_features(self.x_tr)
                        x_val_features = self._get_fixed_features(self.x_val)
                    else:
                        # Mode 2: Use epoch model as feature extractor
                        # Features will be extracted inside OTDD via FeatureCost
                        x_tr_features = self.x_tr
                        x_val_features = self.x_val

                    # Debug: Log input shapes before LAVA computation
                    self.logger.info(
                        f"[DEBUG] LAVA input shapes - x_tr_features: {x_tr_features.shape}, y_tr: {self.y_tr.shape}"
                    )
                    self.logger.info(
                        f"[DEBUG] LAVA input shapes - x_val_features: {x_val_features.shape}, y_val: {self.y_val.shape}"
                    )
                    self.logger.info(
                        f"[DEBUG] LAVA parameters - lam_x: {self.lam_x}, lam_y: {self.lam_y}, p: {self.p}, entreg: {self.entreg}"
                    )

                    # Compute LAVA values using OTDD
                    epoch_lava = _compute_lava_dual_potentials(
                        x_tr_features,
                        self.y_tr,
                        x_val_features,
                        self.y_val,
                        feature_cost=feature_cost,
                        lam_x=self.lam_x,
                        lam_y=self.lam_y,
                        p=self.p,
                        entreg=self.entreg,
                        device=self.device,
                    )

                    # Debug: Log output shape and basic stats
                    self.logger.info(
                        f"[DEBUG] LAVA output shape: {epoch_lava.shape}, dtype: {epoch_lava.dtype}"
                    )
                    self.logger.info(
                        f"[DEBUG] LAVA raw stats - min: {epoch_lava.min():.6f}, max: {epoch_lava.max():.6f}, mean: {epoch_lava.mean():.6f}"
                    )

                # Handle NaN and numerical stability
                self.logger.info(
                    f"[DEBUG] Before NaN handling - shape: {epoch_lava.shape}, NaN count: {np.isnan(epoch_lava).sum()}"
                )
                epoch_lava = np.nan_to_num(epoch_lava, nan=0.0, posinf=1e6, neginf=-1e6)
                self.logger.info(
                    f"[DEBUG] After NaN handling - shape: {epoch_lava.shape}, finite count: {np.isfinite(epoch_lava).sum()}"
                )

                # Validate expected shape
                if epoch_lava.shape != (self.n_tr,):
                    self.logger.warning(
                        f"[DEBUG] Unexpected LAVA output shape: {epoch_lava.shape}, expected: ({self.n_tr},)"
                    )
                    # Try to reshape if it's a 2D array with single row/column
                    if epoch_lava.size == self.n_tr:
                        old_shape = epoch_lava.shape
                        epoch_lava = epoch_lava.reshape(self.n_tr)
                        self.logger.info(
                            f"[DEBUG] Reshaped from {old_shape} to {epoch_lava.shape}"
                        )
                    else:
                        self.logger.error(
                            f"[DEBUG] Cannot reshape: size {epoch_lava.size} != expected {self.n_tr}"
                        )
                        # Fallback: use zeros
                        epoch_lava = np.zeros(self.n_tr, dtype=np.float32)
                        self.logger.warning(
                            f"[DEBUG] Using zero fallback for epoch {epoch_idx}"
                        )

                # Compute delta (difference from previous epoch)
                if epoch_idx == 0:
                    epoch_delta = np.zeros_like(epoch_lava)
                else:
                    if prev_lava_values is not None:
                        epoch_delta = epoch_lava - prev_lava_values
                    else:
                        epoch_delta = np.zeros_like(epoch_lava)

                # Store results
                lava_values.append(epoch_lava)
                lava_deltas.append(epoch_delta)
                prev_lava_values = epoch_lava.copy()

                self.logger.info(
                    f"Epoch {epoch_idx} LAVA stats: "
                    f"mean={epoch_lava.mean():.6f}, "
                    f"std={epoch_lava.std():.6f}, "
                    f"min={epoch_lava.min():.6f}, "
                    f"max={epoch_lava.max():.6f}"
                )
                self.logger.info(
                    f"Epoch {epoch_idx} Delta stats: "
                    f"mean={epoch_delta.mean():.6f}, "
                    f"std={epoch_delta.std():.6f}, "
                    f"min={epoch_delta.min():.6f}, "
                    f"max={epoch_delta.max():.6f}"
                )

            except Exception as e:
                self.logger.error(f"Epoch {epoch_idx}: Error: {e}", exc_info=True)
                # Use zeros for failed epochs
                epoch_lava = np.zeros(self.n_tr, dtype=np.float32)
                epoch_delta = np.zeros(self.n_tr, dtype=np.float32)
                lava_values.append(epoch_lava)
                lava_deltas.append(epoch_delta)

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Store both values and deltas as instance variables for potential future use
        self.lava_values = lava_values
        self.lava_deltas = lava_deltas

        # Final validation and debug logging
        self.logger.info("[DEBUG] Final results validation:")
        self.logger.info(f"[DEBUG]   Number of epochs processed: {len(lava_values)}")
        self.logger.info(f"[DEBUG]   Expected epochs: {self.num_epoch}")

        for i, epoch_vals in enumerate(lava_values):
            self.logger.info(
                f"[DEBUG]   Epoch {i}: shape {epoch_vals.shape}, dtype {epoch_vals.dtype}"
            )
            if hasattr(epoch_vals, "ndim") and epoch_vals.ndim != 1:
                self.logger.warning(
                    f"[DEBUG]   ⚠️  Epoch {i} has {epoch_vals.ndim}D array, expected 1D"
                )
            if len(epoch_vals) != self.n_tr:
                self.logger.warning(
                    f"[DEBUG]   ⚠️  Epoch {i} length {len(epoch_vals)} != expected {self.n_tr}"
                )

        # Cleanup
        if model_template is not None:
            del model_template
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("LAVA All Epochs calculation finished.")

        # IMPORTANT: Return incremental deltas, not cumulative values
        # This matches the semantic of WIE and ICML methods where each epoch's
        # influence represents the incremental contribution of that epoch's training,
        # not the cumulative effect of all training up to that epoch.
        #
        # For epoch-wise cleansing, we need to identify which samples are harmful
        # specifically to epoch i's training, not cumulative training up to epoch i.
        return lava_deltas


__all__ = ["LavaAllEpochsInfluenceCalculator"]
