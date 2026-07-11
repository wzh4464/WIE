"""Extended query toolkit for WIE (paper Def. 2 / Appendix "Extended Toolkit").

The functional influence estimator ``Q_{-j}^{[t1,t2]}(q) = <q(t2), delta[t2]> -
<q(t1), delta[t1]>`` targets ANY differentiable query ``q(t) = <something> at
theta^[t]``. The reverse-SGD sweep is query-agnostic: it only needs the query
VECTOR ``q`` (a per-parameter list) at the two window endpoints. This module
builds the three query instantiations the paper describes:

- **Test-loss** ``q(t) = grad_theta l(z_test; theta)`` -- the default; built with
  ``compute_gradient`` (not here).
- **Prediction-change** ``q(t) = grad_theta f(x_test; theta)`` where ``f`` is a
  scalar model output (a chosen class logit): :func:`prediction_query`.
- **Feature-saliency** ``q(t) = grad_theta ( d l / d x_test[k] )`` -- one
  component of the input-gradient saliency map, i.e. a single row of the paper's
  vector-valued ``grad_theta(grad_x l)`` Jacobian: :func:`saliency_query`. Run
  per coordinate ``k`` to reconstruct the full vector.

Every builder returns one tensor per ``model.parameters()`` (in order), with
zeros for parameters that do not require grad -- matching ``compute_gradient``'s
convention so the vectors drop straight into the WIE sweep.
"""

from typing import List, Optional

import torch


def _aligned_grads(scalar: torch.Tensor, model: torch.nn.Module) -> List[torch.Tensor]:
    """``grad(scalar)`` w.r.t. every model parameter, zeros where no grad flows."""
    all_params = list(model.parameters())
    grad_params = [p for p in all_params if p.requires_grad]
    grads = torch.autograd.grad(
        scalar, grad_params, retain_graph=False, allow_unused=True
    )
    it = iter(grads)
    out: List[torch.Tensor] = []
    for p in all_params:
        if p.requires_grad:
            g = next(it)
            out.append(g.detach().clone() if g is not None else torch.zeros_like(p))
        else:
            out.append(torch.zeros_like(p))
    model.zero_grad()
    return out


def prediction_query(
    x_test: torch.Tensor,
    model: torch.nn.Module,
    class_index: Optional[int] = None,
) -> List[torch.Tensor]:
    """``q = grad_theta f(x_test; theta)``: gradient of a scalar model output.

    ``f`` is the logit for ``class_index`` at the single test input ``x_test``
    (shape ``[1, ...]``). ``class_index=None`` uses the model's predicted
    (argmax) class; a binary head of shape ``[1, 1]`` uses its single logit.
    Captures the counterfactual shift in raw model output induced by ``z_j``.
    """
    model.zero_grad()
    out = model(x_test)
    out2d = out.reshape(out.shape[0], -1)  # [1, C]  (C == 1 for a binary head)
    if out2d.shape[0] != 1:
        raise ValueError(
            f"prediction_query expects a single test input; got batch {out2d.shape[0]}."
        )
    n_out = out2d.shape[1]
    if class_index is None:
        c = int(out2d.argmax(dim=1).item())
    else:
        c = int(class_index)
        if c < 0 or c >= n_out:
            raise ValueError(
                f"class_index {class_index} out of range for output width {n_out}"
            )
    scalar = out2d[0, c]
    return _aligned_grads(scalar, model)


def saliency_query(
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    coord: int = 0,
) -> List[torch.Tensor]:
    """``q = grad_theta ( d l(x_test, y_test) / d x_test[coord] )``.

    A single component (``coord``) of the input-gradient saliency map -- one row
    of the paper's feature-saliency Jacobian ``grad_theta(grad_x l)``. The input
    gradient is taken with ``create_graph`` so the outer ``grad_theta`` can flow.

    Requires a CONTINUOUS (floating-point) input: the loss must be
    differentiable w.r.t. ``x_test``. Token-id models (e.g. BERT, whose inputs
    are integer token IDs / attention masks) have no differentiable path back to
    the input, so this raises a clear error rather than crashing mid-sweep; use
    ``--query loss`` / ``--query prediction`` for those, or compute saliency
    w.r.t. embeddings (not supported here).
    """
    if not torch.is_floating_point(x_test):
        raise ValueError(
            "saliency query requires continuous (floating-point) inputs to "
            f"differentiate the loss w.r.t. x; got a non-float tensor (dtype "
            f"{x_test.dtype}). Token-id models (e.g. BERT) are unsupported for "
            "the saliency query -- use --query loss or --query prediction."
        )
    x = x_test.detach().clone().requires_grad_(True)
    model.zero_grad()
    loss = loss_fn(model(x), y_test)
    # allow_unused: token models pass the float dtype guard above but cast the
    # input back to long inside forward (e.g. BertClassifier), disconnecting the
    # loss from x. Detect that (grad is None) and raise the intended clear error
    # instead of PyTorch's generic "tensor not used in the graph" RuntimeError.
    gx = torch.autograd.grad(loss, x, create_graph=True, allow_unused=True)[0]
    if gx is None:
        raise ValueError(
            "saliency query: the loss has no differentiable path to the input "
            "(input gradient is None). This happens for token-id models (e.g. "
            "BERT casts inputs to long internally, disconnecting the graph). Use "
            "--query loss or --query prediction for such models."
        )
    gx_flat = gx.reshape(-1)
    n = gx_flat.numel()
    if coord < 0 or coord >= n:
        raise ValueError(f"coord {coord} out of range for input of size {n}")
    scalar = gx_flat[coord]
    return _aligned_grads(scalar, model)
