# Standard library imports
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

# Third-party package imports
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import optimize, sparse
from scipy.sparse import linalg as spla

from .activation_maximisation import (
    LinearNetwork,
    MultilayeredNetwork,
    _targets_for_batches,
    get_neuron_activation,
)
from .utils import pytorch_sparse_to_scipy

# Up to this many nodes, the steady-state and stability functions use exact
# dense computations; above it, sparse ones. The dense eigendecomposition in
# stability_penalty runs every training step and takes ~20 ms at 256 nodes but
# ~240 ms at 800 nodes (CPU).
_DENSE_NODE_LIMIT = 256

# GCaMP6f decay time constant in ms (literature range ~200-500 ms)
GCAMP6F_TAU_MS = 300.0
_SENSOR_SUPPORT_TAU = 6.0  # kernel length, in multiples of tau

_MODEL_KEY_PREFIX = "model_"


def _free_and_sensory_indices(model):
    """
    ``(model.free_indices, sensory indices)``, both on the model's device.
    """
    device = model.all_weights.device
    sensory_idx = model.sensory_indices.detach().to(device=device, dtype=torch.long)
    return model.free_indices, sensory_idx


def _as_sensory_values(model, sensory_values):
    """
    Convert sensory values to a flat float32 tensor on the model's device, and
    check that there is one value per sensory node.
    """
    sensory_values = torch.as_tensor(
        sensory_values, dtype=torch.float32, device=model.all_weights.device
    ).reshape(-1)
    if sensory_values.numel() != model.sensory_indices.numel():
        raise ValueError("sensory_values length must match model.sensory_indices.")
    return sensory_values


def network_step(model, state, sensory_values):
    """
    One time step of the network, with the sensory nodes clamped.

    Uses the model's activation (built-in or custom), followed by
    ``output_rectify`` / ``output_clamp_max`` where set, so that the step
    matches one time step of ``model.forward``.

    Args:
        model (MultilayeredNetwork or LinearNetwork): The network. Must use
            ``sensory_input_mode="replace"``.
        state (array-like or torch.Tensor): State of all nodes, shape
            (n_nodes,).
        sensory_values (array-like or torch.Tensor): Values of the sensory
            nodes, in the order of ``model.sensory_indices``.

    Returns:
        torch.Tensor: The next state, shape (n_nodes,).
    """
    if model.sensory_input_mode != "replace":
        raise ValueError("network_step assumes sensory_input_mode='replace'.")
    return _clamped_step(model, state, sensory_values)


def _clamped_step(model, state, sensory_values):
    """
    ``network_step()`` without the ``sensory_input_mode`` check: the sensory
    nodes are clamped to ``sensory_values`` whatever the mode. The solvers use
    this, for which it is exact in ``"replace"`` mode and an approximation in
    ``"add"`` mode, where the model does not clamp the sensory nodes. Written
    by Claude Fable 5.1.
    """
    device = model.all_weights.device
    state = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(-1, 1)
    sensory_values = _as_sensory_values(model, sensory_values).reshape(-1, 1)

    x = model._activation_step(state, model.effective_weights)
    # forward (replace mode) writes the sensory input before and after the
    # output ops; the ops are elementwise, so clamping once at the end is the
    # same for the free nodes
    x = model._output_ops(x).clone()
    x[model.sensory_indices.to(device), :] = sensory_values
    return x.reshape(-1)


def network_fixed_point(
    model,
    sensory_values,
    initial_state=None,
    max_steps: int = 2000,
    min_steps: int = 20,
    tol: float = 1e-4,
    return_info: bool = False,
):
    """
    Iterate one network step with the sensory nodes clamped to a constant input
    until the state stops changing. Runs without gradients.

    Args:
        model (MultilayeredNetwork or LinearNetwork): The network. Exact for
            ``sensory_input_mode="replace"``; in ``"add"`` mode the model does
            not clamp the sensory nodes, so this is an approximation.
        sensory_values (array-like): Constant sensory input, in the order of
            ``model.sensory_indices``.
        initial_state (array-like, optional): Starting state, e.g. the previous
            fixed point, which makes convergence fast when the parameters have
            changed only slightly. Defaults to zeros.
        max_steps (int, optional): Maximum number of steps. Defaults to 2000.
        min_steps (int, optional): Minimum number of steps. Defaults to 20.
        tol (float, optional): Stop when the largest absolute change of the
            state is below this. Defaults to 1e-4.
        return_info (bool, optional): Also return a dict with ``iterations``,
            ``residual`` and ``converged``. Defaults to False.

    Returns:
        torch.Tensor: The fixed point, or ``(state, info)`` if ``return_info``.
    """
    device = model.all_weights.device
    sensory_values = _as_sensory_values(model, sensory_values)
    _, sensory_idx = _free_and_sensory_indices(model)
    with torch.no_grad():
        if initial_state is None:
            state = torch.zeros(
                model.all_weights.shape[0], dtype=torch.float32, device=device
            )
        else:
            state = (
                torch.as_tensor(initial_state, dtype=torch.float32, device=device)
                .reshape(-1)
                .clone()
            )
        state[sensory_idx] = sensory_values
        residual = torch.as_tensor(float("inf"), dtype=torch.float32, device=device)
        steps = 0
        for steps in range(1, int(max_steps) + 1):
            next_state = _clamped_step(model, state, sensory_values)
            residual = torch.max(torch.abs(next_state - state))
            state = next_state
            if steps >= int(min_steps) and float(residual.detach().cpu()) <= float(tol):
                break
    if return_info:
        return state, {
            "iterations": int(steps),
            "residual": float(residual.detach().cpu()),
            "converged": bool(float(residual.detach().cpu()) <= float(tol)),
        }
    return state


def _check_linearisable(model):
    """
    Raise if the step Jacobian cannot be built from per-node derivatives.
    Written by Claude Fable 5.1.
    """
    if model.divisive_normalization is not None:
        raise NotImplementedError(
            "Jacobian-based analysis (steady states, stability) does not "
            "support divisive_normalization: with it a node's activation "
            "depends on other nodes' states, not only on its own input. Use "
            "network_fixed_point() for the steady state."
        )


def _linearisation_is_state_independent(model):
    """
    True if the step Jacobian does not depend on the state: a
    ``LinearNetwork`` with the built-in (affine) activation. Written by Claude
    Fable 5.1.
    """
    return isinstance(model, LinearNetwork) and model.custom_activation_function is None


def _operating_state(model, state):
    """
    The state at which to linearise: ``state``, or zeros if it is None and the
    Jacobian does not depend on the state. Written by Claude Fable 5.1.
    """
    if state is not None:
        return state
    if _linearisation_is_state_independent(model):
        return torch.zeros(
            model.all_weights.shape[0],
            dtype=torch.float32,
            device=model.all_weights.device,
        )
    raise ValueError(
        "The Jacobian of this model depends on the state; pass the state at "
        "which to linearise, e.g. from network_fixed_point()."
    )


def _step_linearisation(model, state, create_graph: bool = False):
    """
    Per-node derivatives of one network step at ``state``, by autograd. Written
    by Claude Fable 5.1.

    With ``u = W x`` (``W`` the effective weights), one step is
    ``z = post(act(u, x))``, where ``act`` is the model's activation (built-in
    or custom, including the leaky integration) and ``post`` its output
    rectification / clamp (``model._output_ops``). Both act on each node
    separately, so the Jacobian of the step factorises into per-node terms::

        dz_i / dx_j = mask_i * (d_prev_i * delta_ij + d_u_i * W_ij)

    with ``d_u = d act / d u``, ``d_prev = d act / d x`` (the direct
    dependence, through the leak) and ``mask = d post / d y``, all evaluated
    elementwise. For the built-in activations ``d_u = gain / tau``,
    ``d_prev = (tau - 1) / tau``, and ``mask`` is 1 except where the clamp or
    the rectification is active. Every function in this module that needs the
    Jacobian, dense or sparse, builds it from these three vectors, so custom
    activations need no hand-written derivative.

    Args:
        model (MultilayeredNetwork or LinearNetwork): The network. Must not use
            ``divisive_normalization``.
        state (torch.Tensor): State of all nodes, shape (n_nodes,).
        create_graph (bool, optional): Keep the autograd graph from ``d_u``
            and ``d_prev`` back to the model parameters, so that a quantity
            built from the Jacobian (e.g. ``stability_penalty()``) can be
            differentiated. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(d_u, d_prev,
        mask)``, each of shape (n_nodes,).

    Raises:
        NotImplementedError: If the model uses ``divisive_normalization``.
        ValueError: If the activation does not act on each node separately
            (checked with a probe vector), e.g. a custom activation that
            normalises across nodes.

    Note:
        At a kink of the activation, autograd's convention applies: for the
        built-in thresholded relu the derivative is 0 at an input exactly
        equal to ``threshold`` (``relu'(0) = 0``), although the forward pass
        lets that input through. This only matters for nodes whose input sits
        exactly on the threshold, e.g. nodes without input at the zero state.
    """
    _check_linearisable(model)
    device = model.all_weights.device
    state = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(-1, 1)
    # without trainable parameters there is nothing to keep a graph for
    create_graph = create_graph and any(p.requires_grad for p in model.parameters())
    with torch.enable_grad():
        # u is formed from a detached copy of the state and x_prev is its own
        # leaf, so that d/dx_prev is the direct dependence only
        x_prev = state.detach().clone().requires_grad_(True)
        u = torch.sparse.mm(model.effective_weights, state.detach())
        if not u.requires_grad:  # no trainable parameter in the weights
            u.requires_grad_(True)
        y = model.activation_function(u, x_previous=x_prev)

        # a node-by-node activation has a diagonal Jacobian, so J^T r equals
        # diag(J) * r for every r: check this with one probe vector before
        # trusting the elementwise derivatives below
        probe = torch.linspace(0.5, 1.5, y.numel(), device=device).reshape(y.shape)
        probe_grads = torch.autograd.grad(
            y, (u, x_prev), probe, retain_graph=True, allow_unused=True
        )
        grads = torch.autograd.grad(
            y,
            (u, x_prev),
            torch.ones_like(y),
            create_graph=create_graph,
            allow_unused=True,
        )
        d_u, d_prev = (torch.zeros_like(y) if g is None else g for g in grads)
        for g_probe, g in zip(probe_grads, (d_u, d_prev)):
            if g_probe is not None and not torch.allclose(
                g_probe, probe * g.detach(), rtol=1e-4, atol=1e-6
            ):
                raise ValueError(
                    "activation_function must act on each node separately "
                    "(its Jacobian with respect to the input must be diagonal) "
                    "for Jacobian-based analysis."
                )

        # derivative of the output rectification / clamp at this operating
        # point: 0 where it is active, else 1
        y_leaf = y.detach().requires_grad_(True)
        z = model._output_ops(y_leaf)
        if z is y_leaf:
            mask = torch.ones_like(y_leaf)
        else:
            (mask,) = torch.autograd.grad(z, y_leaf, torch.ones_like(z))
    return d_u.reshape(-1), d_prev.reshape(-1), mask.reshape(-1)


def _sparse_update_matrix(model, free_idx, state):
    """
    ``free_update_matrix()`` as a float64 scipy CSR matrix, built from the
    sparse weights for large networks (no gradient). Written by Claude Fable
    5.1.
    """
    d_u, d_prev, mask = (
        t.detach().cpu().numpy().astype(np.float64)
        for t in _step_linearisation(model, state)
    )
    free = free_idx.detach().cpu().numpy()
    weights = pytorch_sparse_to_scipy(model.effective_weights.coalesce())
    weights_ff = weights.astype(np.float64)[free][:, free]
    update = sparse.diags(mask[free] * d_prev[free]) + (
        sparse.diags(mask[free] * d_u[free]) @ weights_ff
    )
    return update.tocsr()


class _FreeBlockAdjoint(torch.autograd.Function):
    """
    Gradient of an equilibrium ``x* = f(x*)`` by the implicit function theorem.

    The forward pass is the identity. The backward pass replaces the gradient
    ``g`` on the free nodes by ``solve_T(g) = (I - M)^-T g``, where ``M`` is the
    Jacobian ``df/dx`` on the free nodes at ``x*``. Applied to one
    differentiable evaluation of ``f`` at the (detached) ``x*``, this gives the
    gradient of ``x*`` with respect to the model parameters. Written by Claude
    Opus 5.5.
    """

    @staticmethod
    def forward(ctx, state, solve_T, free_idx):
        ctx.solve_T = solve_T
        ctx.save_for_backward(free_idx)
        return state.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (free_idx,) = ctx.saved_tensors
        grad = grad_output.clone()
        grad[free_idx] = ctx.solve_T(grad.index_select(0, free_idx)).to(grad)
        return grad, None, None


def _steady_state(
    model, sensory_values, initial_state=None, fp_tol: float = 1e-5, nonlinear=False
):
    """
    Shared implementation of ``linear_network_steady_state()`` and
    ``nonlinear_network_steady_state()``: find the fixed point
    ``x* = network_step(x*)`` without gradients, then attach the gradient by
    the implicit function theorem (``_FreeBlockAdjoint``), with the Jacobian
    from ``_step_linearisation()``.

    For a linear network the fixed-point equation on the free nodes is affine,
    ``x_f = J x_f + c`` with ``c`` one step from zero, so one linear solve
    gives ``x*``: dense (torch) up to ``_DENSE_NODE_LIMIT`` nodes, above that
    a sparse LU (scipy ``splu``) whose factorisation the backward pass
    reuses. For a nonlinear network ``x*`` comes from Newton's method (dense
    only). Written by Claude Fable 5.1.
    """
    _check_linearisable(model)
    free_idx, sensory_idx = _free_and_sensory_indices(model)
    device = model.all_weights.device
    n_nodes = model.all_weights.shape[0]
    sensory_values = _as_sensory_values(model, sensory_values)
    state0 = torch.zeros(n_nodes, dtype=torch.float32, device=device)
    state0[sensory_idx] = sensory_values
    if free_idx.numel() == 0:
        return state0
    n_free = free_idx.numel()

    with torch.no_grad():
        if n_nodes <= _DENSE_NODE_LIMIT:
            eye = torch.eye(n_free, dtype=torch.float64, device=device)
            if nonlinear:
                x_star = _newton_fixed_point_dense(
                    model, sensory_values, free_idx, sensory_idx, initial_state, fp_tol
                )
                update = free_update_matrix(model, x_star).to(torch.float64)
            else:
                update = free_update_matrix(model, state0).to(torch.float64)
                c = _clamped_step(model, state0, sensory_values)[free_idx]
                x_star = state0.clone()
                x_star[free_idx] = torch.linalg.solve(
                    eye - update, c.to(torch.float64)
                ).to(torch.float32)
            system_transposed = (eye - update).T.contiguous()

            def solve_T(g):
                return torch.linalg.solve(system_transposed, g.to(system_transposed))

        else:
            if nonlinear:
                raise NotImplementedError(
                    f"nonlinear_network_steady_state supports at most "
                    f"{_DENSE_NODE_LIMIT} nodes; use network_fixed_point (no "
                    "gradient) for larger networks."
                )
            update = _sparse_update_matrix(model, free_idx, state0)
            lu = spla.splu((sparse.eye(n_free, format="csc") - update).tocsc())
            c = _clamped_step(model, state0, sensory_values)[free_idx]
            x_free = lu.solve(c.cpu().numpy().astype(np.float64))
            if not np.isfinite(x_free).all():
                raise ValueError(
                    "Sparse steady-state solve returned non-finite values."
                )
            x_star = state0.clone()
            x_star[free_idx] = torch.as_tensor(
                x_free, dtype=torch.float32, device=device
            )

            def solve_T(g):
                g = g.detach().cpu().numpy().astype(np.float64)
                return torch.as_tensor(lu.solve(g, trans="T"))

    # x_star has no graph, so the parameters are the only gradient path
    stepped = _clamped_step(model, x_star, sensory_values)
    return _FreeBlockAdjoint.apply(stepped, solve_T, free_idx)


def linear_network_steady_state(model, sensory_values):
    """
    Steady state of a linear network with the sensory nodes clamped, from one
    linear solve. Differentiable with respect to the model parameters.

    The fixed-point equation of a linear network is affine in the free nodes,
    so one solve with the step Jacobian (``free_update_matrix()``) gives the
    steady state, which does not depend on ``tau``. The gradient comes from
    the implicit function theorem, exactly as in
    ``nonlinear_network_steady_state()``.

    Args:
        model (LinearNetwork): The network, with the built-in activation.
        sensory_values (array-like or torch.Tensor): Values of the sensory
            nodes, in the order of ``model.sensory_indices``.

    Returns:
        torch.Tensor: The steady state of all nodes.

    Note:
        Exact only with ``sensory_input_mode="replace"``; in ``"add"`` mode the
        sensory nodes are not clamped and this is an approximation. Networks
        with up to 256 nodes use a dense solve; larger networks use a sparse
        LU solve (scipy ``splu``, on the CPU), whose factorisation the
        backward pass reuses. Raises for a model whose Jacobian depends on
        the state (a nonlinear or custom activation) and for
        ``divisive_normalization``.
    """
    if not _linearisation_is_state_independent(model):
        raise ValueError(
            "linear_network_steady_state needs a LinearNetwork with the built-in "
            "activation; use nonlinear_network_steady_state()."
        )
    return _steady_state(model, sensory_values)


def _newton_fixed_point_dense(
    model, sensory_values, free_idx, sensory_idx, initial_state, fp_tol
):
    """
    Solve for the fixed point of ``network_step()`` with Newton's method
    (``scipy.optimize.root``, method ``"hybr"``), using the Jacobian from
    ``free_update_matrix()``. Raises if the result is not a fixed point to
    within ``fp_tol``. Written by Claude Opus 5.5.
    """
    device = model.all_weights.device
    n_nodes = model.all_weights.shape[0]
    n_free = free_idx.numel()
    eye = np.eye(n_free)

    def full_state(free_values):
        state = torch.zeros(n_nodes, dtype=torch.float32, device=device)
        state[free_idx] = torch.as_tensor(
            free_values, dtype=torch.float32, device=device
        )
        state[sensory_idx] = sensory_values
        return state

    def residual(free_values):
        state = full_state(free_values)
        stepped = _clamped_step(model, state, sensory_values)
        return (stepped[free_idx] - state[free_idx]).cpu().numpy().astype(np.float64)

    def jacobian(free_values):
        update = free_update_matrix(model, full_state(free_values))
        return update.cpu().numpy().astype(np.float64) - eye

    if initial_state is None:
        start = np.zeros(n_free)
    else:
        start = (
            torch.as_tensor(initial_state, dtype=torch.float32)
            .detach()
            .cpu()
            .reshape(-1)[free_idx.cpu()]
            .numpy()
            .astype(np.float64)
        )
    solution = optimize.root(residual, start, jac=jacobian, method="hybr")
    state = full_state(solution.x)
    moved = float((_clamped_step(model, state, sensory_values) - state).abs().max())
    if moved > fp_tol:
        raise RuntimeError(
            f"Newton equilibrium solve did not reach a fixed point of "
            f"network_step: residual {moved:.3e} > fp_tol {fp_tol:.1e} "
            f"(scipy: success={solution.success}, {solution.message!r})"
        )
    return state


def nonlinear_network_steady_state(
    model, sensory_values, initial_state=None, fp_tol: float = 1e-5
):
    """
    Steady state of a nonlinear network with the sensory nodes clamped.
    Differentiable with respect to the model parameters.

    The fixed point of ``network_step()`` is found with Newton's method,
    without gradients, using the Jacobian from ``free_update_matrix()``. The
    gradient is then obtained from the implicit function theorem: one
    differentiable ``network_step()`` at the solution, followed by a linear
    solve with the Jacobian at the solution in the backward pass. This is the
    nonlinear counterpart of ``linear_network_steady_state()``, e.g. for
    ``make_initial_state_fn(..., detach=False, warm_start_kw="initial_state")``.

    Args:
        model (MultilayeredNetwork or LinearNetwork): The network, with
            ``sensory_input_mode="replace"`` and at most 256 nodes. For larger
            networks, use ``network_fixed_point()`` (no gradient).
        sensory_values (array-like): Constant sensory input, in the order of
            ``model.sensory_indices``.
        initial_state (array-like, optional): Starting point for the Newton
            solve, e.g. the previous epoch's steady state. Defaults to None
            (zeros).
        fp_tol (float, optional): Largest accepted change of the state under
            one ``network_step()`` at the solution. Defaults to 1e-5.

    Returns:
        torch.Tensor: The steady state of all nodes.

    Note:
        The Jacobian is built by autograd from the model's activation
        (``_step_linearisation()``), so custom activations need no derivative;
        ``divisive_normalization`` is not supported. The steady state does not
        depend on ``tau``, so the gradient with respect to ``tau`` is zero. The
        backward solve becomes ill-conditioned as the spectral radius of the
        Jacobian approaches 1.
    """
    return _steady_state(
        model,
        sensory_values,
        initial_state=initial_state,
        fp_tol=fp_tol,
        nonlinear=True,
    )


def make_initial_state_fn(
    solver, detach: bool = True, warm_start_kw: Optional[str] = None
):
    """
    Wrap a steady-state solver into a callable for
    ``train_model(initial_state=...)``, which calls it at the start of every
    epoch. The initial state then follows the model as it trains, instead of
    staying at the steady state of the initial model.

    Args:
        solver (callable): ``fn(model, **kwargs) -> state``, e.g.
            ``functools.partial(linear_network_steady_state, sensory_values=u0)``.
        detach (bool, optional): Detach the returned state from the autograd
            graph, so the initial state is treated as a constant in each epoch.
            Set to False to let gradients flow through a differentiable solver
            (``linear_network_steady_state()``,
            ``nonlinear_network_steady_state()``). Defaults to True.
        warm_start_kw (str, optional): Name of the solver's keyword for a
            starting state (``"initial_state"`` for ``network_fixed_point()``
            and ``nonlinear_network_steady_state()``). If given, each call
            starts from the previous epoch's result. Defaults to None.

    Returns:
        callable: ``fn(model) -> state``.
    """
    cache = {"state": None}

    def initial_state_fn(model):
        kwargs = {}
        if warm_start_kw is not None and cache["state"] is not None:
            kwargs[warm_start_kw] = cache["state"]
        state = solver(model, **kwargs)
        cache["state"] = state.detach()
        return cache["state"] if detach else state

    return initial_state_fn


def free_update_matrix(model, state=None, create_graph: Optional[bool] = None):
    """
    Jacobian of one network step, restricted to the free (non-sensory) nodes.

    From the per-node derivatives of ``_step_linearisation()`` it is
    ``diag(mask) (diag(d_prev) + diag(d_u) W)``; for the built-in activations
    that is ``diag((tau - 1) / tau) + (gain / tau) * W``, with the rows of the
    nodes at the output clamp or below the rectification threshold zeroed.
    The network is stable around ``state`` if the spectral radius of this
    matrix is below 1.

    Args:
        model (MultilayeredNetwork or LinearNetwork): The network, with at most
            256 nodes.
        state (torch.Tensor, optional): State at which to linearise. May be
            omitted only when the Jacobian does not depend on the state (a
            ``LinearNetwork`` with the built-in activation). Defaults to None.
        create_graph (bool, optional): Keep the autograd graph back to the
            model parameters, see ``_step_linearisation()``. Defaults to None:
            keep it when gradients are enabled.

    Returns:
        torch.Tensor: Square matrix over the free nodes.

    Note:
        Assumes clamped sensory nodes (``sensory_input_mode="replace"``). Not
        available with ``divisive_normalization``.
    """
    n_nodes = model.all_weights.shape[0]
    if n_nodes > _DENSE_NODE_LIMIT:
        raise ValueError(
            f"free_update_matrix is dense and supports at most "
            f"{_DENSE_NODE_LIMIT} nodes; got {n_nodes}."
        )
    if create_graph is None:
        create_graph = torch.is_grad_enabled()
    W = model.effective_weights.to_dense()
    free_idx, _ = _free_and_sensory_indices(model)
    if free_idx.numel() == 0:
        return torch.zeros((0, 0), dtype=W.dtype, device=W.device)
    d_u, d_prev, mask = _step_linearisation(
        model, _operating_state(model, state), create_graph=create_graph
    )
    update = mask.view(-1, 1) * (torch.diag(d_prev) + d_u.view(-1, 1) * W)
    return update.index_select(0, free_idx).index_select(1, free_idx)


def _gershgorin_spectral_radius_bound(model, free_idx, state):
    """
    Upper bound on the spectral radius of the update matrix ``M`` from the
    Gershgorin circle theorem: ``max_i (|M_ii| + sum_{j != i} |M_ij|)``.
    Computed from the sparse weights and the per-node derivatives of
    ``_step_linearisation()``; differentiable.
    """
    weights = model.effective_weights.coalesce()
    idx = weights.indices()
    post, pre = idx[0], idx[1]
    vals = weights.values()
    n = weights.shape[0]
    device = vals.device

    d_u, d_prev, mask = _step_linearisation(
        model, state, create_graph=torch.is_grad_enabled()
    )
    scale = mask * d_u  # per post-synaptic node

    free_mask = torch.zeros(n, dtype=torch.bool, device=device)
    free_mask[free_idx] = True
    edge_free = free_mask[post] & free_mask[pre]
    post_f = post[edge_free]
    m_vals = scale[post_f] * vals[edge_free]
    diag_edge = post_f == pre[edge_free]

    diag_from_w = torch.zeros(n, dtype=m_vals.dtype, device=device)
    if torch.any(diag_edge):
        diag_from_w = diag_from_w.index_add(0, post_f[diag_edge], m_vals[diag_edge])
    m_diag = mask * d_prev + diag_from_w

    offdiag_abs = torch.zeros(n, dtype=m_vals.dtype, device=device)
    off_edge = ~diag_edge
    if torch.any(off_edge):
        offdiag_abs = offdiag_abs.index_add(
            0, post_f[off_edge], torch.abs(m_vals[off_edge])
        )

    gershgorin = torch.abs(m_diag) + offdiag_abs
    return gershgorin[free_idx].max()


def _spectral_radius_sparse(model, free_idx, state):
    """
    Spectral radius of the update matrix of a large network, via ARPACK.
    """
    update = _sparse_update_matrix(model, free_idx, state)
    if free_idx.numel() <= 2:
        eigenvalues = np.linalg.eigvals(update.toarray())
    else:
        eigenvalues = spla.eigs(update, k=1, which="LM", return_eigenvectors=False)
    return float(np.abs(eigenvalues).max())


def stability_penalty(model, state=None, margin: float = 1e-4):
    """
    Differentiable penalty ``relu(rho - (1 - margin)) ** 2`` on the spectral
    radius ``rho`` of ``free_update_matrix()``. It is zero for a stable network
    and can be added to a ``train_model`` loss (see
    ``make_affine_readout_loss()``).

    Args:
        model (MultilayeredNetwork or LinearNetwork): The network.
        state (torch.Tensor, optional): State at which to linearise; see
            ``free_update_matrix()``. Defaults to None.
        margin (float, optional): Required distance of ``rho`` below 1.
            Defaults to 1e-4.

    Returns:
        torch.Tensor: Scalar penalty.

    Note:
        Up to 256 nodes, ``rho`` is the exact spectral radius. Above that, it is
        the Gershgorin upper bound, which is conservative, so the penalty is
        stricter than the exact one. Not available with
        ``divisive_normalization``.
    """
    free_idx, _ = _free_and_sensory_indices(model)
    if free_idx.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=model.all_weights.device)
    state = _operating_state(model, state)

    if model.all_weights.shape[0] > _DENSE_NODE_LIMIT:
        rho = _gershgorin_spectral_radius_bound(model, free_idx, state)
    else:
        update = free_update_matrix(model, state)
        rho = torch.abs(torch.linalg.eigvals(update)).max()
    return torch.relu(rho - (1.0 - float(margin))) ** 2


def spectral_radius(model, state=None):
    """
    Spectral radius of ``free_update_matrix()``, without gradients. Below 1
    means the network is stable around ``state``.

    Args:
        model (MultilayeredNetwork or LinearNetwork): The network.
        state (torch.Tensor, optional): State at which to linearise; see
            ``free_update_matrix()``. Defaults to None.

    Returns:
        float: The spectral radius (0.0 if there are no free nodes).

    Note:
        Up to 256 nodes this uses a dense eigendecomposition, above that ARPACK
        (``scipy.sparse.linalg.eigs``). Both are exact.
    """
    with torch.no_grad():
        free_idx, _ = _free_and_sensory_indices(model)
        if free_idx.numel() == 0:
            return 0.0
        state = _operating_state(model, state)
        if model.all_weights.shape[0] > _DENSE_NODE_LIMIT:
            return _spectral_radius_sparse(model, free_idx, state)
        update = free_update_matrix(model, state)
        return float(torch.abs(torch.linalg.eigvals(update)).max().detach().cpu())


class ExponentialSensor:
    """
    Calcium indicator model: convolves activity with a causal exponential
    kernel ``exp(-t / tau)``, normalised to sum to 1.

    Fitting the model's activity convolved with this kernel to recorded
    fluorescence avoids deconvolving the data, and keeps the fitted time
    constants those of the neurons rather than of the indicator. Because the
    kernel sums to 1, steady levels pass through unchanged.

    Use ``output_transform()`` in training
    (``train_model(output_transform=sensor.output_transform)``) and ``apply()``
    on numpy traces; both compute the same convolution.

    Args:
        tau_ms (float, optional): Decay time constant in ms. Should not be
            fitted together with the model's own time constants, since the two
            are not separately identifiable. Defaults to ``GCAMP6F_TAU_MS``.
        dt_ms (float, optional): Sampling interval of the traces in ms.
            Defaults to 1.0.
        support_tau (float, optional): Kernel length, in multiples of
            ``tau_ms``. Defaults to 6.
    """

    def __init__(
        self,
        tau_ms: float = GCAMP6F_TAU_MS,
        dt_ms: float = 1.0,
        support_tau: float = _SENSOR_SUPPORT_TAU,
    ):
        if tau_ms <= 0:
            raise ValueError("tau_ms must be positive.")
        if dt_ms <= 0:
            raise ValueError("dt_ms must be positive.")
        n = max(1, int(round(support_tau * tau_ms / dt_ms)))
        kernel = np.exp(-np.arange(n) * float(dt_ms) / float(tau_ms))
        self.kernel = (kernel / kernel.sum()).astype(np.float32)
        self.tau_ms = float(tau_ms)
        self.dt_ms = float(dt_ms)
        # F.conv1d computes a cross-correlation, so flip the kernel
        self._weight = torch.as_tensor(self.kernel[::-1].copy()).view(1, 1, -1)

    def _convolve(self, x: torch.Tensor) -> torch.Tensor:
        """
        Causal convolution along the last axis of a (batch, neurons, T) tensor,
        padding the start with the first value.
        """
        b, n, t = x.shape
        weight = self._weight.to(device=x.device, dtype=x.dtype)
        x = F.pad(x.reshape(b * n, 1, t), (weight.shape[-1] - 1, 0), mode="replicate")
        return F.conv1d(x, weight).reshape(b, n, t)

    def output_transform(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Apply the sensor to model outputs. Differentiable.

        Args:
            outputs (torch.Tensor): Shape (batch, neurons, T).

        Returns:
            torch.Tensor: Same shape as ``outputs``.
        """
        if outputs.dim() != 3:
            raise ValueError("output_transform expects (batch, neurons, T).")
        return self._convolve(outputs)

    def apply(self, trace):
        """
        Apply the sensor to a numpy trace.

        Args:
            trace (array-like): Shape (T,) or (n, T).

        Returns:
            numpy.ndarray: Same shape as ``trace``, float64.
        """
        trace = np.asarray(trace, dtype=np.float64)
        if trace.ndim not in (1, 2):
            raise ValueError("trace must be 1-D (T,) or 2-D (n, T).")
        x = torch.as_tensor(trace).reshape(1, -1, trace.shape[-1])
        return self._convolve(x).reshape(trace.shape).numpy()


def affine_readout_solve(latent, target, ridge: float = 1e-4):
    """
    Least-squares scale and offset mapping ``latent`` to ``target``, with the
    scale constrained to be nonnegative. Differentiable with respect to
    ``latent``.

    Args:
        latent (torch.Tensor): 1-D model output for one channel.
        target (torch.Tensor): 1-D target, same length.
        ridge (float, optional): Small regulariser on the scale, for numerical
            stability. Defaults to 1e-4.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(scale, offset)``.
    """
    lm = latent.mean()
    tm = target.mean()
    lc = latent - lm
    tc = target - tm
    scale = (lc * tc).mean() / (lc.pow(2).mean() + ridge)
    scale = torch.relu(scale)
    offset = tm - scale * lm
    return scale, offset


def _affine_readout_penalties(
    scales,
    latent_stds,
    scale_soft_limit: float = 10.0,
    latent_std_floor: float = 0.02,
):
    """
    Penalties that stop the readout from compensating for a vanishing model
    output: ``mean(log1p(|scale| / scale_soft_limit) ** 2)`` on the readout
    scales, and ``mean(relu(latent_std_floor - std) ** 2)`` on the standard
    deviation of the model output.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(scale_penalty, std_penalty)``.
    """
    scale_penalty = (torch.log1p(torch.abs(scales) / scale_soft_limit) ** 2).mean()
    std_penalty = (torch.relu(latent_std_floor - latent_stds) ** 2).mean()
    return scale_penalty, std_penalty


def affine_readout_frame(
    latent_windows: Mapping,
    target_windows: Mapping,
    layers: Optional[Sequence] = None,
    ridge: float = 1e-4,
    window_indices: Optional[Sequence] = None,
):
    """
    Fit the readout (``affine_readout_solve()``) for each channel of a fitted
    model.

    Args:
        latent_windows (Mapping): ``{layer: array}`` of model outputs.
        target_windows (Mapping): ``{layer: array}`` of targets, same shapes.
        layers (sequence, optional): Channels to fit, in order. Defaults to the
            keys of ``latent_windows``.
        ridge (float, optional): See ``affine_readout_solve()``. Defaults to
            1e-4.
        window_indices (sequence, optional): Rows (first axis) to fit on, e.g.
            only the training windows. Defaults to all rows.

    Returns:
        tuple[pd.DataFrame, dict]: A table with columns ``layer``,
        ``readout_offset``, ``readout_scale``, ``latent_mean``,
        ``target_mean``, ``latent_std`` and ``target_std``, and a dict
        ``{layer: (scale, offset)}``.
    """
    layers = list(latent_windows) if layers is None else list(layers)
    if window_indices is not None:
        rows_idx = list(window_indices)
        latent_windows = {
            layer: np.asarray(latent_windows[layer])[rows_idx] for layer in layers
        }
        target_windows = {
            layer: np.asarray(target_windows[layer])[rows_idx] for layer in layers
        }
    rows = []
    readout_by_layer = {}
    for layer in layers:
        latent = np.asarray(latent_windows[layer], dtype=np.float64).reshape(-1)
        target = np.asarray(target_windows[layer], dtype=np.float64).reshape(-1)
        scale, offset = affine_readout_solve(
            torch.as_tensor(latent, dtype=torch.float32),
            torch.as_tensor(target, dtype=torch.float32),
            ridge=ridge,
        )
        scale = float(scale)
        offset = float(offset)
        readout_by_layer[layer] = (scale, offset)
        rows.append(
            {
                "layer": layer,
                "readout_offset": offset,
                "readout_scale": scale,
                "latent_mean": float(latent.mean()),
                "target_mean": float(target.mean()),
                "latent_std": float(latent.std()),
                "target_std": float(target.std()),
            }
        )
    return pd.DataFrame(rows), readout_by_layer


def make_affine_readout_loss(
    model,
    targets: pd.DataFrame,
    layer_ids: Sequence[int],
    expected_layer_windows: Optional[Mapping] = None,
    ridge: float = 1e-4,
    scale_soft_limit: float = 10.0,
    latent_std_floor: float = 0.02,
    scale_reg_weight: float = 1e-2,
    std_reg_weight: float = 30.0,
    stability_weight: float = 0.0,
    stability_margin: float = 1e-4,
    stability_state_fn: Optional[Callable] = None,
    normalize_by_target_var: bool = False,
) -> Callable:
    """
    Build a loss for ``train_model(activation_loss_fn=...)`` that compares the
    model to the targets up to a scale and offset per channel.

    In every evaluation, each channel's model output is mapped to the target
    with ``affine_readout_solve()`` before computing the mean squared error, so
    the model does not need to match the units of the data. Penalties on the
    readout stop it from compensating for a vanishing model output, and a
    stability penalty can be added. Only single-batch targets (``batch == 0``)
    are supported.

    Args:
        model (MultilayeredNetwork or LinearNetwork): The network being fitted.
        targets (pd.DataFrame): The targets passed to ``train_model``, with
            columns ``batch``, ``neuron_idx``, ``layer`` and ``value``.
        layer_ids (sequence of int): The ``neuron_idx`` of each channel: node
            indices, or group ids when fitting with ``target_node_groups``.
        expected_layer_windows (Mapping, optional): ``{layer_id: array}`` of the
            target windows. If given, the first evaluation checks that each
            channel's targets have the same mean and standard deviation, which
            confirms that the channels are matched up correctly. Defaults to
            None.
        ridge (float, optional): See ``affine_readout_solve()``. Defaults to
            1e-4.
        scale_soft_limit (float, optional): Readout scale above which the
            scale penalty grows quickly. Defaults to 10.0.
        latent_std_floor (float, optional): Standard deviation of the model
            output below which it is penalised. Defaults to 0.02.
        scale_reg_weight (float, optional): Weight of the scale penalty.
            Defaults to 1e-2.
        std_reg_weight (float, optional): Weight of the standard-deviation
            penalty. Defaults to 30.0.
        stability_weight (float, optional): Weight of ``stability_penalty()``.
            Defaults to 0.0 (no stability penalty).
        stability_margin (float, optional): ``margin`` of
            ``stability_penalty()``. Defaults to 1e-4.
        stability_state_fn (callable, optional): ``fn(previous_state) -> state``
            giving the state at which to linearise for the stability penalty.
            It receives its previous result (None on the first call), e.g. to
            warm-start ``network_fixed_point()``. Required unless the Jacobian
            does not depend on the state (a ``LinearNetwork`` with the
            built-in activation). Defaults to None.
        normalize_by_target_var (bool, optional): Weight each channel's error by
            ``mean(target_var) / target_var``, so that channels with small
            amplitude count as much as large ones. Defaults to False.

    Returns:
        callable: ``loss_fn(pred, target)`` returning a scalar tensor.
    """
    layer_ids = [int(i) for i in layer_ids]
    batches = sorted(set(targets["batch"].tolist()))
    if batches != [0]:
        raise ValueError(
            "make_affine_readout_loss supports single-batch targets "
            f"(batch == 0 only); got batches {batches}."
        )
    if stability_weight < 0:
        raise ValueError("stability_weight must be >= 0.")
    if stability_weight > 0:
        # fail here rather than on the first loss call inside train_model
        _check_linearisable(model)
        if stability_state_fn is None and not _linearisation_is_state_independent(
            model
        ):
            raise ValueError(
                "stability_weight > 0 needs the state at which to linearise, "
                "since this model's Jacobian depends on the state. Pass "
                "stability_state_fn=..., e.g. lambda previous: "
                "network_fixed_point(model, u0, initial_state=previous)."
            )

    # the masks pick out each channel's entries in the order train_model
    # flattens the targets
    order = _targets_for_batches(targets, [0])
    flat_neuron_idx = order["neuron_idx"].to_numpy().astype(int)
    masks = [torch.as_tensor(flat_neuron_idx == lid) for lid in layer_ids]
    for lid, mask in zip(layer_ids, masks):
        if not bool(mask.any()):
            raise ValueError(f"targets contain no rows with neuron_idx == {lid}.")

    moments = None
    if expected_layer_windows is not None:
        moments = []
        for lid in layer_ids:
            t = torch.as_tensor(
                np.asarray(expected_layer_windows[lid]).reshape(-1),
                dtype=torch.float32,
            )
            moments.append((float(t.mean()), float(t.std())))
    order_checked = {"done": moments is None}
    warm = {"state": None}

    def loss_fn(pred, target):
        channel_masks = [m.to(pred.device) for m in masks]
        if not order_checked["done"]:
            for mask, (tmean, tstd) in zip(channel_masks, moments):
                tgt = target[mask]
                if (
                    tgt.numel() == 0
                    or abs(float(tgt.mean()) - tmean) > 1e-4
                    or abs(float(tgt.std()) - tstd) > 1e-4
                ):
                    raise ValueError(
                        "Per-channel target grouping does not match "
                        "expected_layer_windows; the replicated flat ordering "
                        "is wrong."
                    )
            order_checked["done"] = True

        mses, scales, latent_stds, target_vars = [], [], [], []
        for mask in channel_masks:
            latent = pred[mask]
            tgt = target[mask]
            scale, offset = affine_readout_solve(latent, tgt, ridge=ridge)
            mses.append(((scale * latent + offset - tgt) ** 2).mean())
            scales.append(scale)
            latent_stds.append(latent.std())
            target_vars.append(tgt.var(unbiased=False))
        if normalize_by_target_var:
            tvars = torch.stack(target_vars).clamp_min(1e-12)
            weights = tvars.mean() / tvars
            loss = (weights * torch.stack(mses)).mean()
        else:
            loss = torch.stack(mses).mean()
        scale_penalty, std_penalty = _affine_readout_penalties(
            torch.stack(scales),
            torch.stack(latent_stds),
            scale_soft_limit=scale_soft_limit,
            latent_std_floor=latent_std_floor,
        )
        loss = (
            loss
            + scale_reg_weight * scale_penalty.to(loss.device)
            + std_reg_weight * std_penalty.to(loss.device)
        )
        if stability_weight > 0:
            state = None
            if stability_state_fn is not None:
                state = stability_state_fn(warm["state"])
                warm["state"] = state
            penalty = stability_penalty(model, state, margin=stability_margin)
            loss = loss.to(penalty.device) + stability_weight * penalty
        return loss

    return loss_fn


def _window_starts(transition_idx, window_steps: int):
    """
    Check a set of windows: ``window_steps >= 1`` and no two windows overlap.
    The overlap check sorts the starts, so ``transition_idx`` can be in any
    order. Written by Claude Fable 5.1.

    Returns:
        tuple[np.ndarray, int]: ``(starts, n_steps)``, with ``starts`` in the
        given order.
    """
    starts = np.asarray(transition_idx, dtype=np.int64).reshape(-1)
    n_steps = int(window_steps)
    if n_steps < 1:
        raise ValueError("window_steps must be >= 1.")
    if starts.size > 1 and np.any(np.diff(np.sort(starts)) < n_steps):
        raise ValueError("Windows overlap; reduce window_steps.")
    return starts, n_steps


def score_mask_for_transition_windows(
    n_samples: int, transition_idx, window_steps: int
) -> np.ndarray:
    """
    Boolean mask of the samples in the windows that follow each transition.

    Args:
        n_samples (int): Length of the trace.
        transition_idx (array-like of int): First sample of each window, in
            any order.
        window_steps (int): Length of each window.

    Returns:
        np.ndarray: Boolean mask of length ``n_samples``.

    Raises:
        ValueError: If a window extends past the end of the trace, or windows
            overlap.
    """
    starts, n_steps = _window_starts(transition_idx, window_steps)
    n_samples = int(n_samples)
    if np.any(starts + n_steps > n_samples):
        raise ValueError("At least one scoring window exceeds the trace length.")
    mask = np.zeros(n_samples, dtype=bool)
    for start in starts:
        mask[int(start) : int(start) + n_steps] = True
    return mask


def build_raw_window_targets(
    target_traces: Mapping,
    transition_idx,
    window_steps: int,
    node_index: Mapping,
    layers: Optional[Sequence] = None,
) -> pd.DataFrame:
    """
    Targets for ``train_model`` from the windows that follow each transition,
    with one row per channel and sample.

    Args:
        target_traces (Mapping): ``{layer: 1-D trace}``.
        transition_idx (array-like of int): First sample of each window, in
            any order.
        window_steps (int): Length of each window.
        node_index (Mapping): ``{layer: neuron_idx}``: the node index, or the
            group id when fitting with ``target_node_groups``.
        layers (sequence, optional): Channels to include, in order. Defaults to
            the keys of ``target_traces``.

    Returns:
        pd.DataFrame: Columns ``batch`` (always 0), ``neuron_idx``, ``layer``
        (the time step) and ``value``.

    Raises:
        ValueError: If a window extends past the end of a trace, or windows
            overlap (overlapping windows would repeat samples, which would
            then count more than once in the loss).
    """
    layers = list(target_traces) if layers is None else list(layers)
    starts, n_steps = _window_starts(transition_idx, window_steps)
    steps = (starts[:, None] + np.arange(n_steps)).reshape(-1)
    windows = extract_transition_windows(target_traces, starts, n_steps, layers=layers)
    frames = [
        pd.DataFrame(
            {
                "batch": 0,
                "neuron_idx": int(node_index[layer]),
                "layer": steps,
                "value": windows[layer].reshape(-1).astype(np.float64),
            }
        )
        for layer in layers
    ]
    if not frames:
        return pd.DataFrame(columns=["batch", "neuron_idx", "layer", "value"])
    return pd.concat(frames, ignore_index=True)


def extract_transition_windows(
    traces,
    transition_idx,
    window_steps: int,
    layers: Optional[Sequence] = None,
):
    """
    Cut the windows that follow each transition out of one or more traces.

    Args:
        traces (array-like, Mapping or pd.DataFrame): A single 1-D trace, or
            several by layer.
        transition_idx (array-like of int): First sample of each window.
        window_steps (int): Length of each window.
        layers (sequence, optional): Layers to extract, for Mapping or
            DataFrame input. Defaults to all.

    Returns:
        np.ndarray or dict: Array of shape (n_transitions, window_steps) for a
        single trace, otherwise ``{layer: array}``.
    """
    n_steps = int(window_steps)
    idx = np.asarray(transition_idx, dtype=np.int64)

    def _one(trace):
        trace = np.asarray(trace)
        if np.any(idx + n_steps > trace.size):
            raise ValueError("A window exceeds the trace length.")
        return np.stack(
            [trace[int(start) : int(start) + n_steps] for start in idx]
        ).astype(np.float32)

    if isinstance(traces, pd.DataFrame):
        layers = list(traces.columns) if layers is None else list(layers)
        return {layer: _one(traces[layer].to_numpy()) for layer in layers}
    if isinstance(traces, Mapping):
        layers = list(traces) if layers is None else list(layers)
        return {layer: _one(traces[layer]) for layer in layers}
    return _one(traces)


def r2(y, y_pred) -> float:
    """
    Coefficient of determination (R²) of ``y_pred`` for ``y``. NaN if ``y`` is
    constant, where R² is undefined.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _model_arrays(model) -> dict:
    """
    Everything ``rebuild_network()`` needs, as numpy arrays with ``model_`` keys.
    """

    def np32(tensor):
        return tensor.detach().cpu().numpy().astype(np.float32)

    def optional_float(value):
        # npz cannot store None, so store inf
        return np.asarray(np.inf if value is None else float(value), dtype=np.float32)

    weights = model.effective_weights.coalesce()
    n_nodes = model.all_weights.shape[0]
    out = {
        "model_class": np.asarray(type(model).__name__),
        "model_weights_indices": weights.indices().detach().cpu().numpy(),
        "model_weights_values": np32(weights.values()),
        "model_weights_shape": np.asarray(list(weights.shape), dtype=np.int64),
        "model_node_bias": np32(model.node_parameter("bias")),
        "model_node_tau": np32(model.node_parameter("tau")),
        "model_node_slope": np32(model.node_parameter("slope")),
        "model_sensory_indices": model.sensory_indices.detach()
        .cpu()
        .numpy()
        .astype(np.int64),
        "model_num_layers": np.asarray(int(model.num_layers), dtype=np.int64),
        "model_threshold": np.asarray(float(model.threshold), dtype=np.float32),
        "model_tanh_steepness": np.asarray(
            float(model.tanh_steepness), dtype=np.float32
        ),
        "model_sensory_input_mode": np.asarray(str(model.sensory_input_mode)),
        "model_tau_max": optional_float(model.tau_max),
        "model_output_clamp_max": optional_float(
            getattr(model, "output_clamp_max", None)
        ),
        "model_output_rectify": np.asarray(
            bool(getattr(model, "output_rectify", False))
        ),
    }
    if model.idx_to_group is not None:
        out["model_node_group"] = np.asarray(
            [str(model.idx_to_group[i]) for i in range(n_nodes)]
        )
    return out


def save_fit(path, model=None, arrays: Optional[Mapping] = None) -> None:
    """
    Save a fitted model, and any other arrays, to a compressed ``.npz`` file.

    The model is stored with its effective weights and per-node parameters, so
    ``rebuild_network()`` can recreate it. Custom activation functions cannot be
    saved; pass them again to ``rebuild_network()``.

    Args:
        path (str or Path): Output file.
        model (MultilayeredNetwork or LinearNetwork, optional): The fitted
            network. Defaults to None.
        arrays (Mapping, optional): Other arrays to save, e.g. traces or readout
            parameters. Keys must not start with ``"model_"``. Defaults to None.
    """
    data = {}
    if model is not None:
        data.update(_model_arrays(model))
    if arrays:
        for key, value in arrays.items():
            if str(key).startswith(_MODEL_KEY_PREFIX):
                raise ValueError(
                    f"array key {key!r} collides with the reserved "
                    f"'{_MODEL_KEY_PREFIX}' prefix."
                )
            data[str(key)] = np.asarray(value)
    np.savez_compressed(path, **data)


def load_fit(path) -> dict:
    """
    Load a file written by ``save_fit()`` into a dict of arrays.
    """
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def rebuild_network(
    fit: Union[dict, str, Path],
    activation_function: Optional[Callable] = None,
    device=None,
):
    """
    Recreate a fitted network saved with ``save_fit()``. The rebuilt network
    gives the same output as the saved one; every node becomes its own
    parameter group.

    Args:
        fit (dict, str or Path): The output of ``load_fit()``, or the path to
            the file.
        activation_function (callable, optional): The custom activation used in
            the fit, if any. Defaults to None.
        device (torch.device, optional): Device for the network. Defaults to
            None.

    Returns:
        MultilayeredNetwork or LinearNetwork: The rebuilt network.
    """
    if isinstance(fit, (str, Path)):
        fit = load_fit(fit)
    if "model_weights_values" not in fit:
        raise ValueError("fit contains no model section (was save_fit given a model?).")

    shape = tuple(int(s) for s in np.asarray(fit["model_weights_shape"]))
    idx = np.asarray(fit["model_weights_indices"])
    weights = sparse.coo_matrix(
        (np.asarray(fit["model_weights_values"]), (idx[0], idx[1])), shape=shape
    ).tocsr()
    node_names = [f"n{i}" for i in range(shape[0])]

    def per_node(key):
        return {
            name: float(value) for name, value in zip(node_names, np.asarray(fit[key]))
        }

    def optional_float(key):
        value = float(fit[key])
        return None if np.isinf(value) else value

    kwargs = dict(
        all_weights=weights,
        sensory_indices=np.asarray(fit["model_sensory_indices"]).tolist(),
        num_layers=int(fit["model_num_layers"]),
        threshold=float(fit["model_threshold"]),
        tanh_steepness=float(fit["model_tanh_steepness"]),
        idx_to_group=dict(enumerate(node_names)),
        bias_dict=per_node("model_node_bias"),
        bias_transform="identity",
        slope_dict=per_node("model_node_slope"),
        tau_dict=per_node("model_node_tau"),
        tau_max=optional_float("model_tau_max"),
        sensory_input_mode=str(fit["model_sensory_input_mode"]),
        activation_function=activation_function,
        device=device,
    )
    if str(fit["model_class"]) == "LinearNetwork":
        return LinearNetwork(**kwargs)
    return MultilayeredNetwork(
        **kwargs,
        output_clamp_max=optional_float("model_output_clamp_max"),
        output_rectify=bool(fit["model_output_rectify"]),
    )


def simulate_trace(
    model, inputs: torch.Tensor, node_columns: Mapping, initial_state=None
) -> pd.DataFrame:
    """
    Run the model on ``inputs`` and return selected nodes as a DataFrame.

    Args:
        model (MultilayeredNetwork or LinearNetwork): The network.
        inputs (torch.Tensor): Input of shape (1, num_sensory, T).
        node_columns (Mapping): ``{column name: node index or list of node
            indices}``. Lists are averaged, e.g. over the neurons of a cell type.
        initial_state (optional): Passed to ``model.forward``. Defaults to None.

    Returns:
        pd.DataFrame: One row per time step, one column per entry of
        ``node_columns``.
    """
    with torch.no_grad():
        output = model(inputs, checkpoint_steps=0, initial_state=initial_state)
    if output.ndim == 3:
        output = output[0]
    values = output.detach().cpu().numpy()  # (nodes, T)
    columns = {}
    for name, selector in node_columns.items():
        if isinstance(selector, (int, np.integer)):
            members = [int(selector)]
        else:
            members = [int(member) for member in selector]
        # one group per column: get_neuron_activation averages its members
        frame = get_neuron_activation(
            values, members, idx_to_group={member: name for member in members}
        )
        columns[name] = frame.drop(columns="group").to_numpy()[0]
    return pd.DataFrame(columns)


def pair_slope_table(model) -> pd.DataFrame:
    """
    The per-connection slopes of a model with pairwise slopes: columns ``pre``,
    ``post``, ``pair_slope`` (raw) and ``effective_pair_slope`` (after clamping
    to the bounds). Empty for models with per-node slopes.
    """
    if not model.slope_is_pairwise:
        return pd.DataFrame(
            columns=["pre", "post", "pair_slope", "effective_pair_slope"]
        )
    raw = model.slope.detach().cpu().numpy()
    effective = model.effective_slope.detach().cpu().numpy()
    return pd.DataFrame(
        [
            {
                "pre": pre,
                "post": post,
                "pair_slope": float(raw[i]),
                "effective_pair_slope": float(effective[i]),
            }
            for i, (pre, post) in enumerate(model.slope_pairs)
        ]
    )
