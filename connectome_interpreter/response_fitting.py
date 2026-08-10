"""Tools for fitting network models to measured response time series.

Everything here operates on a :class:`~connectome_interpreter.activation_maximisation.MultilayeredNetwork`
/ :class:`~connectome_interpreter.activation_maximisation.LinearNetwork`, or on
generic (stimulus trace, response trace, transition-window) data. Nothing in
this module knows which nonlinearity a model uses: state-dependent local gains
are read through ``model.activation_gain`` (see the ``activation_gain_fn``
constructor argument), so the same stability tools serve linear and nonlinear
fits.

Sections:

- **Dynamics** — single steps, iterative fixed points, and the closed-form
  linear steady state.
- **Stability** — spectral radius of the free-node update Jacobian and a
  differentiable penalty, with a dense exact path for small networks and a
  sparse Gershgorin / ARPACK path for large ones.
- **Sensor** — :class:`ExponentialSensor`, a fixed causal indicator kernel
  (e.g. GCaMP6f) applied identically in training (torch) and diagnostics
  (numpy). Pass ``sensor=None`` conventions downstream for e.g. voltage
  recordings.
- **Affine readout & loss** — closed-form per-channel scale/offset readout
  (variable projection), its regularisers, and
  :func:`make_affine_readout_loss`, a ``train_model``-compatible loss factory.
- **Windows & metrics** — transition-window scoring masks, long-format target
  builders and window extraction.
- **Persistence & tables** — :func:`save_fit` / :func:`load_fit` /
  :func:`rebuild_network` for a self-contained fitted-model round-trip, plus
  tidy parameter and trace tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse

from .activation_maximisation import LinearNetwork, MultilayeredNetwork

__all__ = [
    # dynamics
    "free_and_sensory_indices",
    "network_step",
    "network_fixed_point",
    "linear_network_steady_state",
    # stability
    "free_update_matrix",
    "stability_penalty",
    "spectral_radius",
    "spectral_radius_bound",
    # sensor
    "ExponentialSensor",
    "GCAMP6F_TAU_MS",
    # readout & loss
    "affine_readout_solve",
    "affine_readout_penalties",
    "affine_readout_frame",
    "implied_affine",
    "make_affine_readout_loss",
    "BoundedLogScalar",
    "make_log_offset_input_transform",
    "log_trace",
    "trace_to_input_tensor",
    # windows & metrics
    "score_mask_for_transition_windows",
    "build_raw_window_targets",
    "extract_transition_windows",
    "r2",
    # persistence & tables
    "save_fit",
    "load_fit",
    "rebuild_network",
    "simulate_trace",
    "pair_slope_table",
    "parameter_table",
    "aggregated_pair_weight_matrix",
    "to_jsonable",
]


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


def free_and_sensory_indices(model):
    """Split a network's nodes into free (dynamical) and sensory indices.

    Args:
        model: the network to inspect.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(free_idx, sensory_idx)`` long
        tensors. ``sensory_idx`` are the clamped input nodes
        (``model.sensory_indices``); ``free_idx`` is everything else.
    """
    n_nodes = model.all_weights.shape[0]
    device = model.all_weights.device
    sensory_idx = model.sensory_indices.detach().long().to(device)
    all_idx = torch.arange(n_nodes, dtype=torch.long, device=device)
    sensory_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    sensory_mask[sensory_idx] = True
    free_idx = all_idx[~sensory_mask]
    return free_idx, sensory_idx


def network_step(model, state, sensory_values):
    """One discrete-time step of the network with clamped sensory nodes.

    Applies the model's installed activation (built-in or custom) plus its
    ``output_rectify`` / ``output_clamp_max`` post-processing where present, so
    the step matches ``model.forward`` for a single timestep.

    Args:
        model: the network. Must run with ``sensory_input_mode="replace"``
            (raises otherwise: in ``"add"`` mode the sensory nodes remain
            dynamical and a clamped step would misrepresent the dynamics).
        state (array-like or torch.Tensor): full-network state, shape
            ``(n_nodes,)``.
        sensory_values (array-like or torch.Tensor): values for the sensory
            nodes, in the order of ``model.sensory_indices``.

    Returns:
        torch.Tensor: the next state, shape ``(n_nodes,)``.
    """
    if model.sensory_input_mode != "replace":
        raise ValueError("network_step assumes sensory_input_mode='replace'.")
    device = model.all_weights.device
    state = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(-1, 1)
    sensory_values = torch.as_tensor(
        sensory_values, dtype=torch.float32, device=device
    ).reshape(-1, 1)
    if sensory_values.numel() != model.sensory_indices.numel():
        raise ValueError("sensory_values length must match model.sensory_indices.")

    x = torch.sparse.mm(model.effective_weights, state)
    x = model.activation_function(x, x_previous=state)
    x = model._apply_sensory_input(x, sensory_values)
    if getattr(model, "output_rectify", False):
        x = torch.relu(x - model.threshold) + model.threshold * (
            x >= model.threshold
        ).to(x.dtype)
    if getattr(model, "output_clamp_max", None) is not None:
        x = torch.clamp(x, max=model.output_clamp_max)
    x = model._apply_sensory_input(x, sensory_values)
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
    """Iterate :func:`network_step` to a fixed point for constant sensory input.

    Warm-startable: pass the previous fixed point as ``initial_state`` to
    converge in a few steps when the parameters have only moved slightly (e.g.
    between optimizer steps). Runs under ``torch.no_grad()`` — the returned
    state carries no graph.

    Args:
        model: the network (``sensory_input_mode="replace"``).
        sensory_values (array-like): constant sensory drive, in the order of
            ``model.sensory_indices``.
        initial_state (array-like, optional): starting state; zeros if None.
        max_steps (int): iteration cap.
        min_steps (int): minimum iterations before the tolerance may stop.
        tol (float): max-abs state change below which iteration stops.
        return_info (bool): also return an info dict.

    Returns:
        torch.Tensor, or ``(state, info)`` when ``return_info`` — ``info`` has
        ``"iterations"``, ``"residual"``, ``"converged"``.
    """
    device = model.all_weights.device
    sensory_values = torch.as_tensor(
        sensory_values, dtype=torch.float32, device=device
    ).reshape(-1)
    _, sensory_idx = free_and_sensory_indices(model)
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
            next_state = network_step(model, state, sensory_values)
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


def _effective_weights_to_scipy(model):
    """``model.effective_weights`` as a scipy CSR matrix (post in rows, pre in
    cols), float64. Used by the sparse (large-network) routines."""
    weights = model.effective_weights.coalesce()
    idx = weights.indices().detach().cpu().numpy()
    vals = weights.values().detach().cpu().numpy().astype(np.float64, copy=False)
    n = weights.shape[0]
    return sparse.coo_matrix((vals, (idx[0], idx[1])), shape=(n, n)).tocsr()


def _steady_state_sparse(model, sensory_values, free_idx, sensory_idx, slopes, biases):
    """Sparse, non-differentiable solve of the free-node linear steady state.

    Solves ``(I - diag(slope_f) W_ff) x_f = slope_f * (W_fs s) + bias_f`` with a
    sparse LU factorisation (scipy ``splu``), on CPU/float64.
    """
    from scipy.sparse import linalg as spla

    weights = _effective_weights_to_scipy(model)
    free = free_idx.detach().cpu().numpy()
    sens = sensory_idx.detach().cpu().numpy()
    weights_ff = weights[free][:, free]
    weights_fs = weights[free][:, sens]
    slope_f = slopes.detach().cpu().numpy()[free].astype(np.float64)
    bias_f = biases.detach().cpu().numpy()[free].astype(np.float64)
    sens_vals = sensory_values.detach().cpu().numpy().astype(np.float64)

    system = (
        sparse.eye(free.size, format="csc") - (sparse.diags(slope_f) @ weights_ff)
    ).tocsc()
    rhs = slope_f * (weights_fs @ sens_vals) + bias_f
    solution = spla.splu(system).solve(rhs)
    if not np.isfinite(solution).all():
        raise ValueError("Sparse steady-state solve returned non-finite values.")
    return torch.as_tensor(
        solution, dtype=torch.float32, device=model.all_weights.device
    )


def linear_network_steady_state(model, sensory_values):
    """Closed-form steady state of a linear network given clamped sensory inputs.

    Solves the fixed point of the linear update with per-node slopes and biases,
    treating ``model.sensory_indices`` as clamped to ``sensory_values``. Assumes
    the linear (identity) activation regime. The fixed point is independent of
    ``tau``, so ``tau`` is not used here.

    Args:
        model (LinearNetwork): the network.
        sensory_values (array-like or torch.Tensor): values for the sensory
            nodes, in the order of ``model.sensory_indices``.

    Returns:
        torch.Tensor: the steady-state activity over all nodes.

    Note:
        Treating the sensory nodes as clamped is only exact when the model runs
        with ``sensory_input_mode="replace"``; in ``"add"`` mode the sensory
        nodes remain dynamical and this is an approximation.
        Networks with at most ``_DENSE_NODE_LIMIT`` nodes use a dense
        differentiable solve; larger ones use a sparse LU solve (scipy
        ``splu``, CPU/float64) which is **not** differentiable.
    """
    free_idx, sensory_idx = free_and_sensory_indices(model)
    device = model.all_weights.device
    n_nodes = model.all_weights.shape[0]
    sensory_values = torch.as_tensor(
        sensory_values,
        dtype=torch.float32,
        device=device,
    ).reshape(-1)
    if sensory_values.numel() != sensory_idx.numel():
        raise ValueError("sensory_values length must match model.sensory_indices.")

    slopes = model.node_parameter("slope")
    biases = model.node_parameter("bias")
    state = torch.zeros(n_nodes, dtype=torch.float32, device=device)
    state[sensory_idx] = sensory_values
    if free_idx.numel() == 0:
        return state

    if n_nodes > _DENSE_NODE_LIMIT:
        with torch.no_grad():
            state[free_idx] = _steady_state_sparse(
                model, sensory_values, free_idx, sensory_idx, slopes, biases
            )
        return state

    W = model.effective_weights.to_dense()
    W_ff = W.index_select(0, free_idx).index_select(1, free_idx)
    W_fs = W.index_select(0, free_idx).index_select(1, sensory_idx)
    slopes_f = slopes.index_select(0, free_idx)
    biases_f = biases.index_select(0, free_idx)
    eye = torch.eye(free_idx.numel(), dtype=W.dtype, device=W.device)
    lhs = eye - slopes_f.view(-1, 1) * W_ff
    rhs = slopes_f * (W_fs @ sensory_values) + biases_f
    state[free_idx] = torch.linalg.solve(lhs, rhs)
    return state


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

# Networks with at most this many nodes use exact dense routines (closed-form
# solve and full eigendecomposition); larger networks switch to sparse routines.
# The binding cost is the dense eigendecomposition in stability_penalty
# (O(n^3), evaluated every training step). On CPU that is ~20 ms at n=256 but
# ~240 ms at n=800, so 256 keeps the exact penalty wherever it is still cheap
# and hands off to the sparse Gershgorin bound only for substantially larger
# models. It sits comfortably above any cell-type model and well below neuron
# scale.
_DENSE_NODE_LIMIT = 256


def free_update_matrix(model, state=None):
    """One-step update Jacobian restricted to the free (non-sensory) nodes.

    Builds ``diag((tau - 1) / tau) + (gain / tau) * W`` — where ``gain`` is the
    model's per-node local gain at ``state`` (``model.activation_gain``) — and
    restricts it to the free-node block. Its spectral radius governs
    discrete-time stability of the linearised network.

    Args:
        model: the network.
        state (torch.Tensor, optional): operating point. Required whenever the
            model's gain is state-dependent (custom activations, the built-in
            ``MultilayeredNetwork`` activation); a ``LinearNetwork`` needs none.

    Returns:
        torch.Tensor: square update matrix over the free nodes (possibly
        ``(0, 0)`` if there are no free nodes).

    Note:
        Restricting to the free-node block is the correct linearisation only
        when the sensory nodes are clamped (``sensory_input_mode="replace"``).
        Densifies the weight matrix; it is the dense backend used by
        :func:`stability_penalty` and :func:`spectral_radius` for networks with
        at most ``_DENSE_NODE_LIMIT`` nodes.
    """
    W = model.effective_weights.to_dense()
    free_idx, _ = free_and_sensory_indices(model)
    if free_idx.numel() == 0:
        return torch.zeros((0, 0), dtype=W.dtype, device=W.device)
    gain = model.activation_gain(state)
    taus = model.node_parameter("tau")
    update = torch.diag((taus - 1.0) / taus) + (gain / taus).view(-1, 1) * W
    return update.index_select(0, free_idx).index_select(1, free_idx)


def _gershgorin_spectral_radius_bound(model, free_idx, gain):
    """Differentiable Gershgorin upper bound on the free-block spectral radius.

    Every eigenvalue of the free-node update matrix ``M`` lies within a disc
    centred at ``M_ii`` of radius ``sum_{j!=i} |M_ij|``, so
    ``max_i (|M_ii| + sum_{j!=i} |M_ij|)`` upper-bounds the spectral radius.
    Computed from the sparse edge list (no densification), so it scales to
    neuron-sized networks and stays differentiable w.r.t. weights/gain/tau.
    """
    weights = model.effective_weights.coalesce()
    idx = weights.indices()
    post, pre = idx[0], idx[1]
    vals = weights.values()
    n = weights.shape[0]
    device = vals.device

    taus = model.node_parameter("tau")
    scale = gain / taus  # per-row (post-synaptic) factor on W

    free_mask = torch.zeros(n, dtype=torch.bool, device=device)
    free_mask[free_idx] = True
    edge_free = free_mask[post] & free_mask[pre]
    post_f = post[edge_free]
    m_vals = scale[post_f] * vals[edge_free]  # M_ij contributions from W
    diag_edge = post_f == pre[edge_free]

    leak = (taus - 1.0) / taus  # diagonal contribution of the leak term
    diag_from_w = torch.zeros(n, dtype=m_vals.dtype, device=device)
    if torch.any(diag_edge):
        diag_from_w = diag_from_w.index_add(0, post_f[diag_edge], m_vals[diag_edge])
    m_diag = leak + diag_from_w

    offdiag_abs = torch.zeros(n, dtype=m_vals.dtype, device=device)
    off_edge = ~diag_edge
    if torch.any(off_edge):
        offdiag_abs = offdiag_abs.index_add(
            0, post_f[off_edge], torch.abs(m_vals[off_edge])
        )

    gershgorin = torch.abs(m_diag) + offdiag_abs
    return gershgorin[free_idx].max()


def _spectral_radius_sparse(model, free_idx, gain):
    """Spectral radius of the free-node update matrix via ARPACK (float, no grad)."""
    from scipy.sparse import linalg as spla

    weights = _effective_weights_to_scipy(model)
    free = free_idx.detach().cpu().numpy()
    weights_ff = weights[free][:, free]
    gain_f = gain.detach().cpu().numpy()[free].astype(np.float64)
    taus = (
        model.node_parameter("tau").detach().cpu().numpy()[free].astype(np.float64)
    )
    update = sparse.diags((taus - 1.0) / taus) + sparse.diags(gain_f / taus) @ weights_ff
    update = update.tocsr()
    if free.size <= 2:
        eigenvalues = np.linalg.eigvals(update.toarray())
    else:
        eigenvalues = spla.eigs(update, k=1, which="LM", return_eigenvectors=False)
    return float(np.abs(eigenvalues).max())


def stability_penalty(model, state=None, margin: float = 1e-4):
    """Differentiable penalty pushing the free-node spectral radius below 1.

    Returns ``relu(rho - (1 - margin)) ** 2``. Zero when the network is
    comfortably stable; positive (and gradient-bearing) otherwise. Suitable as
    an additive term in a ``train_model`` ``activation_loss_fn`` (see
    :func:`make_affine_readout_loss`).

    The local gain comes from ``model.activation_gain(state)``: a model with a
    custom ``activation_function`` and no ``activation_gain_fn`` **raises**
    rather than silently penalising the wrong Jacobian.

    Args:
        model: the network.
        state (torch.Tensor, optional): operating point for state-dependent
            gains; a ``LinearNetwork`` needs none.
        margin (float): stability margin below 1.

    Returns:
        torch.Tensor: scalar penalty.

    Note:
        For networks with at most ``_DENSE_NODE_LIMIT`` nodes, ``rho`` is the
        exact spectral radius of :func:`free_update_matrix` (dense
        eigendecomposition). For larger networks, ``rho`` is a sparse,
        differentiable **Gershgorin upper bound** on the spectral radius. The
        bound is conservative, so switching to the sparse regime penalises more
        aggressively than the exact version at the same ``margin``/loss weight.
        Both regimes assume clamped sensory nodes
        (``sensory_input_mode="replace"``).
    """
    free_idx, _ = free_and_sensory_indices(model)
    if free_idx.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=model.all_weights.device)

    if model.all_weights.shape[0] > _DENSE_NODE_LIMIT:
        gain = model.activation_gain(state)
        rho = _gershgorin_spectral_radius_bound(model, free_idx, gain)
    else:
        update = free_update_matrix(model, state)
        rho = torch.abs(torch.linalg.eigvals(update)).max()
    return torch.relu(rho - (1.0 - float(margin))) ** 2


def spectral_radius_bound(model, state=None) -> float:
    """Gershgorin upper bound on the free-node spectral radius, as a float.

    No-grad diagnostic counterpart of the *sparse* regime of
    :func:`stability_penalty`: the same conservative bound the penalty
    optimises against on large networks, reported as a number. O(nnz), never
    densifies — usable at neuron scale where :func:`spectral_radius`'s ARPACK
    solve may be slow. A value below 1 certifies stability; above 1 is
    inconclusive (the bound is conservative).

    Args:
        model: the network.
        state (torch.Tensor, optional): operating point for state-dependent
            gains; a ``LinearNetwork`` needs none.

    Returns:
        float: the bound (0.0 if there are no free nodes).
    """
    with torch.no_grad():
        free_idx, _ = free_and_sensory_indices(model)
        if free_idx.numel() == 0:
            return 0.0
        gain = model.activation_gain(state)
        return float(
            _gershgorin_spectral_radius_bound(model, free_idx, gain).detach().cpu()
        )


def spectral_radius(model, state=None):
    """Spectral radius of the free-node update matrix, as a Python float.

    No-grad diagnostic counterpart to :func:`stability_penalty`. A value below
    1 indicates a stable discrete-time linearisation at ``state``.

    Args:
        model: the network.
        state (torch.Tensor, optional): operating point for state-dependent
            gains; a ``LinearNetwork`` needs none.

    Returns:
        float: the spectral radius (0.0 if there are no free nodes).

    Note:
        Networks with at most ``_DENSE_NODE_LIMIT`` nodes use a dense
        eigendecomposition; larger ones use ARPACK
        (``scipy.sparse.linalg.eigs``) to get the dominant eigenvalue without
        densifying. Unlike :func:`stability_penalty`, this returns the true
        spectral radius in both regimes (no Gershgorin bound). Assumes clamped
        sensory nodes (``sensory_input_mode="replace"``).
    """
    with torch.no_grad():
        free_idx, _ = free_and_sensory_indices(model)
        if free_idx.numel() == 0:
            return 0.0
        if model.all_weights.shape[0] > _DENSE_NODE_LIMIT:
            gain = model.activation_gain(state)
            return _spectral_radius_sparse(model, free_idx, gain)
        update = free_update_matrix(model, state)
        return float(torch.abs(torch.linalg.eigvals(update)).max().detach().cpu())


# ---------------------------------------------------------------------------
# Sensor (forward observation model)
# ---------------------------------------------------------------------------

# GCaMP6f fluorescence decay time constant (ms). Reasonable literature range
# ~200-500 ms (Chen et al. 2013 report a single-AP decay half-time ~140-200 ms
# -> tau ~200-300 ms). 300 ms is a middle-of-the-road default; it is one
# constant to change and can be swept.
GCAMP6F_TAU_MS = 300.0
_SENSOR_SUPPORT_TAU = 6.0  # truncate the kernel at this many tau (exp(-6) ~ 0.25%)


class ExponentialSensor:
    """Fixed linear indicator forward model: a causal single-exponential kernel.

    Forward observation model for fitting a rate model to *raw* (un-
    deconvolved) indicator data such as calcium dF/F. Instead of deconvolving
    the data (ill-posed for graded cells), the model's latent activity is
    convolved with a fixed kernel before comparison — so the recovered rate-
    model time constants are the neural ones, not neural-convolved-with-sensor.

    The kernel is a sum-normalised causal single exponential (first-order
    buffering/extrusion), ``k(t) = exp(-t / tau) / Z``. Because ``sum(k) = 1``
    the DC gain is 1: sustained/plateau levels pass through unchanged and only
    transient kinetics are low-passed. It is linear time-invariant: it reshapes
    and delays transients but manufactures no increment/decrement amplitude
    asymmetry (that would need a static nonlinearity).

    The same kernel drives both application paths, so they cannot drift:

    - :meth:`output_transform` — torch, differentiable; pass the bound method
      to ``train_model(output_transform=sensor.output_transform)``.
    - :meth:`apply` — numpy, for post-fit diagnostics.

    For recordings without an indicator (e.g. voltage), pass ``sensor=None`` /
    ``output_transform=None`` downstream rather than an identity sensor.

    Args:
        tau_ms (float): decay time constant in **milliseconds** (matching
            ``dt_ms``, model ``tau`` and window lengths elsewhere in a fit).
            Defaults to ``GCAMP6F_TAU_MS`` (300 ms). Do **not** co-fit this
            with the rate model's own time constants — they are degenerate.
        dt_ms (float): sampling step of the traces, in ms.
        support_tau (float): kernel truncation, in multiples of ``tau_ms``.
        device (torch.device, optional): device to pre-place the torch kernel
            on (it follows the input's device regardless).
        kernel (array-like, optional): externally built kernel; overrides
            ``tau_ms``/``support_tau``.
    """

    def __init__(
        self,
        tau_ms: float = GCAMP6F_TAU_MS,
        dt_ms: float = 1.0,
        support_tau: float = _SENSOR_SUPPORT_TAU,
        device=None,
        kernel=None,
    ):
        if kernel is None:
            if tau_ms <= 0:
                raise ValueError("tau_ms must be positive.")
            if dt_ms <= 0:
                raise ValueError("dt_ms must be positive.")
            n = max(1, int(round(support_tau * tau_ms / dt_ms)))
            t = np.arange(n) * float(dt_ms)
            kernel = np.exp(-t / float(tau_ms))
            kernel = kernel / kernel.sum()
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.tau_ms = float(tau_ms)
        self.dt_ms = float(dt_ms)
        # F.conv1d is cross-correlation, so flip the kernel to get true
        # convolution y[t] = sum_k kernel[k] * x[t-k].
        weight = torch.as_tensor(self.kernel[::-1].copy(), dtype=torch.float32).view(
            1, 1, -1
        )
        if device is not None:
            weight = weight.to(device)
        self._weight = weight

    def output_transform(self, outputs: torch.Tensor) -> torch.Tensor:
        """Causal sensor convolution along the time axis (torch path).

        ``outputs`` is ``(batch, neurons, T)``; each node's time series is
        convolved with the fixed kernel, causally, replicate-padded at the
        start (the sensor starts from the initial steady state — no start-up
        ramp). Differentiable w.r.t. ``outputs``; the kernel carries no
        gradient. Shape-preserving, as ``train_model`` requires.
        """
        if outputs.dim() != 3:
            raise ValueError("output_transform expects (batch, neurons, T).")
        b, n, t = outputs.shape
        ksize = self._weight.shape[-1]
        x = outputs.reshape(b * n, 1, t)
        x = F.pad(x, (ksize - 1, 0), mode="replicate")  # causal, start from edge value
        y = F.conv1d(x, self._weight.to(x.device))
        return y.reshape(b, n, t)

    def apply(self, trace):
        """Causal convolution of a numpy trace with the kernel (numpy path).

        ``trace`` is ``(T,)`` or ``(n, T)``; returns the same shape. Matches
        :meth:`output_transform` (replicate-padded, causal).
        """
        trace = np.asarray(trace, dtype=np.float64)
        kernel = np.asarray(self.kernel, dtype=np.float64)
        if trace.ndim == 1:
            pad = np.concatenate([np.full(kernel.size - 1, trace[0]), trace])
            return np.convolve(pad, kernel)[
                kernel.size - 1 : kernel.size - 1 + trace.size
            ]
        if trace.ndim == 2:
            return np.stack([self.apply(row) for row in trace])
        raise ValueError("trace must be 1-D (T,) or 2-D (n, T).")


# ---------------------------------------------------------------------------
# Affine readout & loss factory
# ---------------------------------------------------------------------------


def affine_readout_solve(latent, target, ridge: float = 1e-4):
    """Closed-form nonnegative scale + offset for one readout channel.

    Variable projection: minimises ``||scale * latent + offset - target||^2``
    with a small ridge on the scale (numerical only), then rectifies the scale
    to be nonnegative. ``scale`` and ``offset`` are differentiable w.r.t.
    ``latent``, so gradients reach the dynamics that produced it.

    Args:
        latent (torch.Tensor): 1-D latent samples for this channel.
        target (torch.Tensor): 1-D target samples, same length.
        ridge (float): Tikhonov term on the scale solve.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(scale, offset)`` as 0-dim
        tensors.
    """
    lm = latent.mean()
    tm = target.mean()
    lc = latent - lm
    tc = target - tm
    scale = (lc * tc).mean() / (lc.pow(2).mean() + ridge)
    scale = torch.relu(scale)  # nonnegative scale
    offset = tm - scale * lm
    return scale, offset


def affine_readout_penalties(
    scales,
    latent_stds,
    scale_soft_limit: float = 10.0,
    latent_std_floor: float = 0.02,
):
    """Readout-scale soft-limit and latent-std-floor penalties (means over channels).

    The scale penalty ``mean(log1p(|s| / scale_soft_limit) ** 2)`` discourages
    the readout from amplifying a vanishing latent; the std penalty
    ``mean(relu(latent_std_floor - std) ** 2)`` keeps the latent itself alive.

    Args:
        scales (torch.Tensor): 1-D, one readout scale per channel.
        latent_stds (torch.Tensor): 1-D, one latent std per channel.
        scale_soft_limit (float): knee of the scale penalty.
        latent_std_floor (float): floor below which latent std is penalised.

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
    """Solve the closed-form readout per channel on final latent windows.

    Args:
        latent_windows (Mapping): ``{layer: array}`` of latent samples (any
            shape; flattened).
        target_windows (Mapping): ``{layer: array}`` of matching targets.
        layers (sequence, optional): channel order; defaults to
            ``latent_windows`` keys.
        ridge (float): passed to :func:`affine_readout_solve`.
        window_indices (sequence, optional): subset of window rows (first
            axis of each layer's array) to solve on — e.g. the train windows
            of a train/test split, so the readout never sees held-out
            windows. Default: all rows.

    Returns:
        tuple[pd.DataFrame, dict]: the readout-parameters table (columns
        ``layer, readout_offset, readout_scale, latent_mean, target_mean,
        latent_std, target_std``) and ``{layer: (scale, offset)}`` floats.
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


def implied_affine(x, y):
    """Implied optimal affine calibration of ``x`` to ``y`` (numpy, no ridge).

    The closed-form nonnegative scale and offset that would best rescale a
    model output to the target — useful to check whether a model fitted
    *without* a readout is already calibrated (scale ~1, offset ~0 makes a
    readout-free comparison fair).

    Returns:
        tuple[float, float]: ``(scale, offset)``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    xc = x - x.mean()
    yc = y - y.mean()
    scale = float((xc * yc).sum() / (xc * xc).sum())
    scale = max(scale, 0.0)
    offset = float(y.mean() - scale * x.mean())
    return scale, offset


def _require_activation_gain(model):
    """Raise if Jacobian-based analysis on ``model`` would be impossible."""
    if (
        model.custom_activation_function is not None
        and model.custom_activation_gain_fn is None
    ):
        raise ValueError(
            "stability_weight > 0 requires the model's local gain, but this "
            "model has a custom activation_function and no activation_gain_fn. "
            "Pass activation_gain_fn=... at model construction, or set "
            "stability_weight=0 to fit without the stability penalty."
        )


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
    """Build a ``train_model`` ``activation_loss_fn``: affine-readout MSE
    (+ optional stability penalty + readout regularisers).

    Per readout channel, the closed-form nonnegative scale + offset
    (:func:`affine_readout_solve`) is applied to the model's (flattened)
    prediction before the MSE, so the dynamics are fitted up to an affine
    calibration per channel.

    ``train_model`` flattens the targets in its train/test-split row order,
    which is a deterministic pure function of the ``targets`` frame; this
    factory replicates that order to build per-channel masks. Single-batch
    targets only (``batch == 0`` throughout).

    Args:
        model: the network being fitted (used for the stability term).
        targets (pd.DataFrame): the same frame passed to ``train_model``
            (columns ``batch, neuron_idx, layer, value``).
        layer_ids (sequence of int): the ``neuron_idx`` value of each readout
            channel, in channel order — model row indices for per-node targets,
            group ids when fitting with ``target_node_groups``.
        expected_layer_windows (Mapping, optional): ``{layer_id: array}`` of
            the target windows from the data source. When given, the first loss
            evaluation asserts each channel's flattened targets reproduce that
            array's mean/std — a cheap end-to-end check that the replicated
            ordering is right.
        ridge, scale_soft_limit, latent_std_floor, scale_reg_weight,
        std_reg_weight: readout solve/penalty parameters, see
            :func:`affine_readout_solve` / :func:`affine_readout_penalties`.
        stability_weight (float): weight of :func:`stability_penalty` in the
            loss. Defaults to ``0.0`` — stability is explicit opt-in. A
            positive value with a model whose local gain is unavailable (custom
            activation, no ``activation_gain_fn``) raises **here**, at build
            time, rather than silently fitting without the constraint.
        stability_margin (float): margin passed to :func:`stability_penalty`.
        stability_state_fn (callable, optional): ``fn(previous_state) ->
            state`` returning the operating point to linearise at, called once
            per loss evaluation with the previous returned state (None on the
            first call) so a fixed-point solver can warm-start. Omit for
            state-independent gains (``LinearNetwork``).
        normalize_by_target_var (bool): when True, each channel's MSE is
            weighted by ``mean(target_var) / target_var(channel)`` before
            averaging — inverse-variance channel weighting, so a channel's
            contribution measures its *relative* misfit (proportional to
            ``1 - R^2``) rather than its raw amplitude, and small-amplitude
            channels are no longer under-served. The ``mean(target_var)``
            factor keeps the overall loss on the same scale as the
            unnormalized version (equal-variance channels reproduce it
            exactly), so the penalty weights and loss histories stay
            comparable. Variances come from the targets of each evaluation.

    Returns:
        callable: ``loss_fn(pred, target) -> scalar torch.Tensor``.
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
        _require_activation_gain(model)

    # Replicate train_model's flat target order: filter to the batch, then
    # sort_values("batch") (not stable on the all-equal key, but deterministic),
    # and group by neuron_idx with boolean masks instead of assuming contiguous
    # blocks. Within-channel order is irrelevant to per-channel reductions.
    order = targets[targets["batch"].isin([0])].copy()
    order.loc[:, ["batch"]] = pd.Categorical(order["batch"], categories=[0])
    order = order.sort_values(by="batch")
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
            loss = torch.stack(mses).mean()  # equal-sized channels -> global MSE
        scale_penalty, std_penalty = affine_readout_penalties(
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


class BoundedLogScalar:
    """A trainable positive scalar, box-bounded via a log-scale reparametrisation.

    ``value() = exp(log_lo + (log_hi - log_lo) * sigmoid(raw))`` — updates to
    the raw parameter act multiplicatively on the value, which stays strictly
    inside ``bounds`` with no post-step projection. Used e.g. for the luminance
    log-offset epsilon in ``log(brightness + epsilon)``: pass ``[scalar.raw]``
    as ``train_model(extra_parameters=...)`` and read the fitted value back
    with :meth:`item`.

    Args:
        init (float): initial value, strictly inside ``bounds``.
        bounds (tuple): ``(lower, upper)``, both positive.
        device (torch.device, optional): device for the raw parameter.
    """

    def __init__(self, init: float, bounds, device=None):
        lower, upper = float(bounds[0]), float(bounds[1])
        if not (0 < lower < upper):
            raise ValueError("bounds must be positive and increasing.")
        if not (lower < float(init) < upper):
            raise ValueError("init must lie strictly inside bounds.")
        self.bounds = (lower, upper)
        self._log_lo = float(np.log(lower))
        self._log_hi = float(np.log(upper))
        frac = (float(np.log(float(init))) - self._log_lo) / (
            self._log_hi - self._log_lo
        )
        self.raw = torch.nn.Parameter(
            torch.logit(torch.tensor(frac, dtype=torch.float32, device=device))
        )

    def value(self) -> torch.Tensor:
        """The current bounded value, differentiable w.r.t. ``raw``."""
        return torch.exp(
            self._log_lo + (self._log_hi - self._log_lo) * torch.sigmoid(self.raw)
        )

    def item(self) -> float:
        """The current bounded value as a Python float."""
        return float(self.value().detach().cpu())


def make_log_offset_input_transform(offset: BoundedLogScalar) -> Callable:
    """Build a ``train_model`` ``input_transform``: ``log(inputs + offset)``.

    Feed ``train_model`` the *raw* (e.g. brightness) input tensor together with
    this transform and ``extra_parameters=[offset.raw]``; the transform is
    re-applied every epoch, so the trainable offset receives gradients.
    """

    def input_transform(inputs: torch.Tensor) -> torch.Tensor:
        return torch.log(inputs + offset.value())

    return input_transform


def log_trace(trace, offset: float) -> np.ndarray:
    """``log(trace + offset)`` as float32, validating positivity."""
    trace = np.asarray(trace, dtype=np.float32)
    shifted = trace + np.float32(offset)
    if np.any(shifted <= 0):
        raise ValueError("trace + offset must be positive everywhere.")
    return np.log(shifted).astype(np.float32)


def trace_to_input_tensor(trace, device=None) -> torch.Tensor:
    """A 1-D stimulus trace as a ``(1, 1, T)`` ``train_model`` input tensor."""
    trace = np.asarray(trace, dtype=np.float32)
    if trace.ndim != 1:
        raise ValueError("trace must be 1-D (T,).")
    return torch.as_tensor(trace[None, None, :], dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Windows & metrics
# ---------------------------------------------------------------------------


def score_mask_for_transition_windows(
    n_samples: int, transition_idx, window_steps: int
) -> np.ndarray:
    """Boolean mask selecting the scored post-transition windows of a trace.

    Args:
        n_samples (int): trace length in samples.
        transition_idx (array-like of int): start sample of each window.
        window_steps (int): window length in samples.

    Returns:
        np.ndarray: boolean mask of length ``n_samples``.

    Raises:
        ValueError: if a window runs past the trace end or windows overlap.
    """
    transition_idx = np.asarray(transition_idx, dtype=np.int64)
    n_samples = int(n_samples)
    n_steps = int(window_steps)
    if n_steps < 1:
        raise ValueError("window_steps must be >= 1.")
    if np.any(transition_idx + n_steps > n_samples):
        raise ValueError("At least one scoring window exceeds the trace length.")
    if transition_idx.size > 1 and np.any(np.diff(transition_idx) < n_steps):
        raise ValueError("Scoring windows overlap; reduce window_steps.")
    mask = np.zeros(n_samples, dtype=bool)
    for start in transition_idx:
        mask[int(start) : int(start) + n_steps] = True
    if int(mask.sum()) != int(transition_idx.size) * n_steps:
        raise ValueError("Score mask sample count does not match window count.")
    return mask


def build_raw_window_targets(
    target_traces: Mapping,
    transition_idx,
    window_steps: int,
    node_index: Mapping,
    layers: Optional[Sequence] = None,
) -> pd.DataFrame:
    """Long-format ``train_model`` targets over post-transition windows.

    One row per (channel, scored sample): columns ``batch`` (always 0),
    ``neuron_idx``, ``layer`` (the timestep) and ``value``.

    Args:
        target_traces (Mapping): ``{layer: 1-D target trace}``.
        transition_idx (array-like of int): start sample of each window.
        window_steps (int): window length in samples.
        node_index (Mapping): ``{layer: neuron_idx}`` — the model row index for
            per-node targets, or the group id when fitting with
            ``target_node_groups``. One signature serves both conventions.
        layers (sequence, optional): channel order; defaults to
            ``target_traces`` keys.

    Returns:
        pd.DataFrame: the targets frame.
    """
    layers = list(target_traces) if layers is None else list(layers)
    n_steps = int(window_steps)
    rows = []
    for layer in layers:
        neuron_idx = int(node_index[layer])
        trace = np.asarray(target_traces[layer], dtype=np.float32)
        for start in np.asarray(transition_idx, dtype=np.int64):
            start = int(start)
            stop = start + n_steps
            if stop > trace.size:
                raise ValueError("A target window exceeds the trace length.")
            for layer_idx in range(start, stop):
                rows.append(
                    {
                        "batch": 0,
                        "neuron_idx": neuron_idx,
                        "layer": int(layer_idx),
                        "value": float(trace[layer_idx]),
                    }
                )
    return pd.DataFrame(rows)


def extract_transition_windows(
    traces,
    transition_idx,
    window_steps: int,
    layers: Optional[Sequence] = None,
):
    """Stack per-transition windows out of continuous traces.

    Args:
        traces: a single 1-D trace, or a mapping/DataFrame of them by layer.
        transition_idx (array-like of int): start sample of each window.
        window_steps (int): window length in samples.
        layers (sequence, optional): with mapping input, the layers to extract
            (defaults to all keys/columns).

    Returns:
        np.ndarray ``(n_transitions, window_steps)`` for a single trace, or
        ``{layer: array}`` for mapping input.
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
    """Coefficient of determination of ``y_pred`` against ``y``."""
    y = np.asarray(y, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Persistence & tables
# ---------------------------------------------------------------------------

_MODEL_KEY_PREFIX = "model_"


def _model_arrays(model) -> dict:
    """A model's rebuildable state as flat numpy arrays (``model_*`` keys)."""
    weights = model.effective_weights.coalesce()
    idx = weights.indices().detach().cpu().numpy()
    vals = weights.values().detach().cpu().numpy().astype(np.float32)
    n_nodes = model.all_weights.shape[0]
    out = {
        "model_class": np.asarray(type(model).__name__),
        "model_weights_indices": idx,
        "model_weights_values": vals,
        "model_weights_shape": np.asarray(list(weights.shape), dtype=np.int64),
        "model_node_bias": model.node_parameter("bias")
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "model_node_tau": model.node_parameter("tau")
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "model_node_slope": model.node_parameter("slope")
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
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
        "model_tau_max": np.asarray(
            (
                np.inf
                if getattr(model, "tau_max", None) is None
                else float(model.tau_max)
            ),
            dtype=np.float32,
        ),
        "model_output_clamp_max": np.asarray(
            (
                np.inf
                if getattr(model, "output_clamp_max", None) is None
                else float(model.output_clamp_max)
            ),
            dtype=np.float32,
        ),
        "model_output_rectify": np.asarray(bool(getattr(model, "output_rectify", False))),
    }
    if model.idx_to_group is not None:
        out["model_node_group"] = np.asarray(
            [str(model.idx_to_group[i]) for i in range(n_nodes)]
        )
    if model.slope_is_pairwise:
        pairs = pair_slope_table(model)
        out["model_pair_slope_pre"] = pairs["pre"].to_numpy(dtype=str)
        out["model_pair_slope_post"] = pairs["post"].to_numpy(dtype=str)
        out["model_pair_slope"] = pairs["pair_slope"].to_numpy(dtype=np.float32)
        out["model_effective_pair_slope"] = pairs["effective_pair_slope"].to_numpy(
            dtype=np.float32
        )
    return out


def save_fit(path, model=None, arrays: Optional[Mapping] = None) -> None:
    """Save a fitted model (and any run data) to one compressed ``.npz``.

    The model section stores everything :func:`rebuild_network` needs to
    re-simulate the fit — the **effective** weights (pair-mode gains folded
    in), per-node bias/tau/slope, sensory indices, threshold and the forward
    flags — plus the raw pair-slope table for bookkeeping. Custom activation
    functions cannot be serialised; pass them again at rebuild time.

    Args:
        path: destination ``.npz`` path.
        model (optional): the fitted network to store.
        arrays (Mapping, optional): additional arrays to store alongside
            (traces, windows, readout parameters, ...). Keys must not start
            with ``"model_"`` (reserved for the model section).

    Raises:
        ValueError: on a reserved-key collision.
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
    """Load a :func:`save_fit` file back into a plain ``{key: array}`` dict."""
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def rebuild_network(
    fit: Union[dict, str, Path],
    activation_function: Optional[Callable] = None,
    activation_gain_fn: Optional[Callable] = None,
    device=None,
):
    """Reconstruct a runnable network from a :func:`save_fit` file.

    The rebuilt model carries the saved **effective** weights and exact
    per-node bias/tau/slope values (each node its own parameter group, biases
    signed via ``bias_transform="identity"``), so its forward pass reproduces
    the saved fit — for custom-activation fits, pass the same
    ``activation_function`` (and its ``activation_gain_fn``) again, since
    callables cannot be serialised.

    Args:
        fit: the dict from :func:`load_fit`, or a path to the ``.npz``.
        activation_function (callable, optional): custom activation to install.
        activation_gain_fn (callable, optional): its matching local gain.
        device (torch.device, optional): device for the rebuilt model.

    Returns:
        MultilayeredNetwork or LinearNetwork.
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
    n_nodes = shape[0]
    node_names = [f"n{i}" for i in range(n_nodes)]
    idx_to_group = dict(enumerate(node_names))
    bias_dict = {
        name: float(value)
        for name, value in zip(node_names, np.asarray(fit["model_node_bias"]))
    }
    tau_dict = {
        name: float(value)
        for name, value in zip(node_names, np.asarray(fit["model_node_tau"]))
    }
    slope_dict = {
        name: float(value)
        for name, value in zip(node_names, np.asarray(fit["model_node_slope"]))
    }

    kwargs = dict(
        all_weights=weights,
        sensory_indices=np.asarray(fit["model_sensory_indices"]).tolist(),
        num_layers=int(fit["model_num_layers"]),
        threshold=float(fit["model_threshold"]),
        tanh_steepness=float(fit["model_tanh_steepness"]),
        idx_to_group=idx_to_group,
        bias_dict=bias_dict,
        bias_transform="identity",
        slope_dict=slope_dict,
        tau_dict=tau_dict,
        sensory_input_mode=str(fit["model_sensory_input_mode"]),
        activation_function=activation_function,
        activation_gain_fn=activation_gain_fn,
        device=device,
    )
    # older save_fit files predate model_tau_max; treat as unbounded
    tau_max = float(fit.get("model_tau_max", np.inf))
    kwargs["tau_max"] = None if np.isinf(tau_max) else tau_max
    class_name = str(fit.get("model_class", "MultilayeredNetwork"))
    if class_name == "LinearNetwork":
        return LinearNetwork(**kwargs)
    clamp = float(fit["model_output_clamp_max"])
    kwargs["output_clamp_max"] = None if np.isinf(clamp) else clamp
    kwargs["output_rectify"] = bool(fit["model_output_rectify"])
    return MultilayeredNetwork(**kwargs)


def simulate_trace(
    model, inputs: torch.Tensor, node_columns: Mapping, initial_state=None
) -> pd.DataFrame:
    """Run the model over ``inputs`` and return a tidy per-timestep DataFrame.

    Args:
        model: the network.
        inputs (torch.Tensor): ``(1, num_sensory, T)`` input tensor.
        node_columns (Mapping): ``{column name: node index or sequence of node
            indices}``; sequences are averaged (e.g. a cell type's members).
        initial_state (optional): passed through to ``model.forward``.

    Returns:
        pd.DataFrame: ``T`` rows, one column per ``node_columns`` entry.
    """
    with torch.no_grad():
        output = model(inputs, checkpoint_steps=0, initial_state=initial_state)
    if output.ndim == 3:
        output = output[0]
    values = output.detach().cpu().numpy()  # (nodes, T)
    columns = {}
    for name, selector in node_columns.items():
        if isinstance(selector, (int, np.integer)):
            columns[name] = values[int(selector), :]
        else:
            members = np.asarray(list(selector), dtype=int)
            columns[name] = values[members, :].mean(axis=0)
    return pd.DataFrame(columns)


def pair_slope_table(model) -> pd.DataFrame:
    """Raw and bound-clamped per-(pre, post) slope gains as a DataFrame.

    With ``w_eff = m * w`` the effective per-connection gain simply *is* the
    clamped slope. Empty frame for node-mode models.
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


def parameter_table(
    model, node_columns: Mapping, weight_sums: bool = True
) -> pd.DataFrame:
    """Per-cell-type parameter table: bias, tau and (optionally) weight sums.

    Args:
        model: the network.
        node_columns (Mapping): ``{cell type: node index or sequence of node
            indices}``; sequences are averaged per member.
        weight_sums (bool): also report per-member incoming/outgoing effective
            weight sums (densifies the weight matrix).

    Returns:
        pd.DataFrame: columns ``cell_type, bias, tau`` (+ ``in_weight_sum,
        in_abs_weight_sum, out_weight_sum``).
    """
    biases = model.node_parameter("bias").detach().cpu().numpy()
    taus = model.node_parameter("tau").detach().cpu().numpy()
    weights = (
        model.effective_weights.to_dense().detach().cpu().numpy()
        if weight_sums
        else None
    )
    rows = []
    for name, selector in node_columns.items():
        if isinstance(selector, (int, np.integer)):
            members = np.asarray([int(selector)])
        else:
            members = np.asarray(list(selector), dtype=int)
        row = {
            "cell_type": name,
            "bias": float(biases[members].mean()),
            "tau": float(taus[members].mean()),
        }
        if weight_sums:
            row["in_weight_sum"] = float(weights[members, :].sum() / members.size)
            row["in_abs_weight_sum"] = float(
                np.abs(weights[members, :]).sum() / members.size
            )
            row["out_weight_sum"] = float(weights[:, members].sum() / members.size)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregated_pair_weight_matrix(model, group_members: Mapping) -> pd.DataFrame:
    """Effective weights aggregated to group pairs (post in rows, pre in cols).

    Each entry is the summed effective weight from all of ``pre``'s members
    onto all of ``post``'s members, divided by the number of ``post`` members —
    the mean total input a ``post`` neuron receives from the ``pre`` group.

    Args:
        model: the network.
        group_members (Mapping): ``{group: sequence of node indices}``.

    Returns:
        pd.DataFrame: square frame indexed by group.
    """
    weights = model.effective_weights.to_dense().detach().cpu().numpy()
    groups = list(group_members)
    members = {
        group: np.asarray(list(group_members[group]), dtype=int) for group in groups
    }
    matrix = np.zeros((len(groups), len(groups)), dtype=np.float64)
    for i, post in enumerate(groups):
        post_idx = members[post]
        for j, pre in enumerate(groups):
            block = weights[np.ix_(post_idx, members[pre])]
            matrix[i, j] = block.sum() / post_idx.size
    return pd.DataFrame(matrix, index=groups, columns=groups)


def to_jsonable(value):
    """Recursively convert a value into JSON-serialisable builtins.

    Handles numpy scalars/arrays, :class:`~pathlib.Path`, ``torch.device``,
    dicts (keys stringified) and lists/tuples; everything else passes through
    unchanged. Intended for run-metadata dumps.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, torch.device):
        return str(value)
    return value
