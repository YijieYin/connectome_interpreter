"""Tests for connectome_interpreter.response_fitting.

Covers the dynamics and stability helpers, the model-carried activation
gain, the ExponentialSensor observation model, the affine-readout loss
factory, the window/metric helpers and the save/load/rebuild round-trip.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
import torch

import connectome_interpreter as cin
from connectome_interpreter import response_fitting as rf

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Fixtures: tiny networks
# ---------------------------------------------------------------------------


def tanh_relu_activation(self, x, x_previous=None):
    """The default tanh(thresholded-relu) transfer, re-implemented as a custom
    activation so the custom-dispatch path can be tested against the built-in
    reference (same math, same leaky-tau integration)."""
    if self.slope_is_pairwise:
        slopes = 1.0
    elif self.slope is None:
        slopes = self.tanh_steepness
    else:
        slopes = (
            self.slope[self.indices].view(-1, 1).expand(-1, x.shape[1]).to(x.device)
        )
    if self.biases is None:
        biases = self.default_bias
    else:
        biases = (
            self.biases[self.indices].view(-1, 1).expand(-1, x.shape[1]).to(x.device)
        )
    z = slopes * x + biases
    z = torch.relu(z - self.threshold) + self.threshold * (z >= self.threshold).to(
        z.dtype
    )
    if self.tau_param is not None:
        taus = (
            self.effective_tau[self.indices]
            .view(-1, 1)
            .expand(-1, x.shape[1])
            .to(x.device)
        )
    else:
        taus = self.tau
    return 1 / taus * torch.tanh(z) + (taus - 1) / taus * x_previous


def tanh_relu_gain(self, state):
    """Matching local gain: ``(1 - tanh(u)^2) * 1[u >= threshold]`` (pair-mode
    slopes are ones, folded into the effective weights)."""
    weights = self.effective_weights
    state = torch.as_tensor(state, dtype=torch.float32, device=weights.device).reshape(
        -1, 1
    )
    u = torch.sparse.mm(weights, state).squeeze(1) + self.node_parameter("bias")
    gate = (u >= self.threshold).to(u.dtype)
    return (1.0 - torch.tanh(u) ** 2) * gate


def _input_tensor(trace, device=CPU):
    """A 1-D stimulus trace as a ``(1, 1, T)`` model input."""
    return torch.as_tensor(np.asarray(trace, dtype=np.float32), device=device)[
        None, None, :
    ]


def _dense_update(net, state=None):
    """Dense free-block update Jacobian, built independently of
    ``free_update_matrix`` (which refuses networks above 256 nodes)."""
    W = net.effective_weights.to_dense()
    taus = net.node_parameter("tau")
    gain = net.activation_gain(state)
    update = torch.diag((taus - 1.0) / taus) + (gain / taus).view(-1, 1) * W
    free_idx, _ = rf._free_and_sensory_indices(net)
    return update.index_select(0, free_idx).index_select(1, free_idx)


def _gershgorin_bound(net, state=None):
    free_idx, _ = rf._free_and_sensory_indices(net)
    return rf._gershgorin_spectral_radius_bound(
        net, free_idx, net.activation_gain(state)
    )


def _linear_net(weights, **kwargs):
    return cin.LinearNetwork(
        all_weights=sps.csr_matrix(np.asarray(weights, dtype=np.float32)),
        sensory_indices=[0],
        num_layers=5,
        device=CPU,
        **kwargs,
    )


def _stable_linear_net():
    return _linear_net([[0.0, 0.0, 0.0], [0.10, 0.0, 0.20], [0.0, 0.15, 0.0]])


_MLN_WEIGHTS = np.array(
    [[0.0, 0.0, 0.0], [0.4, 0.0, 0.2], [0.0, 0.3, 0.0]], dtype=np.float32
)


def _mln_pairwise(activation_function=None, activation_gain_fn=None, **kwargs):
    """3-node MultilayeredNetwork, pair-mode slopes, clamped sensory node 0."""
    return cin.MultilayeredNetwork(
        sps.csr_matrix(_MLN_WEIGHTS),
        sensory_indices=[0],
        num_layers=kwargs.pop("num_layers", 6),
        threshold=0.0,
        tanh_steepness=1.0,
        idx_to_group={0: "S", 1: "A", 2: "B"},
        bias_dict={"S": 0.0, "A": 0.05, "B": 0.02},
        bias_transform="identity",
        slope_dict={("S", "A"): 1.0, ("A", "B"): 1.0, ("B", "A"): 1.0},
        tau=1.0,
        tau_dict={"S": 1.0, "A": 2.0, "B": 3.0},
        sensory_input_mode="replace",
        output_clamp_max=None,
        output_rectify=True,
        activation_function=activation_function,
        activation_gain_fn=activation_gain_fn,
        device=CPU,
        **kwargs,
    )


def _mln_node_mode():
    """Node-mode MultilayeredNetwork with the built-in tanh activation."""
    return cin.MultilayeredNetwork(
        sps.csr_matrix(_MLN_WEIGHTS),
        sensory_indices=[0],
        num_layers=6,
        threshold=0.0,
        idx_to_group={0: "S", 1: "A", 2: "B"},
        slope_dict={"S": 1.0, "A": 1.5, "B": 2.0},
        bias_dict={"S": 0.0, "A": 0.1, "B": 0.2},
        bias_transform="identity",
        tau=1.0,
        tau_dict={"S": 1.0, "A": 2.0, "B": 3.0},
        sensory_input_mode="replace",
        device=CPU,
    )


def _big_linear_chain(n=300, weight=0.5):
    """A > _DENSE_NODE_LIMIT chain network to exercise the sparse paths."""
    rows = np.arange(1, n)
    cols = np.arange(0, n - 1)
    vals = np.full(n - 1, weight, dtype=np.float32)
    weights = sps.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return cin.LinearNetwork(
        all_weights=weights,
        sensory_indices=[0],
        num_layers=3,
        tanh_steepness=1.0,
        device=CPU,
    )


def _big_pairwise_linear_chain(n=300, weight=0.4):
    """A > _DENSE_NODE_LIMIT chain LinearNetwork with pair-mode slopes and
    trainable node biases, to exercise the *differentiable* sparse steady state.
    Node 0 is sensory ("S"); every other node shares group "A", so the chain
    edges are the ``(A, A)`` pair plus the single ``(S, A)`` input edge."""
    rows = np.arange(1, n)
    cols = np.arange(0, n - 1)
    vals = np.full(n - 1, weight, dtype=np.float32)
    weights = sps.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    idx_to_group = {0: "S", **{i: "A" for i in range(1, n)}}
    model = cin.LinearNetwork(
        all_weights=weights,
        sensory_indices=[0],
        num_layers=3,
        threshold=0.0,
        tanh_steepness=1.0,
        idx_to_group=idx_to_group,
        bias_dict={"S": 0.0, "A": 0.1},
        bias_transform="identity",
        slope_dict={("S", "A"): 1.0, ("A", "A"): 1.0},
        tau=1.0,
        tau_dict={"S": 1.0, "A": 2.0},
        sensory_input_mode="replace",
        device=CPU,
    )
    model.slope.requires_grad_(True)
    model.biases.requires_grad_(True)
    return model


# ---------------------------------------------------------------------------
# node_parameter / activation_gain (library-side additions)
# ---------------------------------------------------------------------------


class TestNodeParameter:
    def test_pairwise_slope_is_ones(self):
        model = _mln_pairwise()
        assert torch.equal(model.node_parameter("slope"), torch.ones(3))

    def test_node_mode_values(self):
        model = _mln_node_mode()
        np.testing.assert_allclose(
            model.node_parameter("slope").detach().numpy(), [1.0, 1.5, 2.0]
        )
        np.testing.assert_allclose(
            model.node_parameter("bias").detach().numpy(), [0.0, 0.1, 0.2]
        )
        np.testing.assert_allclose(
            model.node_parameter("tau").detach().numpy(), [1.0, 2.0, 3.0]
        )

    def test_defaults_without_groups(self):
        model = _linear_net(np.zeros((3, 3)))
        # no idx_to_group: model's own defaults fill in
        np.testing.assert_allclose(
            model.node_parameter("slope").detach().numpy(), [5.0] * 3
        )
        np.testing.assert_allclose(
            model.node_parameter("bias").detach().numpy(), [0.0] * 3
        )
        np.testing.assert_allclose(
            model.node_parameter("tau").detach().numpy(), [10.0] * 3
        )

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown parameter name"):
            _mln_pairwise().node_parameter("weights")


class TestActivationGain:
    def test_linear_gain_is_slope_and_state_independent(self):
        model = _stable_linear_net()
        gain = model.activation_gain()
        np.testing.assert_allclose(gain.detach().numpy(), [5.0] * 3)

    def test_custom_activation_without_gain_raises(self):
        model = _mln_pairwise(activation_function=tanh_relu_activation)
        with pytest.raises(ValueError, match="activation_gain_fn"):
            model.activation_gain(torch.zeros(3))

    def test_custom_gain_dispatched(self):
        model = _mln_pairwise(
            activation_function=tanh_relu_activation, activation_gain_fn=tanh_relu_gain
        )
        state = torch.tensor([0.5, 0.1, 0.2])
        expected = tanh_relu_gain(model, state)
        np.testing.assert_allclose(
            model.activation_gain(state).detach().numpy(), expected.detach().numpy()
        )
        # the pair re-implements the default activation, so the dispatched
        # gain must agree with the built-in analytic Jacobian
        builtin = _mln_pairwise().activation_gain(state)
        np.testing.assert_allclose(
            model.activation_gain(state).detach().numpy(),
            builtin.detach().numpy(),
            rtol=1e-6,
        )

    def test_custom_pair_matches_builtin_forward(self):
        inputs = _input_tensor(
            np.linspace(0.1, 0.5, 6).astype(np.float32), device=CPU
        )
        custom = _mln_pairwise(
            activation_function=tanh_relu_activation, activation_gain_fn=tanh_relu_gain
        )
        default = _mln_pairwise()
        with torch.no_grad():
            np.testing.assert_allclose(
                np.asarray(custom(inputs, checkpoint_steps=0)),
                np.asarray(default(inputs, checkpoint_steps=0)),
                atol=1e-6,
            )

    def test_builtin_multilayered_gain_matches_analytic(self):
        model = _mln_node_mode()
        state = torch.tensor([0.5, 0.1, 0.2])
        gain = model.activation_gain(state).detach().numpy()
        weights = _MLN_WEIGHTS
        slopes = np.array([1.0, 1.5, 2.0])
        biases = np.array([0.0, 0.1, 0.2])
        u = slopes * (weights @ state.numpy()) + biases
        expected = (1.0 - np.tanh(u) ** 2) * slopes * (u >= 0.0)
        np.testing.assert_allclose(gain, expected, rtol=1e-5)

    def test_builtin_multilayered_gain_requires_state(self):
        with pytest.raises(ValueError, match="state-dependent"):
            _mln_node_mode().activation_gain()

    def test_divisive_normalization_gain_not_implemented(self):
        # divisive edges must be negative (inhibitory) at construction
        weights = np.array(
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.2], [0.0, -0.3, 0.0]], dtype=np.float32
        )
        model = cin.MultilayeredNetwork(
            sps.csr_matrix(weights),
            sensory_indices=[0],
            num_layers=3,
            idx_to_group={0: "S", 1: "A", 2: "B"},
            divisive_normalization={"A": ["B"]},
            sensory_input_mode="replace",
            device=CPU,
        )
        with pytest.raises(NotImplementedError, match="divisive_normalization"):
            model.activation_gain(torch.zeros(3))
        linear = _linear_net(
            weights,
            idx_to_group={0: "S", 1: "A", 2: "B"},
            divisive_normalization={"A": ["B"]},
        )
        with pytest.raises(NotImplementedError, match="divisive_normalization"):
            linear.activation_gain()


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


class TestDynamics:
    def test_free_and_sensory_split(self):
        free_idx, sensory_idx = rf._free_and_sensory_indices(_stable_linear_net())
        assert sensory_idx.tolist() == [0]
        assert set(free_idx.tolist()) == {1, 2}

    def test_steady_state_solves_free_update_fixed_point(self):
        net = _stable_linear_net()
        steady = rf.linear_network_steady_state(net, [1.0])
        assert bool(torch.isfinite(steady).all())
        free_idx, sensory_idx = rf._free_and_sensory_indices(net)
        assert torch.isclose(steady[sensory_idx[0]], torch.tensor(1.0))
        update = rf.free_update_matrix(net)
        assert update.shape == (free_idx.numel(), free_idx.numel())
        assert torch.isfinite(update).all()

    def test_steady_state_sparse_path_matches_recursion(self):
        net = _big_linear_chain(n=300, weight=0.5)
        steady = rf.linear_network_steady_state(net, [1.0]).detach().numpy()
        # chain with slope 1: x_i = 0.5 * x_{i-1}, x_0 clamped to 1
        np.testing.assert_allclose(steady[:6], 0.5 ** np.arange(6), atol=1e-5)

    def test_network_step_requires_replace_mode(self):
        model = cin.MultilayeredNetwork(
            sps.csr_matrix(_MLN_WEIGHTS),
            sensory_indices=[0],
            num_layers=3,
            sensory_input_mode="add",
            device=CPU,
        )
        with pytest.raises(ValueError, match="replace"):
            rf.network_step(model, torch.zeros(3), [0.1])

    def test_fixed_point_converges_and_is_a_fixed_point(self):
        model = _mln_pairwise(
            activation_function=tanh_relu_activation, activation_gain_fn=tanh_relu_gain
        )
        state, info = rf.network_fixed_point(model, [0.3], return_info=True, tol=1e-8)
        assert info["converged"]
        next_state = rf.network_step(model, state, [0.3])
        assert float(torch.abs(next_state - state).max()) <= 1e-6

    def test_fixed_point_warm_start_converges_quickly(self):
        model = _mln_pairwise(
            activation_function=tanh_relu_activation, activation_gain_fn=tanh_relu_gain
        )
        state = rf.network_fixed_point(model, [0.3], tol=1e-10)
        _, info = rf.network_fixed_point(
            model, [0.3], initial_state=state, min_steps=1, tol=1e-8, return_info=True
        )
        assert info["converged"]
        assert info["iterations"] <= 3


class TestMakeInitialStateFn:
    def test_wraps_steady_state_solver(self):
        from functools import partial

        net = _stable_linear_net()
        fn = rf.make_initial_state_fn(
            partial(rf.linear_network_steady_state, sensory_values=[0.3])
        )
        np.testing.assert_allclose(
            fn(net).numpy(),
            rf.linear_network_steady_state(net, [0.3]).detach().numpy(),
        )

    def test_detach_true_strips_grad(self):
        fn = rf.make_initial_state_fn(
            lambda model, **kw: torch.zeros(3, requires_grad=True) + 1.0
        )
        assert fn(None).requires_grad is False

    def test_detach_false_keeps_grad(self):
        fn = rf.make_initial_state_fn(
            lambda model, **kw: torch.zeros(3, requires_grad=True) + 1.0,
            detach=False,
        )
        assert fn(None).requires_grad is True

    def test_warm_start_threads_previous_state(self):
        received = []

        def solver(model, **kwargs):
            received.append(kwargs)
            return torch.full((3,), float(len(received)))

        fn = rf.make_initial_state_fn(solver, warm_start_kw="initial_state")
        first = fn(None)
        second = fn(None)
        assert received[0] == {}
        assert torch.equal(received[1]["initial_state"], first)
        assert torch.equal(second, torch.full((3,), 2.0))


class TestNonlinearNetworkSteadyState:
    """The implicit-differentiation equilibrium solver (dense backend)."""

    U = [0.3]

    def _model(self):
        model = _mln_pairwise(
            activation_function=tanh_relu_activation,
            activation_gain_fn=tanh_relu_gain,
        )
        # train_model normally switches these on (train_slopes/biases/tau)
        model.slope.requires_grad_(True)
        model.biases.requires_grad_(True)
        model.tau_param.requires_grad_(True)
        return model

    @staticmethod
    def _loss(state):
        weights = torch.tensor([0.7, -1.3, 0.9])
        return (weights * state).sum()

    def test_is_fixed_point_and_matches_iterative(self):
        model = self._model()
        state = rf.nonlinear_network_steady_state(model, self.U)
        moved = (
            (rf.network_step(model, state.detach(), self.U) - state.detach())
            .abs()
            .max()
        )
        assert float(moved) <= 1e-5
        iterative = rf.network_fixed_point(
            model, self.U, max_steps=20000, min_steps=20, tol=1e-7
        )
        np.testing.assert_allclose(
            state.detach().numpy(), iterative.numpy(), atol=1e-5
        )

    def test_gradient_matches_finite_differences(self):
        model = self._model()
        grads = torch.autograd.grad(
            self._loss(rf.nonlinear_network_steady_state(model, self.U)),
            [model.slope, model.biases],
        )
        h = 1e-3
        for param, analytic in zip([model.slope, model.biases], grads):
            for i in range(param.numel()):
                values = []
                for sign in (+1.0, -1.0):
                    with torch.no_grad():
                        param.data[i] += sign * h
                        values.append(
                            float(
                                self._loss(
                                    rf.nonlinear_network_steady_state(model, self.U)
                                )
                            )
                        )
                        param.data[i] -= sign * h
                fd = (values[0] - values[1]) / (2 * h)
                np.testing.assert_allclose(
                    float(analytic[i]), fd, rtol=5e-2, atol=2e-3
                )

    def test_tau_gradient_vanishes(self):
        # The equilibrium condition has no tau (the leak cancels at a fixed
        # point), so the implicit gradient wrt tau is zero up to the solve
        # residual.
        model = self._model()
        (tau_grad,) = torch.autograd.grad(
            self._loss(rf.nonlinear_network_steady_state(model, self.U)),
            [model.tau_param],
            allow_unused=False,
        )
        assert float(tau_grad.abs().max()) < 1e-4

    def test_matches_linear_closed_form(self):
        net = _linear_net(
            [[0.0, 0.0, 0.0], [0.10, 0.0, 0.20], [0.0, 0.15, 0.0]],
            sensory_input_mode="replace",
        )
        implicit = rf.nonlinear_network_steady_state(net, self.U)
        closed_form = rf.linear_network_steady_state(net, self.U)
        np.testing.assert_allclose(
            implicit.detach().numpy(),
            closed_form.detach().numpy(),
            atol=1e-5,
        )

    def test_warm_start_accepted(self):
        model = self._model()
        first = rf.nonlinear_network_steady_state(model, self.U)
        second = rf.nonlinear_network_steady_state(
            model, self.U, initial_state=first.detach()
        )
        np.testing.assert_allclose(
            first.detach().numpy(), second.detach().numpy(), atol=1e-6
        )

    def test_raises_without_gain(self):
        model = _mln_pairwise(activation_function=tanh_relu_activation)
        with pytest.raises(ValueError, match="activation_gain_fn"):
            rf.nonlinear_network_steady_state(model, self.U)

    def test_raises_above_dense_limit(self):
        with pytest.raises(NotImplementedError, match="at most 256 nodes"):
            rf.nonlinear_network_steady_state(_big_linear_chain(), [0.3])

    def test_make_initial_state_fn_integration(self):
        from functools import partial

        model = self._model()
        solver = partial(rf.nonlinear_network_steady_state, sensory_values=self.U)
        attached = rf.make_initial_state_fn(
            solver, detach=False, warm_start_kw="initial_state"
        )(model)
        assert attached.requires_grad
        detached = rf.make_initial_state_fn(
            solver, detach=True, warm_start_kw="initial_state"
        )(model)
        assert not detached.requires_grad
        np.testing.assert_allclose(
            attached.detach().numpy(), detached.numpy(), atol=1e-6
        )


class TestLinearNetworkSteadyStateSparse:
    """The differentiable sparse (splu forward + IFT-adjoint backward) linear
    steady state used above ``_DENSE_NODE_LIMIT``. The dense path (autograd
    through ``torch.linalg.solve``) is the reference the sparse gradient must
    reproduce; the small-network tests force the sparse branch by shrinking the
    limit so both run on the same 3-node model."""

    U = [0.3]

    def _model(self):
        model = cin.LinearNetwork(
            sps.csr_matrix(_MLN_WEIGHTS),
            sensory_indices=[0],
            num_layers=5,
            threshold=0.0,
            tanh_steepness=1.0,
            idx_to_group={0: "S", 1: "A", 2: "B"},
            bias_dict={"S": 0.0, "A": 0.05, "B": 0.02},
            bias_transform="identity",
            slope_dict={("S", "A"): 1.0, ("A", "B"): 1.0, ("B", "A"): 1.0},
            tau=1.0,
            tau_dict={"S": 1.0, "A": 2.0, "B": 3.0},
            sensory_input_mode="replace",
            device=CPU,
        )
        model.slope.requires_grad_(True)
        model.biases.requires_grad_(True)
        if model.tau_param is not None:
            model.tau_param.requires_grad_(True)
        return model

    @staticmethod
    def _loss(state):
        weights = torch.tensor([0.7, -1.3, 0.9])
        return (weights * state).sum()

    def test_sparse_matches_dense_value(self, monkeypatch):
        model = self._model()
        dense = rf.linear_network_steady_state(model, self.U).detach().numpy()
        monkeypatch.setattr(rf, "_DENSE_NODE_LIMIT", 1)
        sparse_state = rf.linear_network_steady_state(model, self.U).detach().numpy()
        np.testing.assert_allclose(sparse_state, dense, atol=1e-6)

    def test_sparse_gradient_matches_dense(self, monkeypatch):
        model = self._model()
        dense_grads = torch.autograd.grad(
            self._loss(rf.linear_network_steady_state(model, self.U)),
            [model.slope, model.biases],
        )
        monkeypatch.setattr(rf, "_DENSE_NODE_LIMIT", 1)
        sparse_grads = torch.autograd.grad(
            self._loss(rf.linear_network_steady_state(model, self.U)),
            [model.slope, model.biases],
        )
        for dense_g, sparse_g in zip(dense_grads, sparse_grads):
            np.testing.assert_allclose(
                sparse_g.numpy(), dense_g.numpy(), rtol=1e-4, atol=1e-6
            )

    def test_gradient_matches_finite_differences(self, monkeypatch):
        model = self._model()
        monkeypatch.setattr(rf, "_DENSE_NODE_LIMIT", 1)
        grads = torch.autograd.grad(
            self._loss(rf.linear_network_steady_state(model, self.U)),
            [model.slope, model.biases],
        )
        h = 1e-3
        for param, analytic in zip([model.slope, model.biases], grads):
            for i in range(param.numel()):
                values = []
                for sign in (+1.0, -1.0):
                    with torch.no_grad():
                        param.data[i] += sign * h
                        values.append(
                            float(
                                self._loss(
                                    rf.linear_network_steady_state(model, self.U)
                                )
                            )
                        )
                        param.data[i] -= sign * h
                fd = (values[0] - values[1]) / (2 * h)
                np.testing.assert_allclose(
                    float(analytic[i]), fd, rtol=5e-2, atol=2e-3
                )

    def test_tau_gradient_is_none(self, monkeypatch):
        # The linear equilibrium does not involve tau, so the sparse path (like
        # the dense one) yields no gradient for the taus.
        model = self._model()
        monkeypatch.setattr(rf, "_DENSE_NODE_LIMIT", 1)
        (tau_grad,) = torch.autograd.grad(
            self._loss(rf.linear_network_steady_state(model, self.U)),
            [model.tau_param],
            allow_unused=True,
        )
        assert tau_grad is None or float(tau_grad.abs().max()) < 1e-6

    def test_large_network_dispatches_to_sparse_and_is_fixed_point(self):
        model = _big_pairwise_linear_chain()
        assert model.all_weights.shape[0] > rf._DENSE_NODE_LIMIT
        state = rf.linear_network_steady_state(model, self.U)
        moved = (
            (rf.network_step(model, state.detach(), self.U) - state.detach())
            .abs()
            .max()
        )
        assert float(moved) <= 1e-5
        iterative = rf.network_fixed_point(
            model, self.U, max_steps=20000, min_steps=20, tol=1e-8
        )
        np.testing.assert_allclose(
            state.detach().numpy(), iterative.numpy(), atol=1e-5
        )

    def test_large_network_is_differentiable(self):
        model = _big_pairwise_linear_chain()
        state = rf.linear_network_steady_state(model, self.U)
        assert state.requires_grad
        slope_grad, bias_grad = torch.autograd.grad(
            state.sum(), [model.slope, model.biases]
        )
        assert torch.isfinite(slope_grad).all()
        assert float(slope_grad.abs().max()) > 0
        assert torch.isfinite(bias_grad).all()
        assert float(bias_grad.abs().max()) > 0

    def test_make_initial_state_fn_integration_large(self):
        from functools import partial

        model = _big_pairwise_linear_chain()
        solver = partial(rf.linear_network_steady_state, sensory_values=self.U)
        attached = rf.make_initial_state_fn(solver, detach=False)(model)
        assert attached.requires_grad
        detached = rf.make_initial_state_fn(solver, detach=True)(model)
        assert not detached.requires_grad
        np.testing.assert_allclose(
            attached.detach().numpy(), detached.numpy(), atol=1e-6
        )


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


class TestStability:
    def test_spectral_radius_and_penalty_agree_for_stable_network(self):
        net = _stable_linear_net()
        rho = rf.spectral_radius(net)
        assert np.isfinite(rho) and rho < 1.0
        assert float(rf.stability_penalty(net, margin=1e-4)) == 0.0

    def test_penalty_positive_for_unstable_network(self):
        # a strong self-excitatory free node pushes the update radius above 1
        net = _linear_net([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
        assert rf.spectral_radius(net) > 1.0
        assert float(rf.stability_penalty(net)) > 0.0

    def test_free_update_matrix_matches_linear_formula(self):
        net = _stable_linear_net()
        update = rf.free_update_matrix(net).detach().numpy()
        weights = np.array(
            [[0.0, 0.0, 0.0], [0.10, 0.0, 0.20], [0.0, 0.15, 0.0]], dtype=np.float64
        )
        slopes, taus = 5.0, 10.0
        full = np.diag([(taus - 1.0) / taus] * 3) + (slopes / taus) * weights
        np.testing.assert_allclose(update, full[1:, 1:], rtol=1e-6)

    def test_nonlinear_penalty_uses_state_dependent_gain(self):
        model = _mln_pairwise(
            activation_function=tanh_relu_activation, activation_gain_fn=tanh_relu_gain
        )
        # gain = 1 - tanh(u)^2: saturation lowers the local gain, so a
        # high-activity operating point is more stable than a low one
        low = rf.spectral_radius(model, torch.zeros(3))
        high = rf.spectral_radius(model, torch.tensor([5.0, 5.0, 5.0]))
        assert high < low

    def test_penalty_raises_without_gain(self):
        model = _mln_pairwise(activation_function=tanh_relu_activation)
        with pytest.raises(ValueError, match="activation_gain_fn"):
            rf.stability_penalty(model, torch.zeros(3))

    def test_sparse_paths_run_on_large_network(self):
        # random sparse net: generic spectrum (the degenerate chain defeats
        # ARPACK), small weights so even the Gershgorin bound stays below 1
        n = 300
        rng = np.random.RandomState(0)
        nnz = 4 * n
        weights = sps.coo_matrix(
            (
                rng.uniform(-0.1, 0.1, nnz).astype(np.float32),
                (rng.randint(1, n, nnz), rng.randint(0, n, nnz)),
            ),
            shape=(n, n),
        ).tocsr()
        net = cin.LinearNetwork(
            all_weights=weights,
            sensory_indices=[0],
            num_layers=3,
            tanh_steepness=1.0,
            device=CPU,
        )
        rho = rf.spectral_radius(net)  # ARPACK path (n > _DENSE_NODE_LIMIT)
        reference = float(torch.abs(torch.linalg.eigvals(_dense_update(net))).max())
        assert abs(rho - reference) < 1e-4
        assert rho < 1.0
        assert float(rf.stability_penalty(net)) == 0.0  # Gershgorin path
        bound = float(_gershgorin_bound(net))
        assert rho <= bound + 1e-6  # Gershgorin upper-bounds the true radius
        assert bound < 1.0

    def test_gershgorin_bound_small_net(self):
        # bound >= true radius on the dense-regime nets too
        net = _stable_linear_net()
        assert rf.spectral_radius(net) <= float(_gershgorin_bound(net)) + 1e-6

    def test_free_update_matrix_refuses_large_network(self):
        with pytest.raises(ValueError, match="at most 256 nodes"):
            rf.free_update_matrix(_big_linear_chain())

    def test_penalty_is_differentiable(self):
        # pair-mode gain enters the Jacobian through effective_weights; at the
        # zero state the built-in tanh gain is 1, so the strong self-edge
        # (w_eff = 2 * 2 = 4) makes the update radius 0.9 + 4/10 > 1
        model = cin.MultilayeredNetwork(
            sps.csr_matrix(np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)),
            sensory_indices=[0],
            num_layers=3,
            threshold=0.0,
            idx_to_group={0: "S", 1: "A"},
            slope_dict={("A", "A"): 2.0},
            sensory_input_mode="replace",
            device=CPU,
        )
        model.slope.requires_grad_(True)
        state = torch.zeros(2)
        penalty = rf.stability_penalty(model, state)
        assert float(penalty) > 0.0
        penalty.backward()
        assert model.slope.grad is not None
        assert torch.isfinite(model.slope.grad).all()


# ---------------------------------------------------------------------------
# ExponentialSensor
# ---------------------------------------------------------------------------


class TestExponentialSensor:
    def test_kernel_normalised_and_decaying(self):
        sensor = rf.ExponentialSensor(tau_ms=300.0, dt_ms=1.0)
        kernel = sensor.kernel
        assert kernel.dtype == np.float32
        assert abs(kernel.sum() - 1.0) < 1e-5  # unit DC gain -> plateaus preserved
        assert np.all(np.diff(kernel) < 0)  # monotonically decaying
        assert kernel.size == int(round(6.0 * 300.0))  # support_tau * tau / dt

    def test_kernel_tap_ratio_matches_tau(self):
        sensor = rf.ExponentialSensor(tau_ms=250.0, dt_ms=1.0)
        np.testing.assert_allclose(
            sensor.kernel[1] / sensor.kernel[0], np.exp(-1.0 / 250.0), rtol=1e-5
        )

    def test_apply_constant_preserved(self):
        sensor = rf.ExponentialSensor(tau_ms=300.0, dt_ms=1.0)
        const = np.full(5000, 0.37)
        np.testing.assert_allclose(sensor.apply(const), const, atol=1e-6)

    def test_apply_impulse_gives_kernel(self):
        sensor = rf.ExponentialSensor(tau_ms=200.0, dt_ms=1.0)
        x = np.zeros(4000)
        x[1000] = 1.0
        y = sensor.apply(x)
        np.testing.assert_allclose(
            y[1000 : 1000 + sensor.kernel.size], sensor.kernel, atol=1e-6
        )
        assert np.allclose(y[:1000], 0.0)  # causal: nothing before the impulse

    def test_apply_causal_step_smoothed(self):
        sensor = rf.ExponentialSensor(tau_ms=300.0, dt_ms=1.0)
        x = np.concatenate([np.zeros(3000), np.ones(6000)])
        y = sensor.apply(x)
        assert np.allclose(y[:3000], 0.0)
        assert 0.0 < y[3050] < 1.0
        assert abs(y[-1] - 1.0) < 1e-3

    def test_output_transform_matches_apply(self):
        sensor = rf.ExponentialSensor(tau_ms=300.0, dt_ms=1.0)
        rng = np.random.RandomState(0)
        x = rng.randn(2, 3, 2000).astype(np.float32)
        y = sensor.output_transform(torch.as_tensor(x)).numpy()
        for b in range(2):
            for n in range(3):
                np.testing.assert_allclose(
                    y[b, n], sensor.apply(x[b, n]), rtol=1e-4, atol=1e-4
                )

    def test_output_transform_shape_and_differentiable(self):
        sensor = rf.ExponentialSensor(tau_ms=300.0, dt_ms=1.0)
        x = torch.randn(1, 4, 1500, requires_grad=True)
        y = sensor.output_transform(x)
        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_output_transform_rejects_non_3d(self):
        sensor = rf.ExponentialSensor(tau_ms=10.0, dt_ms=1.0)
        with pytest.raises(ValueError, match="batch, neurons, T"):
            sensor.output_transform(torch.zeros(5, 5))

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            rf.ExponentialSensor(tau_ms=0.0)
        with pytest.raises(ValueError):
            rf.ExponentialSensor(tau_ms=1.0, dt_ms=0.0)


# ---------------------------------------------------------------------------
# Affine readout & loss factory
# ---------------------------------------------------------------------------


class TestAffineReadout:
    def test_solve_recovers_affine_relation(self):
        latent = torch.linspace(-1, 1, 100)
        target = 2.5 * latent + 0.3
        scale, offset = rf.affine_readout_solve(latent, target, ridge=0.0)
        assert abs(float(scale) - 2.5) < 1e-5
        assert abs(float(offset) - 0.3) < 1e-5

    def test_solve_clamps_scale_nonnegative(self):
        latent = torch.linspace(-1, 1, 100)
        target = -2.0 * latent
        scale, offset = rf.affine_readout_solve(latent, target)
        assert float(scale) == 0.0

    def test_penalties_zero_when_comfortable(self):
        scale_pen, std_pen = rf._affine_readout_penalties(
            torch.tensor([0.0]), torch.tensor([1.0])
        )
        assert float(scale_pen) == 0.0
        assert float(std_pen) == 0.0

    def test_penalties_active_branches_match_formulas(self):
        scale_pen, std_pen = rf._affine_readout_penalties(
            torch.tensor([20.0]),
            torch.tensor([0.005]),
            scale_soft_limit=10.0,
            latent_std_floor=0.02,
        )
        # the scale penalty is a soft knee, nonzero for any nonzero scale
        assert float(scale_pen) == pytest.approx(np.log1p(2.0) ** 2, rel=1e-5)
        assert float(std_pen) == pytest.approx(0.015**2, rel=1e-5)

    def test_frame_schema_and_values(self):
        latent = {"L1": np.linspace(0, 1, 50)}
        target = {"L1": 3.0 * np.linspace(0, 1, 50) + 1.0}
        frame, by_layer = rf.affine_readout_frame(latent, target, ridge=0.0)
        assert list(frame.columns) == [
            "layer",
            "readout_offset",
            "readout_scale",
            "latent_mean",
            "target_mean",
            "latent_std",
            "target_std",
        ]
        scale, offset = by_layer["L1"]
        assert abs(scale - 3.0) < 1e-4
        assert abs(offset - 1.0) < 1e-4


def _toy_targets():
    """Targets over two windows for channels with neuron_idx 1 and 2."""
    rng = np.random.RandomState(1)
    traces = {"A": rng.randn(20).astype(np.float32), "B": rng.randn(20).astype(np.float32)}
    transition_idx = [2, 10]
    window = 4
    targets = rf.build_raw_window_targets(
        traces, transition_idx, window, node_index={"A": 1, "B": 2}
    )
    windows = rf.extract_transition_windows(traces, transition_idx, window)
    return traces, targets, windows


def _flat_target_vector(targets):
    """Replicate train_model's flat target order for hand-built loss inputs."""
    order = targets[targets["batch"].isin([0])].copy()
    order.loc[:, ["batch"]] = pd.Categorical(order["batch"], categories=[0])
    order = order.sort_values(by="batch")
    return torch.as_tensor(order["value"].to_numpy(), dtype=torch.float32)


class TestMakeAffineReadoutLoss:
    def test_zero_loss_for_affinely_related_prediction(self):
        model = _mln_pairwise()
        _, targets, windows = _toy_targets()
        loss_fn = rf.make_affine_readout_loss(
            model,
            targets,
            layer_ids=[1, 2],
            expected_layer_windows={1: windows["A"], 2: windows["B"]},
            ridge=0.0,
            scale_reg_weight=0.0,
            std_reg_weight=0.0,
        )
        target = _flat_target_vector(targets)
        pred = 2.0 * target + 3.0  # affinely related -> perfect readout fit
        assert float(loss_fn(pred, target)) < 1e-10

    def test_moment_check_catches_wrong_grouping(self):
        model = _mln_pairwise()
        _, targets, windows = _toy_targets()
        loss_fn = rf.make_affine_readout_loss(
            model,
            targets,
            layer_ids=[1, 2],
            # swapped windows: the one-time moment assertion must fire
            expected_layer_windows={1: windows["B"], 2: windows["A"]},
        )
        target = _flat_target_vector(targets)
        with pytest.raises(ValueError, match="grouping does not match"):
            loss_fn(target.clone(), target)

    def test_normalize_by_target_var_reweights_channels(self):
        model = _mln_pairwise()
        _, targets, _ = _toy_targets()
        kwargs = dict(layer_ids=[1, 2], scale_reg_weight=0.0, std_reg_weight=0.0)
        plain_fn = rf.make_affine_readout_loss(model, targets, **kwargs)
        normed_fn = rf.make_affine_readout_loss(
            model, targets, normalize_by_target_var=True, **kwargs
        )
        order = targets[targets["batch"].isin([0])].copy()
        order.loc[:, ["batch"]] = pd.Categorical(order["batch"], categories=[0])
        order = order.sort_values(by="batch")
        nid = torch.as_tensor(order["neuron_idx"].to_numpy().astype(int))
        target = _flat_target_vector(targets)
        # near-perfect on channel 1, uninformative (constant) on channel 2 ->
        # channel MSEs differ, so the inverse-variance weighting must matter
        pred = torch.where(nid == 1, target, torch.zeros_like(target))

        mses, tvars = [], []
        for lid in (1, 2):
            m = nid == lid
            s, o = rf.affine_readout_solve(pred[m], target[m])
            mses.append(((s * pred[m] + o - target[m]) ** 2).mean())
            tvars.append(target[m].var(unbiased=False))
        mses, tvars = torch.stack(mses), torch.stack(tvars)

        assert torch.allclose(plain_fn(pred, target), mses.mean(), atol=1e-7)
        expected = ((tvars.mean() / tvars) * mses).mean()
        got = normed_fn(pred, target)
        assert torch.allclose(got, expected, atol=1e-7)
        assert abs(float(got - plain_fn(pred, target))) > 1e-4

    def test_normalize_matches_plain_for_equal_variance_channels(self):
        model = _mln_pairwise()
        _, targets, _ = _toy_targets()
        # force both channels onto the same target values -> equal variance,
        # where the weighting must reduce to the plain channel mean
        vals = targets["value"].to_numpy().copy()
        vals[targets["neuron_idx"] == 2] = vals[targets["neuron_idx"] == 1]
        targets = targets.assign(value=vals)
        kwargs = dict(layer_ids=[1, 2], scale_reg_weight=0.0, std_reg_weight=0.0)
        plain_fn = rf.make_affine_readout_loss(model, targets, **kwargs)
        normed_fn = rf.make_affine_readout_loss(
            model, targets, normalize_by_target_var=True, **kwargs
        )
        target = _flat_target_vector(targets)
        rng = np.random.RandomState(0)
        pred = torch.as_tensor(rng.randn(len(target)).astype(np.float32))
        assert torch.allclose(normed_fn(pred, target), plain_fn(pred, target), atol=1e-7)

    def test_multi_batch_targets_rejected(self):
        model = _mln_pairwise()
        _, targets, _ = _toy_targets()
        shifted = targets.copy()
        shifted["batch"] = 1
        with pytest.raises(ValueError, match="single-batch"):
            rf.make_affine_readout_loss(
                model, pd.concat([targets, shifted]), layer_ids=[1, 2]
            )

    def test_unknown_layer_id_rejected(self):
        model = _mln_pairwise()
        _, targets, _ = _toy_targets()
        with pytest.raises(ValueError, match="no rows"):
            rf.make_affine_readout_loss(model, targets, layer_ids=[1, 7])

    def test_stability_optin_without_gain_raises_at_build(self):
        model = _mln_pairwise(activation_function=tanh_relu_activation)
        _, targets, _ = _toy_targets()
        with pytest.raises(ValueError, match="stability_weight=0"):
            rf.make_affine_readout_loss(
                model, targets, layer_ids=[1, 2], stability_weight=1.0
            )

    def test_stability_weight_zero_skips_gain_requirement(self):
        # same gain-less model: with the default weight 0 the factory builds
        # and the loss runs -- backward compatible
        model = _mln_pairwise(activation_function=tanh_relu_activation)
        _, targets, _ = _toy_targets()
        loss_fn = rf.make_affine_readout_loss(model, targets, layer_ids=[1, 2])
        target = _flat_target_vector(targets)
        assert torch.isfinite(loss_fn(target.clone(), target))

    def test_stability_term_added_and_warm_started(self):
        model = _mln_pairwise(
            activation_function=tanh_relu_activation, activation_gain_fn=tanh_relu_gain
        )
        _, targets, _ = _toy_targets()
        calls = []

        def state_fn(previous):
            calls.append(previous)
            return rf.network_fixed_point(model, [0.3], initial_state=previous)

        weight = 7.0
        loss_fn = rf.make_affine_readout_loss(
            model,
            targets,
            layer_ids=[1, 2],
            ridge=0.0,
            scale_reg_weight=0.0,
            std_reg_weight=0.0,
            stability_weight=weight,
            stability_state_fn=state_fn,
        )
        target = _flat_target_vector(targets)
        pred = 2.0 * target + 3.0  # readout part is exactly zero
        first = float(loss_fn(pred, target))
        state = rf.network_fixed_point(model, [0.3])
        expected = weight * float(rf.stability_penalty(model, state))
        assert abs(first - expected) < 1e-8
        loss_fn(pred, target)
        assert calls[0] is None and calls[1] is not None  # warm start passed on


# ---------------------------------------------------------------------------
# Windows & metrics
# ---------------------------------------------------------------------------


class TestWindowsAndMetrics:
    def test_score_mask(self):
        mask = rf.score_mask_for_transition_windows(20, [2, 10], 4)
        assert mask.sum() == 8
        assert mask[2:6].all() and mask[10:14].all()
        with pytest.raises(ValueError, match="exceeds"):
            rf.score_mask_for_transition_windows(10, [8], 4)
        with pytest.raises(ValueError, match="overlap"):
            rf.score_mask_for_transition_windows(20, [2, 4], 4)

    def test_build_raw_window_targets_both_conventions(self):
        traces = {"A": np.arange(20, dtype=np.float32)}
        # per-node convention: neuron_idx is a model row
        node = rf.build_raw_window_targets(traces, [2], 3, node_index={"A": 7})
        assert node["neuron_idx"].unique().tolist() == [7]
        # group convention: neuron_idx is a group id
        group = rf.build_raw_window_targets(traces, [2], 3, node_index={"A": 0})
        assert group["neuron_idx"].unique().tolist() == [0]
        assert node["value"].tolist() == [2.0, 3.0, 4.0]
        assert node["layer"].tolist() == [2, 3, 4]

    def test_extract_transition_windows_variants(self):
        trace = np.arange(20, dtype=np.float32)
        single = rf.extract_transition_windows(trace, [2, 10], 3)
        assert single.shape == (2, 3)
        np.testing.assert_allclose(single[1], [10, 11, 12])
        as_dict = rf.extract_transition_windows({"A": trace}, [2], 3)
        np.testing.assert_allclose(as_dict["A"][0], [2, 3, 4])
        as_frame = rf.extract_transition_windows(pd.DataFrame({"A": trace}), [2], 3)
        np.testing.assert_allclose(as_frame["A"][0], [2, 3, 4])

    def test_r2(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rf.r2(y, y) == pytest.approx(1.0)
        assert rf.r2(y, np.full(3, y.mean())) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Persistence & tables
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_load_rebuild_round_trip(self, tmp_path):
        model = _mln_pairwise(
            activation_function=tanh_relu_activation, activation_gain_fn=tanh_relu_gain
        )
        inputs = _input_tensor(
            np.linspace(0.1, 0.5, 6).astype(np.float32), device=CPU
        )
        with torch.no_grad():
            expected = model(inputs, checkpoint_steps=0)
        path = tmp_path / "fit.npz"
        rf.save_fit(path, model, arrays={"brightness": np.linspace(0.1, 0.5, 6)})
        fit = rf.load_fit(path)
        assert "brightness" in fit
        rebuilt = rf.rebuild_network(
            fit,
            activation_function=tanh_relu_activation,
            activation_gain_fn=tanh_relu_gain,
            device=CPU,
        )
        with torch.no_grad():
            actual = rebuilt(inputs, checkpoint_steps=0)
        np.testing.assert_allclose(
            np.asarray(actual.detach().cpu()),
            np.asarray(expected.detach().cpu()),
            atol=1e-5,
        )

    def test_rebuild_from_path_and_linear_class(self, tmp_path):
        model = _stable_linear_net()
        path = tmp_path / "fit.npz"
        rf.save_fit(path, model)
        rebuilt = rf.rebuild_network(path, device=CPU)
        assert isinstance(rebuilt, cin.LinearNetwork)
        np.testing.assert_allclose(
            rebuilt.node_parameter("slope").detach().numpy(),
            model.node_parameter("slope").detach().numpy(),
        )

    def test_tau_max_round_trips_through_save_fit(self, tmp_path):
        model = _mln_pairwise(tau_max=8000.0)
        path = tmp_path / "fit.npz"
        rf.save_fit(path, model)
        rebuilt = rf.rebuild_network(path, device=CPU)
        assert rebuilt.tau_max == pytest.approx(8000.0)

    def test_reserved_key_collision_raises(self, tmp_path):
        with pytest.raises(ValueError, match="reserved"):
            rf.save_fit(
                tmp_path / "fit.npz",
                _mln_pairwise(),
                arrays={"model_something": np.zeros(2)},
            )

    def test_rebuild_without_model_section_raises(self, tmp_path):
        path = tmp_path / "fit.npz"
        rf.save_fit(path, arrays={"trace": np.zeros(3)})
        with pytest.raises(ValueError, match="no model section"):
            rf.rebuild_network(path)


class TestTables:
    def test_simulate_trace_columns(self):
        model = _mln_pairwise(
            activation_function=tanh_relu_activation, activation_gain_fn=tanh_relu_gain
        )
        inputs = _input_tensor(
            np.full(6, 0.3, dtype=np.float32), device=CPU
        )
        frame = rf.simulate_trace(
            model, inputs, node_columns={"lum": 0, "A": 1, "AB": [1, 2]}
        )
        assert list(frame.columns) == ["lum", "A", "AB"]
        assert len(frame) == 6
        np.testing.assert_allclose(frame["lum"], 0.3, atol=1e-6)

    def test_pair_slope_table(self):
        model = _mln_pairwise()
        table = rf.pair_slope_table(model)
        assert set(zip(table["pre"], table["post"])) == {
            ("S", "A"),
            ("A", "B"),
            ("B", "A"),
        }
        np.testing.assert_allclose(table["pair_slope"], 1.0)
        assert rf.pair_slope_table(_mln_node_mode()).empty


# ---------------------------------------------------------------------------
# train_model integration
# ---------------------------------------------------------------------------


class TestTrainModelIntegration:
    def test_end_to_end_fit_with_loss_factory_and_sensor(self):
        torch.manual_seed(0)
        model = _mln_pairwise(
            activation_function=tanh_relu_activation,
            activation_gain_fn=tanh_relu_gain,
            num_layers=30,
        )
        brightness = np.full(30, 0.3, dtype=np.float32)
        brightness[10:] = 0.6
        inputs = _input_tensor(brightness, device=CPU)
        rng = np.random.RandomState(0)
        traces = {
            "A": rng.rand(30).astype(np.float32),
            "B": rng.rand(30).astype(np.float32),
        }
        transition_idx = [10]
        window = 8
        targets = rf.build_raw_window_targets(
            traces, transition_idx, window, node_index={"A": 1, "B": 2}
        )
        windows = rf.extract_transition_windows(traces, transition_idx, window)
        sensor = rf.ExponentialSensor(tau_ms=5.0, dt_ms=1.0)
        loss_fn = rf.make_affine_readout_loss(
            model,
            targets,
            layer_ids=[1, 2],
            expected_layer_windows={1: windows["A"], 2: windows["B"]},
            stability_weight=1.0,
            stability_state_fn=lambda prev: rf.network_fixed_point(
                model, [np.log(0.3 + 0.17)], initial_state=prev
            ),
        )
        trained_model, history, *_ = cin.train_model(
            model,
            inputs,
            targets,
            num_epochs=2,
            learning_rate=1e-3,
            train_fraction=1.0,
            checkpoint_steps=0,
            activation_loss_fn=loss_fn,
            output_transform=sensor.output_transform,
            wandb=False,
        )
        assert trained_model is model
        assert len(history["loss"]) == 2
        for value in history["loss"]:
            assert np.isfinite(value)
