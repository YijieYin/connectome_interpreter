"""train_model(output_transform=...) hook: applied to the model output before
the targets are gathered.

This is the hook the L123 fits use to push the rate model through a forward GCaMP
sensor convolution; the convolution itself lives in the analysis tree, so the
cases here use plain shape-preserving stand-ins."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from connectome_interpreter.activation_maximisation import (
    MultilayeredNetwork,
    train_model,
)


def _model():
    torch.manual_seed(0)
    w = np.random.RandomState(0).rand(6, 6)
    w = w / w.sum(axis=1, keepdims=True)
    groups = {i: f"n{i}" for i in range(6)}
    # bias_dict + tau_dict register trainable parameters so train_model's
    # optimizer has something to step (a bare network exposes none).
    return MultilayeredNetwork(
        csr_matrix(w),
        sensory_indices=[0, 1],
        num_layers=4,
        idx_to_group=groups,
        bias_dict={f"n{i}": 0.0 for i in range(6)},
        tau_dict={f"n{i}": 5.0 for i in range(6)},
    )


def _targets():
    # time-series targets on neuron 5 at every layer (uses the "layer" path).
    return pd.DataFrame(
        {
            "batch": [0, 0, 0, 0],
            "neuron_idx": [5, 5, 5, 5],
            "layer": [0, 1, 2, 3],
            "value": [0.2, 0.3, 0.4, 0.5],
        }
    )


def _inputs():
    return torch.zeros(1, 2, 4, dtype=torch.float32)


def _capture_first_pred(output_transform):
    grabbed = {}

    def loss_fn(pred, target):
        grabbed.setdefault("pred", pred.detach().clone())
        return ((pred - target) ** 2).mean()

    train_model(
        _model_fixed(),
        _inputs(),
        _targets(),
        num_epochs=1,
        train_fraction=1.0,
        param_reg_lambda=0.0,
        checkpoint_steps=0,
        activation_loss_fn=loss_fn,
        output_transform=output_transform,
        seed=0,
    )
    return grabbed["pred"]


_SHARED = _model()


def _model_fixed():
    # identical model each call, so the epoch-0 forward is deterministic.
    return copy.deepcopy(_SHARED)


def test_output_transform_applied_before_gather():
    """A shape-preserving scale on the output shows up in the gathered pred."""
    pred_none = _capture_first_pred(None)
    pred_scaled = _capture_first_pred(lambda o: o * 3.0)
    assert pred_none.shape == pred_scaled.shape == (4,)
    torch.testing.assert_close(pred_scaled, pred_none * 3.0, rtol=1e-5, atol=1e-6)


def test_output_transform_receives_3d_output():
    seen = {}

    def spy(o):
        seen["shape"] = tuple(o.shape)
        return o

    train_model(
        _model_fixed(),
        _inputs(),
        _targets(),
        num_epochs=1,
        train_fraction=1.0,
        param_reg_lambda=0.0,
        checkpoint_steps=0,
        output_transform=spy,
        seed=0,
    )
    assert seen["shape"] == (1, 6, 4)  # (batch, num_neurons, num_layers)


def test_non_callable_output_transform_rejected():
    with pytest.raises(TypeError):
        train_model(
            _model_fixed(),
            _inputs(),
            _targets(),
            num_epochs=1,
            output_transform=123,
            seed=0,
        )


def test_output_transform_trains_over_multiple_epochs():
    """A stateful, shape-preserving transform keeps the loss finite while training."""
    hist = train_model(
        _model_fixed(),
        _inputs(),
        _targets(),
        num_epochs=2,
        train_fraction=1.0,
        param_reg_lambda=0.0,
        checkpoint_steps=0,
        output_transform=lambda o: torch.cumsum(o, dim=-1) * 0.5,
        seed=0,
    )[1]
    losses = hist["activation_loss"] if isinstance(hist, dict) else None
    assert losses is not None and np.all(np.isfinite(losses))
