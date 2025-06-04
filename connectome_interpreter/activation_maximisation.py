from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import contextlib


import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy import sparse
from scipy.sparse import coo_matrix

from .compress_paths import result_summary
from .utils import (
    adjacency_df_to_el,
    arrayable,
    get_activations,
    to_nparray,
    scipy_sparse_to_pytorch,
)


@dataclass
class TargetActivation:
    """
    Dataclass to handle target activations for activation maximisation.

    The target activations can be specified as a dictionary or a DataFrame.
    The dictionary should have the following structure:

    `{layer: {neuron_index: target_activation_value}}`

    The DataFrame should have the following columns:

    - 'batch': The batch index.
    - 'layer': The layer index.
    - 'neuron': The neuron index.
    - 'value': The target activation value.

    Args:
        targets (Union[Dict[int, Dict[int, float]], pd.DataFrame]): The target
            activations. If a dictionary, all batches will have the same
            target. If a DataFrame, each row represents a target activation for
            a specific batch.
        batch_size (Optional[int], optional): The number of batches. Defaults
            to None.
    """

    targets: Union[Dict[int, Dict[int, float]], pd.DataFrame]
    batch_size: Optional[int] | None = None

    def __post_init__(self):
        if isinstance(self.targets, dict):
            self.batch_size = self.batch_size or 1
            rows = []
            for layer, neurons in self.targets.items():
                for neuron, value in neurons.items():
                    for batch in range(self.batch_size):
                        rows.append(
                            {
                                "batch": batch,
                                "layer": layer,
                                "neuron": neuron,
                                "value": value,
                            }
                        )
            self.targets_df = pd.DataFrame(rows)
        else:
            required_cols = ["layer", "neuron", "value"]
            if not all(col in self.targets.columns for col in required_cols):
                raise ValueError(f"DataFrame must contain columns: {required_cols}")

            if "batch" in self.targets.columns:
                self.batch_size = self.batch_size or (self.targets["batch"].max() + 1)
                self.targets_df = self.targets.copy()
            else:
                self.batch_size = self.batch_size or 1
                self.targets_df = pd.DataFrame(
                    [
                        {**row, "batch": i}
                        for i in range(self.batch_size)
                        for _, row in self.targets.iterrows()
                    ]
                )

    def get_batch_targets(self, batch_idx: int) -> Dict[int, Dict[int, float]]:
        batch_data = self.targets_df[self.targets_df["batch"] == batch_idx]
        result = {}
        for _, row in batch_data.iterrows():
            if row["layer"] not in result:
                result[row["layer"]] = {}
            result[row["layer"]][row["neuron"]] = row["value"]
        return result


class MultilayeredNetwork(nn.Module):
    """
    A PyTorch module representing a multilayered neural network model with trainable
    parameters and flexible activation functions.

    This network architecture is designed to process temporal sequences
    of sensory data through multiple layers, with the initial layer
    handling only external inputs and subsequent layers processing
    external+internal input.

    The forward pass of the network unrolls the connectome through time,
    with each layer receiving its own time-specific sensory input.

    Attributes:
        all_weights (torch.nn.Parameter): The connectome. Input neurons are in
            the columns.
        sensory_indices (list[int]): Indices indicating which rows/columns in
            the all_weights matrix correspond to sensory neurons.
        num_layers (int): The number of layers in the network.
        threshold (float): The activation threshold for neurons in the network.
        activations (numpy.ndarray): An array storing the activations of all (batches of)
            neurons (rows) across time steps (columns).
        custom_activation_function (Callable): A custom activation function to use
            instead of the default.
        default_bias (float): Default bias value for all neurons.
        slope (torch.nn.Parameter): Trainable parameter for the steepness of the tanh
            activation function, grouped by neuron type.
        raw_biases (torch.nn.Parameter): Trainable parameter for the biases of neurons,
            grouped by neuron type.
        indices (torch.Tensor): Indices mapping each neuron to its group for parameter
            sharing.

    Methods:
        activation_function(x: torch.Tensor) -> torch.Tensor:
            Applies the activation function to the input tensor.
        forward(inputs: torch.Tensor) -> torch.Tensor:
            Processes inputs through the network and returns the activations.
    Context Managers:
        training_mode(model: MultilayeredNetwork, train_slopes=True, train_biases=True):
            Context manager to set the model in training mode, enabling gradients for
            slopes and biases.

    Args:
        all_weights (Union[torch.Tensor, scipy.sparse.spmatrix]): The connectome. Input
            neurons are in the columns.
        sensory_indices (list[int]): A list indicating the indices of sensory neurons
            within the network.
        num_layers (int, optional): The number of temporal layers to unroll the network
            through.  Defaults to 2.
        threshold (float, optional): The threshold for activation of neurons. Defaults
            to 0.01.
        tanh_steepness (float, optional): Default steepness for tanh activation.
            Defaults to 5.
        idx_to_group (dict, optional): Mapping from neuron indices to group names for
            parameter sharing (all neurons in the same group share the same parameter).
            Defaults to None (no parameter sharing).
        default_bias (float, optional): Default bias value for all groups. Defaults to
            0.0.
        bias_dict (dict, optional): Dict mapping group names to bias values. Defaults to
            None (uses default_bias).
        slope_dict (dict, optional): Dict mapping group names to slope values. Uses
            tanh_steepness if None.
        activation_function (Callable, optional): Custom activation function. If None,
            uses default implementation.
        device (torch.device, optional): Device for computation.
    """

    def __init__(
        self,
        all_weights: Union[torch.Tensor, sparse.spmatrix],
        sensory_indices: arrayable,
        num_layers: int = 2,
        threshold: float = 0.01,
        tanh_steepness: float = 5,
        idx_to_group: Optional[dict] = None,
        default_bias: float = 0.0,
        bias_dict: Optional[dict] = None,
        slope_dict: Optional[dict] = None,
        activation_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert all_weights to a sparse tensor if it is a scipy sparse matrix
        if isinstance(all_weights, sparse.spmatrix):
            all_weights = scipy_sparse_to_pytorch(all_weights)
        # check if all_weights is sparse
        assert all_weights.is_sparse, "all_weights must be sparse"

        self.all_weights = all_weights
        self.sensory_indices = torch.tensor(sensory_indices, device=device)
        self.num_layers = num_layers
        self.threshold = threshold
        self.tanh_steepness = tanh_steepness
        self.activations = []
        self.custom_activation_function = activation_function
        self.default_bias = default_bias

        # Setup trainable parameters if idx_to_group is provided
        if idx_to_group is not None:
            self._setup_trainable_parameters(
                idx_to_group, bias_dict, slope_dict, device
            )
        else:
            # For backward compatibility - no trainable parameters
            self.slope = None
            self.raw_biases = None
            self.indices = None

    def _setup_trainable_parameters(
        self, idx_to_group: dict, bias_dict, slope_dict, device
    ):
        """Setup trainable slope and bias parameters grouped by neuron type."""
        all_types = sorted(set(idx_to_group.values()))
        num_types = len(all_types)
        type2idx = {t: i for i, t in enumerate(all_types)}

        # Setup slope values
        if slope_dict is None:
            slope_init = torch.full(
                (num_types,), self.tanh_steepness, dtype=torch.float32
            )
        else:  # dict
            slope_init = torch.zeros(num_types, dtype=torch.float32)
            for group_name, slope_val in slope_dict.items():
                if group_name in type2idx:
                    slope_init[type2idx[group_name]] = slope_val
            # Fill missing values with default
            for i, group_name in enumerate(all_types):
                if group_name not in slope_dict:
                    slope_init[type2idx[group_name]] = self.tanh_steepness

        # Setup bias values
        if bias_dict is None:
            bias_init = torch.full((num_types,), self.default_bias, dtype=torch.float32)
        else:  # dict
            bias_init = torch.zeros(num_types, dtype=torch.float32)
            for group_name, bias_val in bias_dict.items():
                if group_name in type2idx:
                    bias_init[type2idx[group_name]] = bias_val
            # Fill missing values with default
            for group_name in all_types:
                if group_name not in bias_dict:
                    bias_init[type2idx[group_name]] = self.default_bias

        self.slope = nn.Parameter(slope_init.to(device), requires_grad=False)
        self.raw_biases = nn.Parameter(bias_init.to(device), requires_grad=False)
        self.indices = torch.tensor(
            [type2idx[idx_to_group[i]] for i in range(self.all_weights.shape[0])],
            device=device,
        )

    def set_param_grads(self, slopes=False, raw_biases=False):
        """Set requires_grad for specific parameters."""
        if self.slope is not None:
            self.slope.requires_grad_(slopes)
        if self.raw_biases is not None:
            self.raw_biases.requires_grad_(raw_biases)

    @property
    def biases(self):
        """
        Get the biases of the neurons, applying softplus activation if raw_biases
        are provided. If raw_biases is None, returns None.

        Returns:
            torch.Tensor or None: The biases of the neurons, or None if raw_biases
            is not set.
        """
        if self.raw_biases is None:
            return None
        return torch.nn.functional.softplus(self.raw_biases)

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the activation function to the input tensor. Currently it is
        tanh(relu(slope * x + bias)). x = weights @ inputs (done outside the activation
        function).

        Args:
            x (torch.Tensor): The input tensor to the activation function.

        Returns:
            torch.Tensor: The output tensor after applying the activation function.
        """
        if self.custom_activation_function is not None:
            return self.custom_activation_function(x)

        if self.slope is None:
            slopes = self.tanh_steepness
        else:
            slopes = self.slope[self.indices]
            slopes = slopes.view(-1, 1).expand(-1, x.shape[1]).to(x.device)

        if self.biases is None:
            biases = self.default_bias
        else:
            biases = self.biases[self.indices]
            biases = biases.view(-1, 1).expand(-1, x.shape[1]).to(x.device)

        x = slopes * x + biases

        # Thresholded relu
        x = torch.where(x >= self.threshold, x, torch.zeros_like(x))

        # Tanh activation
        x = torch.tanh(x)

        return x

    def forward(self, inputs: torch.Tensor):
        """
        Process inputs through the network. Use matrix-matrix multiplication for batched
        processing.

        Args:
            inputs (torch.Tensor): Either a 2D tensor (num_input, time_steps) or a 3D
                tensor (batch_size, num_input, time_steps)

        Returns:
            torch.Tensor: The activations of all neurons across time steps (in batches).
        """

        # if inputs is numpy, convert to tensor
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, device=self.all_weights.device)

        # Handle 2D inputs by expanding to 3D
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)  # Add batch dimension
            single_input = True
        else:
            single_input = False

        acts = []

        inputs = torch.where(inputs >= self.threshold, inputs, torch.zeros_like(inputs))
        inputs = torch.where(inputs > 1, torch.ones_like(inputs), inputs)

        # check if inputs requires grad
        req_grad = inputs.requires_grad
        # shape: (all_neurons, batch_size)
        full_input = torch.zeros(
            self.all_weights.size(1),
            inputs.size(0),
            device=inputs.device,
            requires_grad=req_grad,
        )
        full_input = full_input.index_add(0, self.sensory_indices, inputs[:, :, 0].t())

        # sparse matrix multiplication
        # shape: (all_neurons, batch_size)
        x = torch.sparse.mm(self.all_weights, full_input)
        x = self.activation_function(x)

        # add external input to sensory neurons
        # so the output activation includes the external input as well
        if self.num_layers > 1:
            x = x.clone()
            x[self.sensory_indices, :] = (
                x[self.sensory_indices, :] + inputs[:, :, 1].t()
            )
            # make sure the max is 1
            x = torch.where(x > 1, torch.ones_like(x), x)

        acts.append(x.t())  # shape: (batch_size, all_neurons)

        # Process remaining layers
        for alayer in range(1, self.num_layers):
            # shape: (all_neurons, batch_size)
            x = torch.sparse.mm(self.all_weights, x)
            # thresholded relu and tanh
            x = self.activation_function(x)

            # add external input to sensory neurons, if not the last layer
            if alayer != self.num_layers - 1:
                x = x.clone()  # need to do this to avoid in-place operation
                x[self.sensory_indices, :] = (
                    x[self.sensory_indices, :] + inputs[:, :, alayer + 1].t()
                )
                # make sure the max is 1
                x = torch.where(x > 1, torch.ones_like(x), x)

            acts.append(x.t())  # shape: (batch_size, all_neurons)

        # Stack activations
        # shape: (batch_size, all_neurons, num_layers)
        self.activations = torch.stack(acts, dim=-1)
        # free up memory as much as possible
        del inputs
        del x
        del acts
        torch.cuda.empty_cache()

        if single_input:
            self.activations = self.activations.squeeze(0)
        return self.activations


# Note: input gradients are handled in the activation_maximisation and saliency
# functions themselves, so no context manager for that here
@contextlib.contextmanager
def training_mode(model: MultilayeredNetwork, train_slopes=True, train_biases=True):
    """Context manager for training mode - enables gradients for slopes and biases."""
    model.set_param_grads(slopes=train_slopes, raw_biases=train_biases)
    try:
        yield model
    finally:
        model.set_param_grads(slopes=False, raw_biases=False)


def train_model(
    model: MultilayeredNetwork,
    inputs: torch.Tensor,
    targets: pd.DataFrame,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    param_reg_lambda: float = 0.01,
    wandb: bool = False,
    wandb_project_name: str = "connectome_interpreter",
    train_fraction: float = 0.8,
    seed: int = 42,
    train_slopes: bool = True,
    train_biases: bool = True,
):
    """
    Train the model to approximate the targets, while keeping the model parameter change
    minumum (param_reg_lambda decdes how strongly). The loss is the average absolute
    difference between the model activations and the target activations.

    Args:
        model (MultilayeredNetwork): The model to train.
        inputs (torch.Tensor): The input data. Shape: (batch_size, num_input_neurons,
            num_layers).
        targets (pd.DataFrame): The target activations. A DataFrame with columns:
            "batch", "neuron_idx", "layer" (optional), "value". The "layer" column
            specifies the timestep (0-indexed). If 'layer' is not present, the average
            activation across all timesteps is used.
        num_epochs (int, optional): Number of training epochs. Defaults to 100.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to
            0.01.
        param_reg_lambda (float, optional): Regularization parameter for the model
            parameters. Defaults to 0.01.
        wandb (bool, optional): Whether to use Weights & Biases logging.
        wandb_project_name (str, optional): Project name for wandb.
        train_fraction (float, optional): Fraction of data to use for training.
        seed (int, optional): Random seed for reproducibility.
        train_slopes (bool, optional): Whether to train slopes. Defaults to True.
        train_biases (bool, optional): Whether to train biases. Defaults to True.
    """

    def train_test_split(inputs, targets, train_fraction):
        """Split the inputs and targets into training and validation sets."""
        train_num = int(inputs.shape[0] * train_fraction)
        # random selection of train indices
        train_indices = np.random.choice(
            range(inputs.shape[0]), train_num, replace=False
        )
        # target is the rest
        val_indices = np.setdiff1d(range(inputs.shape[0]), train_indices)

        train_inputs = inputs[train_indices]
        val_inputs = inputs[val_indices]

        train_targets = targets[targets["batch"].isin(train_indices)]
        train_targets.loc[:, ["batch"]] = pd.Categorical(
            train_targets["batch"], categories=list(train_indices)
        )
        train_targets = train_targets.sort_values(by="batch")
        # change to local batch indices
        batch2local_batch = {b: i for i, b in enumerate(train_indices)}
        train_targets.loc[:, ["batch"]] = train_targets.batch.map(batch2local_batch)

        val_targets = targets[targets["batch"].isin(val_indices)]
        val_targets.loc[:, ["batch"]] = pd.Categorical(
            val_targets["batch"], categories=list(val_indices)
        )
        val_targets = val_targets.sort_values(by="batch")
        # change to local batch indices
        batch2local_batch = {b: i for i, b in enumerate(val_indices)}
        val_targets.loc[:, ["batch"]] = val_targets.batch.map(batch2local_batch)

        return (
            train_inputs,
            val_inputs,
            train_targets,
            val_targets,
            train_indices,
            val_indices,
        )

    def get_activation_loss(actual_activations, targets):
        """Calculate the activation loss for the model with time series support."""
        activation_loss = torch.tensor(0.0, device=actual_activations.device)

        # Calculate loss
        for b in targets["batch"].unique():
            batch_targets = targets[targets["batch"] == b]

            batch_loss = torch.tensor(0.0, device=actual_activations.device)
            for _, row in batch_targets.iterrows():
                target_value = row["value"]
                neuron_idx = int(row["neuron_idx"])  # Convert to int

                if "layer" in row:
                    # Time series mode - use specific layer/timestep
                    actual_value = actual_activations[b, neuron_idx, int(row["layer"])]
                else:
                    # Backward compatibility - average across timesteps
                    actual_value = actual_activations[b, neuron_idx, :].mean()

                # batch loss is absolute difference between actual and target
                batch_loss += torch.abs(actual_value - target_value)
            batch_loss /= len(batch_targets)  # average over neurons
            activation_loss += batch_loss

        if len(targets["batch"].unique()) == 0:
            return activation_loss
        else:
            activation_loss /= len(targets["batch"].unique())  # average over batches
        return activation_loss

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Check column names of targets - support both old and new format
    required_cols = ["batch", "neuron_idx", "value"]
    if not all(col in targets.columns for col in required_cols):
        raise ValueError(
            "Targets DataFrame must contain columns: 'batch', 'neuron_idx', 'value'"
            "Optionally include 'layer' for time series targets."
        )

    if wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Please install it using 'pip install wandb' or set wandb=False."
            ) from exc

        wandb.init(project=wandb_project_name)

    device = model.all_weights.device
    # Move data to device
    if not torch.is_tensor(inputs):
        inputs = torch.tensor(inputs, device=device)
    else:
        inputs = inputs.to(device)

    # Use training mode context manager
    with training_mode(model, train_slopes, train_biases):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training history
        history = {"loss": [], "mse_loss": [], "param_reg_loss": [], "val_loss": []}

        initial_params = [param.clone() for param in model.parameters()]

        # train validation split
        (
            train_inputs,
            val_inputs,
            train_targets,
            val_targets,
            train_indices,
            val_indices,
        ) = train_test_split(inputs, targets, train_fraction)

        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()
            # Forward pass
            outputs = model(train_inputs)  # shape: (train_num, num_neurons, num_layers)

            activation_loss = get_activation_loss(outputs, train_targets)

            # regularization loss
            param_reg_loss = 0
            for param, param0 in zip(model.parameters(), initial_params):
                param_reg_loss += torch.norm(param - param0, p=1)

            param_reg_loss = param_reg_lambda * param_reg_loss

            loss = activation_loss + param_reg_loss
            loss.backward()
            optimizer.step()

            # validation loss
            with torch.no_grad():
                val_outputs = model(val_inputs)
                val_activation_loss = get_activation_loss(val_outputs, val_targets)
                if wandb:
                    wandb.log({"val_activation_loss": val_activation_loss.item()})

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: Loss = {loss.item()}, "
                    f"Activation Loss = {activation_loss.item()}, "
                    f"Parameter Regularization Loss = {param_reg_loss.item()}, "
                    f"Validation Activation Loss = {val_activation_loss.item()}"
                )

            if wandb:
                wandb.log(
                    {
                        "learning_rate": learning_rate,
                        "param_reg_lambda": param_reg_lambda,
                        "activation_loss": activation_loss.item(),
                        "param_reg_loss": param_reg_loss.item(),
                    }
                )

            history["loss"].append(loss.item())
            history["mse_loss"].append(activation_loss.item())
            history["param_reg_loss"].append(param_reg_loss.item())
            history["val_loss"].append(val_activation_loss.item())

    # Free up memory
    torch.cuda.empty_cache()
    # clear the computational graph of activations
    with torch.no_grad():
        model.activations = model.activations.detach()

    # return the model and the training history
    return (
        model,
        history,
        train_inputs,
        val_inputs,
        train_targets,
        val_targets,
        train_indices,
        val_indices,
    )


def activation_maximisation(
    model: MultilayeredNetwork,
    target_activations: TargetActivation,
    input_tensor: Optional[torch.Tensor] = None,
    num_iterations: int = 50,
    learning_rate: float = 0.1,
    # regularization
    in_reg_lambda: float = 0.01,
    out_reg_lambda: float = 0.01,
    custom_reg_functions: Optional[
        Dict[str, Callable[[torch.Tensor], torch.Tensor]]
    ] = None,
    # early stopping
    early_stopping: bool = True,
    stopping_threshold: float = 1e-5,
    n_runs: int = 10,
    use_tqdm: bool = True,
    print_output: bool = True,
    device: Optional[torch.device] = None,
    wandb: bool = False,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[float],
    List[float],
    List[float],
    List[np.ndarray],
]:
    """
    Performs activation maximisation on a given model to identify input
    patterns that result in the target activations.

    This is done by adjusting the input tensor over `num_iterations`
    using gradient descent, while also regularising the overall input
    and output (to keep activated neurons sparse).

    Args:
        model: A PyTorch model with `activations`, `sensory_indices`, and
            `threshold` attributes.
        target_activations (TargetActivation): Target activations
            specification.
        input_tensor (torch.Tensor, optional): The initial tensor to optimize.
            If None, a random tensor is created.  Defaults to None.
        num_iterations (int, optional): The number of iterations to run the
            optimization for. Defaults to 50.
        learning_rate (float, optional): The learning rate for the optimizer.
            Defaults to 0.1.
        in_reg_lambda (float, optional): The coefficient for input
            regularization. Defaults to 0.01.
        out_reg_lambda (float, optional): The coefficient for output
            regularization. Defaults to 0.01.
        custom_reg_functions (Dict[str, Callable[[torch.Tensor]], optional):
            A dictionary with keys 'in' and 'out' that map to functions that
            calculate the input and output regularization losses, respectively.
            If None, the default regularization function (L1 plus L2) is used.
            Defaults to None.
        early_stopping (bool, optional): Whether to stop the optimization early
            if the difference between the biggest and the smallest loss within
            the last n_runs falls below `stopping_threshold`. Defaults to True.
        stopping_threshold (float, optional): The threshold for early stopping.
            Defaults to 1e-6.
        n_runs (int, optional): The number of runs to consider for early
            stopping. Defaults to 10.
        use_tqdm (bool, optional): Whether to use tqdm progress bars to track
            optimization progress. Defaults to True.
        print_output (bool, optional): Whether to print loss information during
            optimization. Defaults to True.
        report_memory_usage (bool, optional): Whether to report GPU memory
            usage during optimization. Defaults to False.
        device: The device to run the optimization on. If None, automatically
            selects a device. Defaults to None.
        wandb (bool, optional): Whether to log optimization details to Weights
            & Biases (https://wandb.ai/site/). Defaults to True.  Requires
            wandb to be installed.

    Returns:
        tuple: A tuple containing:

            - numpy.ndarray: The optimized input as a numpy array.
            - numpy.ndarray: The output of the model after optimization as a
                numpy array.
            - list(float): A list of output activation losses over iterations.
            - list(float): A list of input activation regularization losses
                over iterations.
            - list(float): A list of output activation regularization losses
                over iterations.
            - list(numpy.ndarray): A list of input tensor snapshots taken
                during optimization.


    Examples:

        .. code-block:: python

            # Single target for multiple batches
            targets_dict = {
                # layer 0: neuron 1 -> 0.5, neuron 2 -> 0.8
                0: {1: 0.5, 2: 0.8},
                # layer 1: neuron 0 -> 0.3
                1: {0: 0.3}
            }
            targets = TargetActivation(targets=targets_dict, batch_size=4)
            inputs, outputs, *losses = activation_maximisation(model, targets)

            # Different targets per batch using DataFrame
            targets_df = pd.DataFrame([
                {'batch': 0, 'layer': 0, 'neuron': 1, 'value': 0.5},
                {'batch': 0, 'layer': 0, 'neuron': 2, 'value': 0.8},
                {'batch': 1, 'layer': 1, 'neuron': 0, 'value': 0.3}
            ])
            # batch_size inferred
            targets = TargetActivation(targets=targets_df)
            results = activation_maximisation(model, targets)

            # Custom regularization
            def sparse_reg(x):
                return torch.sum(torch.abs(x))
            custom_reg = {'in': sparse_reg, 'out': sparse_reg}
            results = activation_maximisation(
                model, targets, custom_reg_functions=custom_reg
            )
    """

    def default_reg(x):
        return torch.norm(x, p=1) + torch.norm(x, p=2)

    def calculate_activation_loss(activations, batch_idx):
        """Calculate loss for specific batch using its target activations"""
        batch_idx = int(batch_idx)
        batch_targets = target_activations.get_batch_targets(batch_idx)
        loss = torch.tensor(0.0, device=device)
        n_neurons = 0

        for layer, neuron_targets in batch_targets.items():
            for neuron_index, target_value in neuron_targets.items():
                actual_value = activations[batch_idx, int(neuron_index), int(layer)]
                loss += (actual_value - target_value) ** 2
                n_neurons += 1

        return loss / n_neurons if n_neurons > 0 else loss

    # Setup and validation code
    if wandb:
        try:
            import wandb as wandb_lib
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Please install it using 'pip install wandb' or set wandb=False."
            ) from exc
        wandb_lib.init(project="connectome_interpreter")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = target_activations.batch_size

    if input_tensor is None:
        input_tensor = torch.rand(
            (batch_size, len(model.sensory_indices), model.num_layers),
            requires_grad=True,
            device=device,
        )
    else:
        if input_tensor.shape != (
            batch_size,
            len(model.sensory_indices),
            model.num_layers,
        ):
            raise ValueError(
                f"Expected input shape (batch_size={batch_size}, "
                f"num_input_neurons={len(model.sensory_indices)}, "
                f"num_layers={model.num_layers})"
            )
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([input_tensor], lr=learning_rate)

    input_snapshots = []
    act_loss = []
    out_reg_losses = []
    in_reg_losses = []
    losses = []
    iteration_range = range(num_iterations)

    if use_tqdm:
        iteration_range = tqdm(iteration_range)

    for iteration in iteration_range:
        optimizer.zero_grad()
        # Forward pass
        _ = model(input_tensor)

        # Take snapshot every 5 iterations
        if iteration % 5 == 0:
            snapshot = input_tensor.clone().detach().cpu().numpy()
            snapshot = np.where(snapshot >= model.threshold, snapshot, 0)
            snapshot = np.tanh(snapshot)
            input_snapshots.append(snapshot)

        # Calculate activation loss
        activation_loss = torch.mean(
            torch.stack(
                [
                    calculate_activation_loss(model.activations, batch_idx)
                    for batch_idx in range(batch_size)
                ]
            )
        )

        # Regularization loss
        if custom_reg_functions and "in" in custom_reg_functions:
            in_reg_loss = in_reg_lambda * custom_reg_functions["in"](input_tensor)
        else:
            in_reg_loss = in_reg_lambda * default_reg(input_tensor)

        if custom_reg_functions and "out" in custom_reg_functions:
            out_reg_loss = out_reg_lambda * custom_reg_functions["out"](
                model.activations
            )
        else:
            out_reg_loss = out_reg_lambda * default_reg(model.activations)

        loss = activation_loss + in_reg_loss + out_reg_loss
        losses.append(loss.item())

        # Early stopping
        if early_stopping and (iteration > n_runs):
            if np.max(losses[-n_runs:]) - np.min(losses[-n_runs:]) < stopping_threshold:
                break

        if wandb:
            wandb_lib.log(
                {
                    "activation_loss": activation_loss.item(),
                    "in_regularisation_loss": in_reg_loss.item(),
                    "out_regularisation_loss": out_reg_loss.item(),
                    "loss": loss.item(),
                }
            )

        act_loss.append(activation_loss.item())
        out_reg_losses.append(out_reg_loss.item())
        in_reg_losses.append(in_reg_loss.item())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if print_output and iteration % 10 == 0:
            print(
                f"Iteration {iteration}: Activation Loss = {activation_loss.item()}, "
                f"Input regularization Loss = {in_reg_loss.item()}, "
                f"Output regularization Loss = {out_reg_loss.item()}"
            )

        torch.cuda.empty_cache()

    # Final processing
    print(
        f"Final - Activation loss: {act_loss[-1]}, Input reg: {in_reg_losses[-1]}, Output reg: {out_reg_losses[-1]}"
    )

    input_tensor = torch.where(input_tensor >= model.threshold, input_tensor, 0)
    input_tensor = torch.tanh(input_tensor)

    output_after = model(input_tensor).cpu().detach().numpy()
    input_tensor = input_tensor.cpu().detach().numpy()

    # Clear computational graph
    with torch.no_grad():
        model.activations = model.activations.detach()

    # Handle single batch case
    if batch_size == 1:
        input_tensor = input_tensor[0]
        output_after = output_after[0]
        input_snapshots = [snap[0] for snap in input_snapshots]

    return (
        input_tensor,
        output_after,
        act_loss,
        out_reg_losses,
        in_reg_losses,
        input_snapshots,
    )


def saliency(
    model: MultilayeredNetwork,
    input_tensor: torch.Tensor,
    neurons_of_interest: Dict[int, List[int]],
    method: str = "vanilla",
    normalize: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Computes saliency maps: given the current input, to what extent will each element's
    change result in change in the activations of the neurons of interest?

    Args:
        model: A MultilayeredNetwork model
        input_tensor: Input tensor to analyze
        neurons_of_interest: Dictionary mapping layer indices to lists of neuron indices
            to analyze: {layer_idx: [neuron_indices]}
        method: Saliency computation method ("vanilla" or "input_x_gradient")
        normalize: Whether to normalize saliency maps by their maximum value
        device: Computation device

    Returns:
        torch.Tensor: Saliency maps with same shape as input tensor
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensor if needed
    if isinstance(input_tensor, np.ndarray):
        input_tensor = torch.tensor(input_tensor, device=device)

    input_tensor = input_tensor

    # Ensure input requires gradients
    input_tensor = input_tensor.clone().detach().to(device)
    input_tensor.requires_grad = True

    # Run forward pass
    _ = model(input_tensor)

    # Sum activations of neurons we care about
    activation_signal = 0
    for layer, neuron_indices in neurons_of_interest.items():
        for neuron_idx in neuron_indices:
            if input_tensor.dim() == 3:  # batched input
                for batch_idx in range(input_tensor.shape[0]):
                    activation_signal = (
                        activation_signal
                        + model.activations[batch_idx, neuron_idx, layer]
                    )
            else:  # single input
                activation_signal = (
                    activation_signal + model.activations[neuron_idx, layer]
                )

    # Compute gradients by backpropagation
    activation_signal.backward()

    # Get the gradients based on selected method
    if method == "vanilla":
        saliency_maps = input_tensor.grad.clone()
    elif method == "input_x_gradient":
        saliency_maps = input_tensor * input_tensor.grad
    else:
        raise ValueError(f"Unknown saliency method: {method}")

    if normalize and torch.max(saliency_maps) > 0:
        if saliency_maps.dim() == 3:  # Normalize each batch separately
            for i in range(saliency_maps.shape[0]):
                if torch.max(saliency_maps[i]) > 0:
                    saliency_maps[i] = saliency_maps[i] / torch.max(saliency_maps[i])
        else:
            saliency_maps = saliency_maps / torch.max(saliency_maps)

    return saliency_maps


def activations_to_df(
    inprop,
    model_input: np.ndarray,
    out: np.ndarray,
    sensory_indices: List[int],
    inidx_mapping: dict | None = None,
    outidx_mapping: dict | None = None,
    activation_threshold: float = 0,
    connectivity_threshold: float = 0,
    high_ram: bool = True,
) -> pd.DataFrame:
    """
    Generates a dataframe representing the paths in a layered plot,filtering
    by activation and connectivity thresholds.

    This function takes the direct connectivity matrix (inprop), input
    neuron activity, output neuron activity, indices for sensory
    neurons, and mapping between input and output indices to groups. It
    generates a dataframe that represents the paths through the network
    layers.

    Args:
        inprop (scipy.sparse matrix or numpy.ndarray): Matrix
            representing the synaptic strengths between neurons, can be
            dense or sparse. Presynaptic is in the rows, postsynaptic in the
            columns.
        model_input (numpy.ndarray): A 2D array representing input to
            the network. Neurons are in the rows, timepoints in the columns.
            Only the first timepoint is used, since `out` is expected to
            have activity of all neurons, including input neurons.
        out (numpy.ndarray): A 2D array representing the output from the
            network. The second dimension represents timepoints.
        sensory_indices (list of int): A list of indices corresponding
            to sensory neurons in `inprop`.
        inidx_mapping (dict, optional): A dictionary mapping indices in
            `inprop` to new indices (e.g. cell type). If None, indices are
            not remapped. Defaults to None.
        outidx_mapping (dict, optional): A dictionary mapping indices in `out`
            to new indices. If None, `inidx_mapping` is used for mapping.
            Defaults to None.
        activation_threshold (float, optional): A threshold value for
            activation. Neurons with activations below this threshold are not
            considered. Defaults to 0.
        connectivity_threshold (float, optional): A threshold for filtering
            connections. Connections with weights below this threshold are
            ignored. Defaults to 0.
        high_ram (bool, optional): Whether to use a high RAM implementation
            (which is slightly faster). This implementation gets direct
            connections between *all* relevant neurons at once, instead of
            within each layer. Defaults to True.

    Returns:
        pandas.DataFrame: A dataframe representing the paths in the network.
        Each row is a connection, with columns for 'pre' and 'post' neuron
        indices, 'layer', and their respective activations ('pre_activation',
        'post_activation').
    """
    all_indices = list(range(inprop.shape[0]))

    if inidx_mapping is None:
        inidx_mapping = {idx: idx for idx in all_indices}

    if outidx_mapping is None:
        outidx_mapping = inidx_mapping

    # move to CPU in case it's still on GPU
    if torch.is_tensor(model_input):
        model_input = model_input.cpu().numpy()
    if torch.is_tensor(out):
        out = out.detach().cpu().numpy()

    print("Getting activations...")
    # get activations from input and out, based on the mappings provided by
    # inidx_mapping and outidx_mapping
    sensory_act = get_activations(
        model_input,
        sensory_indices,
        inidx_mapping,
        threshold=activation_threshold,
    )
    all_act = get_activations(
        out, all_indices, outidx_mapping, threshold=activation_threshold
    )

    print(
        "Getting connectivity... If this takes a while, consider increasing "
        "the connectivity/activation threshold."
    )

    if high_ram:
        # get all pre and post indices across layers
        post_groups = {key for _, d in all_act.items() for key in d}
        pre_groups = set(sensory_act[0].keys()).union(post_groups)
        pre_indices = [idx for idx, key in inidx_mapping.items() if key in pre_groups]
        post_indices = [
            idx for idx, key in outidx_mapping.items() if key in post_groups
        ]

        # if there are literally no neurons
        if len(pre_indices) == 0 or len(post_indices) == 0:
            raise ValueError(
                "No neurons found. Consider lowering the activation threshold."
            )

        # get connectivity
        conn = result_summary(
            inprop,
            inidx=pre_indices,
            outidx=post_indices,
            inidx_map=inidx_mapping,
            outidx_map=outidx_mapping,
            display_output=False,
        )
        # turn to edgelist, and filter
        conn_el = adjacency_df_to_el(conn, threshold=connectivity_threshold)

    # make paths df
    paths = []
    for layer in range(out.shape[1]):
        if layer == 0:
            # the initial layer only has input neuron activations
            pre = sensory_act[layer]
            post = all_act[layer]
        else:
            # all the input, not just the optimised external input
            pre = all_act[layer - 1]
            post = all_act[layer]

        pre_stringkeys = {str(key): val for key, val in pre.items()}
        post_stringkeys = {str(key): val for key, val in post.items()}

        if high_ram:
            # index the big connectivity matrix for this layer
            connections = conn_el[
                conn_el.pre.isin(pre_stringkeys.keys())
                & conn_el.post.isin(post_stringkeys.keys())
            ]

        else:
            # pre and post are already grouped by inidx and outidx_mapping
            # so need to recover the indices using pre, inidx_map, post,
            # outidx_map
            pre_indices = [
                idx for idx, val in inidx_mapping.items() if val in pre.keys()
            ]
            post_indices = [
                idx for idx, val in outidx_mapping.items() if val in post.keys()
            ]

            conn = result_summary(
                inprop,
                inidx=pre_indices,
                outidx=post_indices,
                inidx_map=inidx_mapping,
                outidx_map=outidx_mapping,
                display_output=False,
            )
            # turn to edgelist, and filter
            connections = adjacency_df_to_el(conn, threshold=connectivity_threshold)

        # so that direct connectivity is layer 1
        connections.loc[:, ["layer"]] = layer + 1
        connections.loc[:, ["pre_activation"]] = connections.pre.map(pre_stringkeys)
        connections.loc[:, ["post_activation"]] = connections.post.map(post_stringkeys)
        if connections.shape[0] > 0:
            paths.append(connections)
        else:
            print(f"Warning: No connections found in layer {layer+1}.")
    paths = pd.concat(paths)
    return paths


def activations_to_df_batched(
    inprop,
    opt_in: np.ndarray,
    out: np.ndarray,
    sensory_indices: List[int],
    inidx_mapping: dict | None = None,
    outidx_mapping: dict | None = None,
    activation_threshold: float = 0,
    connectivity_threshold: float = 0,
    high_ram: bool = True,
) -> pd.DataFrame:
    """
    Generates a dataframe representing the paths in a layered plot,
    filtering by activation and connectivity thresholds.

    This function takes the direct connectivity matrix (inprop), optimal
    input neuron activity, output neuron activity, indices for sensory
    neurons, and mapping between input and output indices to groups. It
    generates a dataframe that represents the paths through the network
    layers.

    Args:
        inprop (scipy.sparse matrix or numpy.ndarray): Matrix representing the
            synaptic strengths between neurons, can be dense or sparse.
            Presynaptic is in the rows, postsynaptic in the columns.
        opt_in (numpy.ndarray): A **3D** array representing optimal input to
            the network. The first dimension represents the batch size, the
            second dimension represents input neurons, and the third dimension
            represents timepoints. Only the first timepoint is used, since
            `out` is expected to have activity of all neurons, including input
            neurons.
        out (numpy.ndarray): A **3D** array representing the activation of all
            neurons. The first dimension represents the batch size, the second
            dimension represents all neurons, and the third dimension
            represents timepoints.
        sensory_indices (list of int): A list of indices corresponding to
            sensory neurons in `inprop`.
        inidx_mapping (dict, optional): A dictionary mapping indices in
            `inprop` to new indices. If None, indices are not remapped.
            Defaults to None.
        outidx_mapping (dict, optional): A dictionary mapping indices in `out`
            to new indices. If None, `inidx_mapping` is used for mapping.
            Defaults to None.
        activation_threshold (float, optional): A threshold value for
            activation. Neurons with activations below this threshold are not
            considered. Defaults to 0.
        connectivity_threshold (float, optional): A threshold for filtering
            connections. Connections with weights below this threshold are
            ignored. Defaults to 0.
        high_ram (bool, optional): Whether to use a high RAM implementation
            (which is slightly faster). This implementation gets direct
            connections between *all* relevant neurons at once, instead of
            within each layer. Defaults to True.

    Returns:
        pandas.DataFrame: A dataframe representing the paths in the network.
        Each row is a connection, with columns for 'pre' and 'post' neuron
        indices, 'layer', and their respective activations ('pre_activation',
        'post_activation').
    """
    # use activations_to_df for each batch
    paths = []
    # assume the first dimnesion of opt_in and out is the batch size
    for i in tqdm(range(opt_in.shape[0])):
        path = activations_to_df(
            inprop,
            opt_in[i],
            out[i],
            sensory_indices,
            inidx_mapping,
            outidx_mapping,
            activation_threshold,
            connectivity_threshold,
            high_ram,
        )
        path.loc[:, ["batch"]] = i
        paths.append(path)
    return pd.concat(paths)


def input_from_df(
    df: pd.DataFrame,
    sensory_indices: list,
    idx_to_group: dict,
    num_layers: int,
    timepoints: Union[int, List[int], np.ndarray, set] = 0,
) -> np.ndarray:
    """
    Make well-formatted input for the model, based on defined vectors of
    input neuron activation (df). The function returens a 3D tensor,
    with the first dimension being the batch size, the second dimension
    being the number of input neurons of the model, and the third
    dimension being the number of layers.

    Args:
        df : pd.DataFrame
            Rows correspond to the values in `idx_to_group`, and columns
            correspond to the batches (number of columns = number of batch).
            For instance, rows of df can be olfactory glomeruli, and columns
            can be odours. df is the reaction of each glomerulus to each odour.
        sensory_indices : list
            The indices of sensory neurons. For instance, these could be the
            indices of individual olfactory receptor neurons.
        idx_to_group : dict
            A dictionary that maps indices to group. For instance, this could
            map from indices of indivudal olfactory receptor neuron to the
            glomerulus they innervate (rows of df).
        num_layers : int
            The number of layers in the model.

    Returns:
        np.ndarray: The input for the model.

    """
    # first initialise empty input
    inarray = np.zeros((df.shape[1], len(sensory_indices), num_layers))

    # Ensure timepoints is iterable
    if isinstance(timepoints, int):
        timepoints = [timepoints]
    elif isinstance(timepoints, (np.ndarray, set)):
        timepoints = list(timepoints)

    for l in range(df.shape[1]):
        grp2act = df.iloc[:, l]
        grp2act = dict(zip(grp2act.index, grp2act.values.flatten()))
        actvec = [
            grp2act[idx_to_group[idx]] if idx_to_group[idx] in grp2act else 0
            for idx in sensory_indices
        ]
        for t in timepoints:
            inarray[l, :, t] = actvec
    return inarray


def get_neuron_activation(
    activations: torch.Tensor | npt.NDArray,
    neuron_indices: arrayable,
    batch_names: arrayable | None = None,
    idx_to_group: dict | None = None,
) -> pd.DataFrame:
    """
    Get the activations for specified indices across timepoints, include
    batch name and group information when available.

    Args:
        activations (torch.Tensor | numpy.ndarray): Output activation from
            the model. Shape should be (batch_size, num_neurons,
            num_timepoints) or (num_neurons, num_timepoints).
        neuron_indices (arrayable): The indices of the neurons to get
            activations for.
        batch_names (arrayable, optional): The names of the batches.
            Defaults to None. If activations.ndim == 3, then this should be
            supplied. If not, batch names will be e.g. 'batch_0', 'batch_1',
            etc.
        idx_to_group (dict, optional): A dictionary mapping indices to
            groups. Defaults to None.

    Returns:
        pd.DataFrame: The activations for the neurons, with the first
            columns being batch_names, neuron_indices, and group. The rest
            are the timesteps.
    """
    neuron_indices = list(to_nparray(neuron_indices))

    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().detach().numpy()

    if idx_to_group is None:
        idx_to_group = {idx: idx for idx in range(activations.shape[0])}

    if activations.ndim == 2:
        # message if batch_names is not None
        if batch_names is not None:
            print("batch_names is ignored for 2D activations.")

        data = activations[neuron_indices, :]
        df = pd.DataFrame(
            data,
            columns=[f"time_{i}" for i in range(activations.shape[1])],
            index=neuron_indices,
        )
        df.index.name = "idx"
        df["group"] = [idx_to_group[idx] for idx in neuron_indices]
        # discard index, use group as index
        # then groupby group, and get mean
        df = df.set_index("group").groupby("group").mean().reset_index()

        return df

    if activations.ndim == 3:
        if batch_names is None:
            # make batch names
            batch_names = [f"batch_{i}" for i in range(activations.shape[0])]

        batch_names = list(to_nparray(batch_names, unique=False))

        # make sure length matches
        if activations.shape[0] != len(batch_names):
            raise ValueError(
                "Length of batch_names has to be the same as " "activations.shape[0]."
            )

        data = activations[:, neuron_indices, :].reshape(
            -1, activations.shape[2]
        )  # from (batch, neuron, time) to (batch*neuron, time)

        # Create indices for the first two dimensions
        # in the end, we want batch * n_indices rows
        # that's activations.shape[0] * len(neuron_indices) rows
        batch_names = [o for o in batch_names for _ in range(len(neuron_indices))]
        neuron_indices = neuron_indices * activations.shape[0]

        # Combine indices and data into a DataFrame
        df = pd.DataFrame(
            data, columns=[f"time_{i}" for i in range(activations.shape[2])]
        )
        df["batch_name"] = batch_names
        df["group"] = [idx_to_group[idx] for idx in neuron_indices]

        # groupby batch_name and group, calculate average
        df = df.groupby(["batch_name", "group"]).mean().reset_index()

    return df


def get_activations_for_path(
    path: pd.DataFrame,
    activations: torch.Tensor | npt.NDArray,
    model_in: torch.Tensor | npt.NDArray | None = None,
    sensory_indices: arrayable | None = None,
    idx_to_group: dict | None = None,
    activation_start: int = 0,
) -> pd.DataFrame:
    """
    Get the activations for the pre and post neurons in the path, based on
    the activations of the model and the input.

    Args:
        path (pd.DataFrame): A dataframe representing the paths in the network.
            Each row is a connection, with columns for 'pre' and 'post' neuron
            indices, and 'layer'.
        activations (torch.Tensor | numpy.ndarray): The activations of the
            model. Shape should be (num_neurons, num_layers).
        model_in (torch.Tensor | numpy.ndarray): The input to the model. Shape
            should be (num_neurons, something) -  only the first column
            (num_neurons, 0), is used, when there is 1 in 'layer' in `path`. It
            is otherwise not used.
        sensory_indices (arrayable): The indices of sensory neurons.
        idx_to_group (dict, optional): A dictionary mapping indices from the
            model to the groups in path (e.g. cell type). Defaults to None.
        activation_start (int, optional): Which layer corresponds to the start
            layer of path. By default, activation_start = 0, the sensory-only
            layer in the model corresponds to path.pre[path.layer == 1]. If you
            want activations[:,1] (i.e. two timesteps forward) to correspond to
            path.pre[path.layer == 1], set activation_start = 2. If you want
            the last timepoint to correspond to the last layer in path, set
            activation_start = activations.shape[1] - path.layer.max().

    Returns:
        pd.DataFrame: The activations for the pre and post neurons in the path.
    """

    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().detach().numpy()
    if isinstance(model_in, torch.Tensor):
        model_in = model_in.cpu().detach().numpy()

    if idx_to_group is not None:
        # turn value into string
        idx_to_group = {k: str(v) for k, v in idx_to_group.items()}
    else:
        idx_to_group = {idx: idx for idx in range(activations.shape[0])}

    # if starting later, just bump up the layer numbers
    path.loc[:, ["layer"]] += activation_start

    out_df = []
    for l in sorted(path.layer.unique()):  # layer number starts at 1
        layer_path = path[path.layer == l]

        # pre activations
        prenodes = set(layer_path.pre)
        pre_indices = [idx for idx, group in idx_to_group.items() if group in prenodes]
        if l == 1:
            # raise error if sensory_indices or model_in doesn't exist
            if sensory_indices is None or model_in is None:
                raise ValueError(
                    "sensory_indices and model_in must be provided when layer == 1."
                )
            # need to get local indices for sensory_indices
            global2local = {idx: i for i, idx in enumerate(sensory_indices)}
            local_indices = [global2local[idx] for idx in pre_indices]
            pre_activations = model_in[local_indices, 0]
        else:
            pre_activations = activations[pre_indices, l - 2]
        pre_group_act = pd.DataFrame(
            {"idx": pre_indices, "activation": pre_activations}
        )
        pre_group_act.loc[:, ["group"]] = [idx_to_group[idx] for idx in pre_indices]
        # mean per group
        pre_group_act = pre_group_act.groupby("group").activation.mean()

        # post activations
        postnodes = set(layer_path.post)
        post_indices = [
            idx for idx, group in idx_to_group.items() if group in postnodes
        ]
        post_activations = activations[post_indices, l - 1]
        post_group_act = pd.DataFrame(
            {"idx": post_indices, "activation": post_activations}
        )
        post_group_act.loc[:, ["group"]] = [idx_to_group[idx] for idx in post_indices]
        # mean per group
        post_group_act = post_group_act.groupby("group").activation.mean()

        layer_path.loc[:, ["pre_activation"]] = layer_path.pre.map(pre_group_act)
        layer_path.loc[:, ["post_activation"]] = layer_path.post.map(post_group_act)
        out_df.append(layer_path)

    out = pd.concat(out_df)
    # reset layer numbers
    out.loc[:, ["layer"]] -= activation_start
    return out


def activated_path_for_ngl(path):
    """
    Convert a path DataFrame [with 'pre_activation' and 'post_activation'
    columns] to a format suitable for Neuroglancer visualization
    (`get_ngl_link(df_format='long')`). Neurons are coloured by their
    activation.

    Args:
        path (pd.DataFrame): A DataFrame containing the columns 'pre', 'post',
            'layer', 'pre_activation', and 'post_activation' (standard output
            from function `activations_to_df()` and
            `get_activations_for_path()`).

    Returns:
        pd.DataFrame: A DataFrame with columns 'neuron_id', 'layer', and
            'activation', suitable for Neuroglancer visualization.
    """
    dfs = []
    for l in path.layer.unique():
        path_l = path[path.layer == l]
        if l == path.layer.min():
            df = pd.DataFrame(
                {
                    "neuron_id": path_l.pre,
                    "layer": l,
                    "activation": path_l.pre_activation,
                }
            )
            # drop duplicate rows
            df = df.drop_duplicates()
            dfs.append(df)

        df = pd.DataFrame(
            {
                "neuron_id": path_l.post,
                "layer": l + 1,
                "activation": path_l.post_activation,
            }
        )
        df = df.drop_duplicates()
        dfs.append(df)
    return pd.concat(dfs)


def get_input_activation(
    model_in: npt.NDArray | torch.Tensor,
    sensory_indices: arrayable,
    idx_to_group: dict,
    selected_indices: arrayable | None = None,
    activation_threshold: float = 0,
) -> pd.DataFrame:
    """
    Get selected input activation.

    Args:
        model_in (numpy.ndarray | torch.Tensor): The input to the model. Shape should be
            (num_neurons, num_timepoints).
        sensory_indices (arrayable): All the indices of sensory neurons. Length should
            be equal to the first dimension of model_in.
        idx_to_group (dict): A dictionary mapping indices from the model to the groups
            of interest (e.g. cell type). max(idx_to_group.keys()) should be equal to
            number of units in the model.
        selected_indices (arrayable, optional): The indices of the neurons to select.
            Defaults to None. If None, all neurons are selected.
        activation_threshold (float, optional): A threshold value for activation.
            Neurons with activations below this threshold are not considered. Defaults
            to 0.

    Notes:
        - If there are two neurons in a cell type, one passes the activation threshold
            and the other doesn't, only the one that passes the threshold is selected.

    Returns:
        pd.DataFrame: The input activation for the model, with the first
            columns being group and sensory_indices. The rest are the
            timesteps.
    """
    sensory_indices = to_nparray(sensory_indices)

    if isinstance(model_in, torch.Tensor):
        model_in = model_in.cpu().detach().numpy()

    # check the shape of model_in, if not correct raise an error
    if len(model_in.shape) != 2:
        raise ValueError(
            f"Expected input shape (num_neurons={len(sensory_indices)}, num_timepoints)"
        )

    global2local = {glo: lo for lo, glo in enumerate(sensory_indices)}
    local2global = {lo: glo for lo, glo in enumerate(sensory_indices)}

    # selection boolean
    if selected_indices is not None:
        selected_local = [global2local[idx] for idx in selected_indices]
        selection_bool = np.zeros(len(sensory_indices), dtype=bool)
        selection_bool[selected_local] = True
    else:
        selection_bool = np.ones(len(sensory_indices), dtype=bool)

    # activation boolean
    # if any >= threhsold for each row
    activation_bool = np.any(model_in >= activation_threshold, axis=1)

    selection = selection_bool & activation_bool

    local_selected_indices = [
        i for i, s in zip(range(len(sensory_indices)), selection) if s
    ]
    global_selected_indices = [local2global[i] for i in local_selected_indices]
    groups = [idx_to_group[idx] for idx in global_selected_indices]

    # out dataframe
    out = pd.DataFrame(model_in[selection, :], index=groups)
    out.index.name = "group"
    # group by group/index, take average
    out = out.groupby("group").mean()
    # change column names
    out.columns = [f"time_{i}" for i in range(model_in.shape[1])]
    return out


def add_sign(inprop: sparse.spmatrix, idx_to_sign: dict):
    """
    Add sign to the inprop matrix based on the idx_to_sign dictionary, *and transpose
    it* such that the pre is in the columns (required by torch).

    Args:
        inprop (scipy.sparse matrix): The matrix to which the sign will be added.
        idx_to_sign (dict): A dictionary mapping indices to their sign (-1 or 1).

    Returns:
        scipy.sparse matrix: The matrix with the sign added.
    """
    # add negative connections
    neg_indices = [idx for idx, val in idx_to_sign.items() if val == -1]
    inpropcoo = inprop.tocoo().copy()
    mask = np.isin(inpropcoo.row, neg_indices)

    data = inpropcoo.data.copy()
    data[mask] = -inpropcoo.data[mask].copy()
    # make a signed sparse matrix
    inprop_signed = coo_matrix(
        (data, (inpropcoo.row, inpropcoo.col)), shape=inpropcoo.shape
    ).transpose()
    return inprop_signed
