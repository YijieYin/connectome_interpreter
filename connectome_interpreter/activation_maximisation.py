from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm


class MultilayeredNetwork(nn.Module):
    """
    A PyTorch module representing a multilayered neural network model. 
    This network architecture is designed to process temporal sequences of sensory data 
    through multiple layers, with the initial layer handling sensory inputs and subsequent 
    layers processing sensory + non-sensory input.

    The forward pass of the network unrolls the connectome through time, with each layer 
    receiving its own time-specific sensory input. 

    Attributes:
        all_weights (torch.nn.Parameter): The connectome. Input neurons are in the columns. 
        sensory_indices (list[int]): Indices indicating which rows/columns in the all_weights matrix 
            correspond to sensory neurons.
        num_layers (int): The number of layers in the network.
        threshold (float): The activation threshold for neurons in the network.
        activations (list[torch.Tensor]): A list storing the activations of each layer 
            after the forward pass.

    Args:
        all_weights (torch.Tensor): The connectome. Input neurons are in the columns.
        sensory_indices (list[int]): A list indicating the indices of sensory neurons 
            within the network.
        num_layers (int, optional): The number of temporal layers to unroll the network through. 
            Defaults to 2.
        threshold (float, optional): The threshold for activation of neurons. Defaults to 0.01.

    Methods:
        forward(inthroughtime): Implements the forward pass of the multilayered network 
            using sensory input through time.
    """

    def __init__(self, all_weights, sensory_indices, num_layers=2, threshold=0.01, tanh_steepness=5):
        super(MultilayeredNetwork, self).__init__()

        self.all_weights = torch.nn.parameter.Parameter(all_weights)
        # Update sensory_indices to reflect the new order
        # Sensory neurons are now the first ones
        self.sensory_indices = torch.tensor(sensory_indices)
        # non_sensory_indices are the rest
        self.non_sensory_indices = torch.tensor(
            [i for i in range(all_weights.size(0)) if i not in sensory_indices])
        self.num_layers = num_layers
        self.threshold = threshold
        self.tanh_steepness = tanh_steepness
        self.activations = []  # List for activations of middle layers

    def forward(self, inthroughtime):
        """
        Processes the input through the multilayered neural network, applying thresholding
        and tanh activation functions. This method sequentially processes the input through
        each layer of the network, considering both sensory and non-sensory neurons, to produce
        the final activations.

        Args:
            inthroughtime (torch.Tensor): A 2D tensor representing the input to the network
                across different time steps. Each column corresponds to a different time step,
                and each row corresponds to different sensory inputs.

        Returns:
            torch.Tensor: A 2D tensor of activations for non-sensory neurons across all layers and
                time steps. Each column represents the activations for a specific time step,
                while each row corresponds to different non-sensory neurons.

        This method first applies a threshold ReLU to the input, setting all values below
        the threshold to zero. It then limits the range of activations between 0 and 1 using
        a tanh function. For each layer (after the initial),the method combines sensory and previous layer's non-sensory activations into a
        single input tensor, which is then processed to produce the next layer's non-sensory neuron activations.
        The process accounts for the model's threshold and `tanh_steepness` to modulate
        the activations. The final activations are stored in the `activations` attribute
        of the model, with each entry corresponding to the activations for a specific layer.
        """
        self.activations = []  # Clear activations list at each forward pass

        # Ensure the activation of the sensory inputs is less than 1
        # inthroughtime = torch.sigmoid(inthroughtime)
        inthroughtime = torch.where(inthroughtime >= self.threshold,
                                    inthroughtime, 0)  # Thresholded ReLU
        # Limit the range between 0 and 1
        inthroughtime = torch.tanh(inthroughtime)

        # Use broadcasting to create a meshgrid for indexing
        # The unsqueeze methods are used to align dimensions for broadcasting
        row_indices = self.non_sensory_indices.unsqueeze(
            1).expand(-1, len(self.sensory_indices))
        col_indices = self.sensory_indices.unsqueeze(
            0).expand(len(self.non_sensory_indices), -1)

        # Initial activations are based only on sensory inputs for the first time step
        x = self.all_weights[row_indices,
                             col_indices] @ inthroughtime[:, 0]
        x = torch.where(x >= self.threshold, x, 0)  # Thresholded ReLU
        # Limit the range between 0 and 1
        x = torch.tanh(self.tanh_steepness*x)
        self.activations.append(x)

        # Process remaining layers
        for i in range(1, self.num_layers):
            # Create an empty tensor to hold both sensory and non-sensory neuron activations.
            combined_input = torch.zeros(
                self.all_weights.shape[0], device=x.device, dtype=x.dtype)
            # Fill in the sensory inputs at their respective positions.
            combined_input[self.sensory_indices] = inthroughtime[:, i]
            # Fill in the non-sensory outputs at their respective positions.
            combined_input[self.non_sensory_indices] = x

            x = self.all_weights[self.non_sensory_indices, :] @ combined_input
            x = torch.where(x >= self.threshold, x, 0)  # Thresholded ReLU
            # Limit the range between 0 and 1
            x = torch.tanh(self.tanh_steepness*x)

            self.activations.append(x)

        self.activations = torch.stack(self.activations, dim=1)
        return self.activations


def activation_maximisation(
        model, selected_neurons_per_layer: Dict[int, List[int]],
        input_tensor=None,
        num_iterations=300, learning_rate=0.4,
        in_regularisation_lambda=0.1, custom_in_regularisation: Callable[[torch.Tensor], torch.Tensor] = None,
        out_regularisation_lambda=0.1,
        early_stopping=True, stopping_threshold=1e-6, n_runs=10,
        use_tqdm=True, print_output=True,
        device=None, wandb=True) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Performs activation maximisation on a given model to identify input patterns that maximally activate selected neurons.

    This method adjusts `input_tensor` over `num_iterations` to maximize the activation of specified neurons in `model`,
    while also applying custom regularization to the input and output. The function supports early stopping based on a threshold
    to prevent unnecessary computations if the activation change becomes negligible.

    Args:
        model: A PyTorch model with `activations`, `sensory_indices`, and `threshold` attributes.
        selected_neurons_per_layer (Dict[int, List[int]]): A dictionary mapping from layer indices to lists of neuron indices
            whose activations are to be maximized.
        input_tensor (torch.Tensor, optional): The initial tensor to optimize. If None, a random tensor is created.
            Defaults to None.
        num_iterations (int, optional): The number of iterations to run the optimization for. Defaults to 300.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        in_regularisation_lambda (float, optional): The regularization coefficient. Defaults to 0.4.
        custom_in_regularisation (Callable[[torch.Tensor], torch.Tensor], optional): A function that applies custom
            regularization to the `input_tensor`. Defaults to L1+L2 norm if None.
        out_regularisation_lambda (float, optional): The coefficient for the output activation loss. Defaults to 0.1.
        early_stopping (bool, optional): Whether to stop the optimization early if the difference between the biggest and the smallest loss within the last n_runs falls below `stopping_threshold`.
            Defaults to True.
        stopping_threshold (float, optional): The threshold for early stopping. Defaults to 1e-6.
        n_runs (int, optional): The number of runs to consider for early stopping. Defaults to 10.
        use_tqdm (bool, optional): Whether to use tqdm progress bars to track optimization progress. Defaults to True.
        print_output (bool, optional): Whether to print loss information during optimization. Defaults to True.
        device: The device to run the optimization on. If None, automatically selects a device. Defaults to None.
        wandb (bool, optional): Whether to log optimization details to Weights & Biases. Defaults to True. Requires wandb to be installed.

    Returns:
        A tuple containing:
        - The optimized `input_tensor` as a numpy array.
        - The output of the model after optimization as a numpy array.
        - A list of input activation losses over iterations.
        = A list of output activation losses over iterations. 
        - A list of regularization losses over iterations.
        - A list of input tensor snapshots taken during optimization.

    Raises:
        ImportError: If `wandb` is True and wandb is not installed.
    """
    if wandb:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Please install it using 'pip install wandb' or set wandb=False.")

        wandb.init(project="connectome_interpreter")

        wandb.log({'learning_rate': learning_rate,
                  'input_regularization_lambda': in_regularisation_lambda,
                   'output_regularization_lambda': out_regularisation_lambda, })
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if input_tensor is None:
        # Make sure this function returns the correct input size for your model
        input_tensor = torch.rand(
            (len(model.sensory_indices), model.num_layers), requires_grad=True, device=device)

    if custom_in_regularisation is None:
        # Define default regularization as L1+L2 if none is provided
        def custom_in_regularisation(x):
            return torch.norm(x, p=1) + torch.norm(x, p=2)

    optimizer = torch.optim.Adam([input_tensor], lr=learning_rate)

    input_snapshots = []
    act_loss = []
    out_reg_loss = []
    in_reg_loss = []
    iteration_range = range(num_iterations)
    if use_tqdm:
        iteration_range = tqdm(iteration_range)

    losses = []
    for iteration in iteration_range:
        optimizer.zero_grad()
        # Forward pass
        _ = model(input_tensor)

        if iteration % 5 == 0:
            # Clone the current state of input_tensor and detach it from the computation graph
            snapshot = input_tensor.clone().detach().cpu().numpy()
            snapshot = np.where(snapshot >= model.threshold,
                                snapshot, 0)  # Thresholded ReLU
            # Limit the range between 0 and 1
            snapshot = np.tanh(snapshot)
            input_snapshots.append(snapshot)

        activation_loss = torch.tensor(0.0, device=input_tensor.device)
        # Check if the model has an 'activations' attribute and the selected_neurons_per_layer is not empty
        if hasattr(model, 'activations') and selected_neurons_per_layer:
            for layer_index, neuron_indices in selected_neurons_per_layer.items():
                # Ensure layer index is valid
                if layer_index < len(model.activations):
                    # Get activations for this layer
                    layer_activations = model.activations[:, layer_index]
                    # Negative sign because we want to maximize activation
                    # Only select activations from specified neurons
                    activation_loss -= torch.mean(
                        layer_activations[neuron_indices])
        # in the end, activation loss is the sum of mean activation across layers.

        out_regularisation_loss = out_regularisation_lambda * torch.mean(
            model.activations)
        # Apply custom regularisation
        in_regularisation_loss = in_regularisation_lambda * custom_in_regularisation(
            input_tensor)
        loss = activation_loss + in_regularisation_loss + out_regularisation_loss
        losses.append(loss.item())

        if early_stopping and iteration > n_runs:
            # when the difference between the max and the min < stopping_threshold
            if np.max(losses[-n_runs:]) - np.min(losses[-n_runs:]) < stopping_threshold:
                break

        if wandb:
            dct = {"activation_loss": activation_loss.item(
            ), "in_regularisation_loss": in_regularisation_loss.item(),
                'out_regularisation_loss': out_regularisation_loss.item(),
                "loss": loss.item()}
            wandb.log(dct)

        act_loss.append(activation_loss.item())
        out_reg_loss.append(out_regularisation_loss.item())
        in_reg_loss.append(in_regularisation_loss.item())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Optional: Print information about the optimization process
        if print_output and iteration % 10 == 0:  # Print every 100 iterations
            print(
                f"Iteration {iteration}: Activation Loss = {activation_loss.item()}, Input regularization Loss = {in_regularisation_loss.item()}, Output regularization Loss = {out_regularisation_loss.item()}")

    output_after = model(input_tensor)
    input_tensor = torch.where(input_tensor >= model.threshold,
                               input_tensor, 0)  # Thresholded ReLU
    # Limit the range between 0 and 1
    input_tensor = torch.tanh(input_tensor)
    return (input_tensor.cpu().detach().numpy(),
            output_after.cpu().detach().numpy(),
            act_loss, out_reg_loss, in_reg_loss, input_snapshots)
