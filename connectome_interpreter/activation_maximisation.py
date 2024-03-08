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

    def __init__(self, all_weights, sensory_indices, num_layers=2, threshold=0.01):
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
        self.activations = []  # List for activations of middle layers

    def forward(self, inthroughtime):
        self.activations = []  # Clear activations list at each forward pass

        # Ensure the activation of the sensory inputs is less than 1
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
        x = torch.tanh(x)  # Limit the range between 0 and 1
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
            x = torch.tanh(x)  # Limit the range between 0 and 1

            self.activations.append(x)

        self.activations = torch.stack(self.activations, dim=1)
        return self.activations


def activation_maximisation(
        model, selected_neurons_per_layer: Dict[int, List[int]], input_tensor=None,
        num_iterations=3000, learning_rate=0.01,
        regularisation_lambda=0.1,
        custom_regularisation: Callable[[torch.Tensor], torch.Tensor] = None,
        use_tqdm=True, print_output=True,
        device=None) -> Tuple[np.ndarray, np.ndarray, list, list]:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if input_tensor is None:
        # Make sure this function returns the correct input size for your model
        input_tensor = torch.rand(
            (len(model.sensory_indices), model.num_layers), requires_grad=True, device=device)

    if custom_regularisation is None:
        # Define default regularization as L1+L2 if none is provided
        def custom_regularisation(x):
            return torch.norm(x, p=1) + torch.norm(x, p=2)

    optimizer = torch.optim.Adam([input_tensor], lr=learning_rate)

    act_loss = []
    reg_losses = []
    iteration_range = range(num_iterations)
    if use_tqdm:
        iteration_range = tqdm(iteration_range)
    for iteration in iteration_range:
        optimizer.zero_grad()
        # Forward pass
        _ = model(input_tensor)

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

        # Apply custom regularisation
        regularisation_loss = regularisation_lambda * custom_regularisation(
            input_tensor)

        loss = activation_loss + regularisation_loss
        act_loss.append(activation_loss.item())
        reg_losses.append(regularisation_loss.item())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Optional: Print information about the optimization process
        if print_output and iteration % 10 == 0:  # Print every 100 iterations
            print(
                f"Iteration {iteration}: Activation Loss = {activation_loss.item()}, Regularization Loss = {regularisation_loss.item()}")

        output_after = model(input_tensor)
    return (input_tensor.cpu().detach().numpy(),
            output_after.cpu().detach().numpy(),
            act_loss, reg_losses)
