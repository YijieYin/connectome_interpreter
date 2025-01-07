from dataclasses import dataclass
from typing import Union, Dict, Optional, Callable, Tuple, List, Collection
from numbers import Real

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from tqdm import tqdm

from .compress_paths import result_summary
from .utils import adjacency_df_to_el, get_activations, to_nparray

# types that can be made into a numeric numpy array
arrayable = Real | Collection[Real]


class MultilayeredNetwork(nn.Module):
    """
    A PyTorch module representing a multilayered neural network model.
    This network architecture is designed to process temporal sequences of sensory data
    through multiple layers, with the initial layer handling only external inputs and subsequent
    layers processing external+internal input. 

    The forward pass of the network unrolls the connectome through time, with each layer
    receiving its own time-specific sensory input.

    Attributes:
        all_weights (torch.nn.Parameter): The connectome. Input neurons are in the columns.
        sensory_indices (list[int]): Indices indicating which rows/columns in the all_weights matrix
            correspond to sensory neurons.
        num_layers (int): The number of layers in the network.
        threshold (float): The activation threshold for neurons in the network.
        activations (numpy.ndarray): A 2D array storing the activations of all neurons (rows) across time steps (columns).

    Args:
        all_weights (torch.Tensor): The connectome. Input neurons are in the columns.
        sensory_indices (list[int]): A list indicating the indices of sensory neurons
            within the network.
        num_layers (int, optional): The number of temporal layers to unroll the network through.
            Defaults to 2.
        threshold (float, optional): The threshold for activation of neurons. Defaults to 0.01.
    """

    def __init__(self, all_weights, sensory_indices, num_layers=2, threshold=0.01, tanh_steepness=5):
        super(MultilayeredNetwork, self).__init__()

        # num_neurons x num_neurons = e.g. 18k * 18k, ~1.2GB
        self.all_weights = all_weights  # this does not require grad
        self.sensory_indices = torch.tensor(
            sensory_indices)  # shape: vector of sensory indices. These are the ones we manipulate
        self.num_layers = num_layers
        self.threshold = threshold
        self.tanh_steepness = tanh_steepness
        self.activations = []

    def _process_single(self, inputs: torch.Tensor):
        """
        Process a single input sequence (2D tensor).

        Args:
            inputs (torch.Tensor): A 2D tensor (number of sensory neurons, number of time steps)
        """
        # Clear activations list at each forward pass
        acts = []

        # thresholded relu and tanh
        inputs = torch.where(inputs >= self.threshold,
                             inputs, torch.zeros_like(inputs))
        # Limit the range between 0 and 1
        inputs = torch.tanh(inputs)

        # Initial activations for first time step
        # shape: (all_neurons, 1)
        x = self.all_weights[:, self.sensory_indices] @ inputs[:, 0:1]
        # thresholded relu
        x = torch.where(x >= self.threshold, x, torch.zeros_like(x))
        # Limit the range between 0 and 1
        x = torch.tanh(self.tanh_steepness * x)
        acts.append(x)

        # Process remaining layers
        for i in range(1, self.num_layers):
            # shape: (all_neurons, all_neurons) * (all_neurons, 1) = (all_neurons, 1)
            x = self.all_weights @ x
            x[self.sensory_indices, :] += inputs[:, i:i+1]
            # thresholded relu and tanh
            x = torch.where(x >= self.threshold, x, torch.zeros_like(x))
            x = torch.tanh(self.tanh_steepness * x)
            acts.append(x)

        # free up memory as much as possible
        del inputs
        del x
        torch.cuda.empty_cache()

        # Stack activations
        acts = torch.cat(acts, dim=-1)
        return acts

    def forward(self, inputs: torch.Tensor):
        """
        Process inputs through the network. Automatically handles both 2D and 3D inputs.

        Args:
            inputs (torch.Tensor): Either a 2D tensor (num_input, time_steps) or 
                                 a 3D tensor (batch_size, num_input, time_steps)

        Returns:
            torch.Tensor: The activations of all neurons across time steps (in batches).
        """
        if inputs.dim() == 2:
            self.activations = self._process_single(inputs)
            return self.activations
        elif inputs.dim() == 3:
            # Use vmap for batched processing
            batch_forward = torch.vmap(self._process_single)
            self.activations = batch_forward(inputs)
            return self.activations
        else:
            raise ValueError(
                f"Expected 2D or 3D input tensor, got {inputs.dim()}D")


@dataclass
class TargetActivation:
    targets: Union[Dict[int, Dict[int, float]], pd.DataFrame]
    batch_size: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.targets, dict):
            self.batch_size = self.batch_size or 1
            rows = []
            for layer, neurons in self.targets.items():
                for neuron, value in neurons.items():
                    for batch in range(self.batch_size):
                        rows.append({
                            'batch': batch,
                            'layer': layer,
                            'neuron': neuron,
                            'value': value
                        })
            self.targets_df = pd.DataFrame(rows)
        else:
            required_cols = ['layer', 'neuron', 'value']
            if not all(col in self.targets.columns for col in required_cols):
                raise ValueError(
                    f"DataFrame must contain columns: {required_cols}")

            if 'batch' in self.targets.columns:
                self.batch_size = self.batch_size or (
                    self.targets['batch'].max() + 1)
                self.targets_df = self.targets.copy()
            else:
                self.batch_size = self.batch_size or 1
                self.targets_df = pd.DataFrame([
                    {**row, 'batch': i}
                    for i in range(self.batch_size)
                    for _, row in self.targets.iterrows()
                ])

    def get_batch_targets(self, batch_idx: int) -> Dict[int, Dict[int, float]]:
        batch_data = self.targets_df[self.targets_df['batch'] == batch_idx]
        result = {}
        for _, row in batch_data.iterrows():
            if row['layer'] not in result:
                result[row['layer']] = {}
            result[row['layer']][row['neuron']] = row['value']
        return result


def activation_maximisation(
        model,
        target_activations: TargetActivation,
        input_tensor: Optional[torch.Tensor] = None,
        num_iterations: int = 50,
        learning_rate: float = 0.1,
        # regularization
        in_reg_lambda: float = 0.01,
        out_reg_lambda: float = 0.01,
        custom_reg_functions: Optional[Dict[str, Callable[[
            torch.Tensor], torch.Tensor]]] = None,
        # early stopping
        early_stopping: bool = True,
        stopping_threshold: float = 1e-5,
        n_runs: int = 10,
        use_tqdm: bool = True,
        print_output: bool = True,
        report_memory_usage: bool = False,
        device: Optional[torch.device] = None,
        wandb: bool = True) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], List[float], List[np.ndarray]]:
    """
    Performs activation maximisation on a given model to identify input patterns that result in the target activations.

    This is done by adjusting the input tensor over `num_iterations` using gradient descent, while also regularising the overall input and output (to keep activated neurons sparse). The function supports early stopping based on a threshold to prevent unnecessary computations if the activation change becomes negligible.

    Args:
        model: A PyTorch model with `activations`, `sensory_indices`, and `threshold` attributes.
        target_activations (TargetActivation): Target activations specification. 
        input_tensor (torch.Tensor, optional): The initial tensor to optimize. If None, a random tensor is created.
            Defaults to None.
        num_iterations (int, optional): The number of iterations to run the optimization for. Defaults to 50.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.1.
        in_reg_lambda (float, optional): The coefficient for input regularization. Defaults to 0.01. 
        out_reg_lambda (float, optional): The coefficient for output regularization. Defaults to 0.01.
        custom_reg_functions (Dict[str, Callable[[torch.Tensor], torch.Tensor]], optional): A dictionary with keys 'in' and 'out' that
            map to functions that calculate the input and output regularization losses, respectively. If None, the default
            regularization function (L1 plus L2) is used. Defaults to None.
        early_stopping (bool, optional): Whether to stop the optimization early if the difference between the biggest and the smallest loss within the last n_runs falls below `stopping_threshold`.
            Defaults to True.
        stopping_threshold (float, optional): The threshold for early stopping. Defaults to 1e-6.
        n_runs (int, optional): The number of runs to consider for early stopping. Defaults to 10.
        use_tqdm (bool, optional): Whether to use tqdm progress bars to track optimization progress. Defaults to True.
        print_output (bool, optional): Whether to print loss information during optimization. Defaults to True.
        report_memory_usage (bool, optional): Whether to report GPU memory usage during optimization. Defaults to False.
        device: The device to run the optimization on. If None, automatically selects a device. Defaults to None.
        wandb (bool, optional): Whether to log optimization details to Weights & Biases (https://wandb.ai/site/). Defaults to True. Requires wandb to be installed.

    Returns:
        tuple: A tuple containing:

            - numpy.ndarray: The optimized input as a numpy array.
            - numpy.ndarray: The output of the model after optimization as a numpy array.
            - list(float): A list of output activation losses over iterations.
            - list(float): A list of input activation regularization losses over iterations.
            - list(float): A list of output activation regularization losses over iterations.
            - list(numpy.ndarray): A list of input tensor snapshots taken during optimization.


    Examples:

        .. code-block:: python

            # Single target for multiple batches
            targets_dict = {
                0: {1: 0.5, 2: 0.8},  # layer 0: neuron 1 -> 0.5, neuron 2 -> 0.8
                1: {0: 0.3}           # layer 1: neuron 0 -> 0.3
            }
            targets = TargetActivation(targets=targets_dict, batch_size=4)
            inputs, outputs, *losses = activation_maximisation(model, targets)

            # Different targets per batch using DataFrame
            targets_df = pd.DataFrame([
                {'batch': 0, 'layer': 0, 'neuron': 1, 'value': 0.5},
                {'batch': 0, 'layer': 0, 'neuron': 2, 'value': 0.8},
                {'batch': 1, 'layer': 1, 'neuron': 0, 'value': 0.3}
            ])
            targets = TargetActivation(targets=targets_df)  # batch_size inferred
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
        # turn batch_idx into int
        batch_idx = int(batch_idx)
        batch_targets = target_activations.get_batch_targets(batch_idx)
        loss = torch.tensor(0.0, device=device)
        n_neurons = 0

        for layer, neuron_targets in batch_targets.items():
            for neuron_index, target_value in neuron_targets.items():
                actual_value = activations[batch_idx,
                                           int(neuron_index), int(layer)]
                loss += (actual_value - target_value) ** 2
                # this scales with the number of neurons
                # so need to divide by the number of neurons
                n_neurons += 1

        return loss / n_neurons if n_neurons > 0 else loss

    # if using Weight & Biases
    if wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Please install it using 'pip install wandb' or set wandb=False."
            ) from exc

        wandb.init(project="connectome_interpreter")

        wandb.log({'learning_rate': learning_rate,
                  'input_regularization_lambda': in_reg_lambda,
                   'output_regularization_lambda': out_reg_lambda, })

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = target_activations.batch_size

    if input_tensor is None:
        # Make sure this function returns the correct input size for your model
        input_tensor = torch.rand(
            (batch_size, len(model.sensory_indices), model.num_layers),
            requires_grad=True, device=device)
    else:
        # check the shape of the input tensor
        if input_tensor.shape != (batch_size, len(model.sensory_indices), model.num_layers):
            raise ValueError(
                f"Expected input shape (batch_size={batch_size}, "
                f"num_input_neurons={len(model.sensory_indices)}, "
                f"num_layers={model.num_layers})")

    optimizer = torch.optim.Adam([input_tensor], lr=learning_rate)

    input_snapshots = []
    act_loss = []
    out_reg_losses = []
    in_reg_losses = []
    losses = []
    iteration_range = range(num_iterations)

    if report_memory_usage:
        print('GPU memory before optimization:', torch.cuda.memory_allocated(
            device) / 1e9, 'GB')

    if use_tqdm:
        iteration_range = tqdm(iteration_range)

    for iteration in iteration_range:
        optimizer.zero_grad()
        # Forward pass
        _ = model(input_tensor)

        # Take a snapshot of the input tensor every 5 iterations
        if iteration % 5 == 0:
            # Clone the current state of input_tensor and detach it from the computation graph
            snapshot = input_tensor.clone().detach().cpu().numpy()
            snapshot = np.where(snapshot >= model.threshold,
                                snapshot, 0)  # Thresholded ReLU
            # Limit the range between 0 and 1
            snapshot = np.tanh(snapshot)
            input_snapshots.append(snapshot)

        # Calculate activation loss ----
        activation_loss = torch.mean(torch.stack([
            calculate_activation_loss(model.activations, batch_idx)
            for batch_idx in range(batch_size)
        ]))

        # regularisation loss ----
        if custom_reg_functions and 'in' in custom_reg_functions:
            in_reg_loss = in_reg_lambda * custom_reg_functions['in'](
                input_tensor)
        else:
            in_reg_loss = in_reg_lambda * default_reg(input_tensor)

        if custom_reg_functions and 'out' in custom_reg_functions:
            out_reg_loss = out_reg_lambda * custom_reg_functions['out'](
                model.activations)
        else:
            out_reg_loss = out_reg_lambda * default_reg(model.activations)

        loss = activation_loss + in_reg_loss + out_reg_loss
        losses.append(loss.item())

        if early_stopping and (iteration > n_runs):
            # when the difference between the max and the min < stopping_threshold
            if np.max(losses[-n_runs:]) - np.min(losses[-n_runs:]) < stopping_threshold:
                break

        if wandb:
            dct = {"activation_loss": activation_loss.item(),
                   "in_regularisation_loss": in_reg_loss.item(),
                   'out_regularisation_loss': out_reg_loss.item(),
                   "loss": loss.item()}
            wandb.log(dct)

        act_loss.append(activation_loss.item())
        out_reg_losses.append(out_reg_loss.item())
        in_reg_losses.append(in_reg_loss.item())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Optional: Print information about the optimization process
        if print_output and iteration % 10 == 0:  # Print every 10 iterations
            print(
                f"Iteration {iteration}: Activation Loss = {activation_loss.item()}, Input regularization Loss = {in_reg_loss.item()}, Output regularization Loss = {out_reg_loss.item()}")

        if report_memory_usage and iteration % 10 == 0:
            print(f'GPU memory at iteration {iteration}:', torch.cuda.memory_allocated(
                device) / 1e9, 'GB')

        torch.cuda.empty_cache()

    # print final loss
    print(
        f'Activation loss: {act_loss[-1]}, Input regularization loss: {in_reg_losses[-1]}, Output regularization loss: {out_reg_losses[-1]}')

    input_tensor = torch.where(input_tensor >= model.threshold,
                               input_tensor, 0)
    # Limit the range between 0 and 1
    input_tensor = torch.tanh(input_tensor)

    output_after = model(input_tensor).cpu().detach().numpy()
    input_tensor = input_tensor.cpu().detach().numpy()

    if report_memory_usage:
        print('GPU memory after optimization:', torch.cuda.memory_allocated(
            device) / 1e9, 'GB')

    return (input_tensor, output_after,
            act_loss, out_reg_losses, in_reg_losses, input_snapshots)


def activations_to_df(inprop, model_input: np.ndarray, out: np.ndarray, sensory_indices: List[int],
                      inidx_mapping: dict = None, outidx_mapping: dict = None,
                      activation_threshold: float = 0, connectivity_threshold: float = 0,
                      high_ram: bool = True) -> pd.DataFrame:
    """
    Generates a dataframe representing the paths in a layered plot, filtering by activation and connectivity thresholds.

    This function takes the direct connectivity matrix (inprop), input neuron activity, output neuron activity, indices for sensory neurons, and mapping between input and output indices to groups. It generates a dataframe that represents the paths through the network layers.

    Args:
        inprop (scipy.sparse matrix or numpy.ndarray): Matrix representing the synaptic strengths
            between neurons, can be dense or sparse. Presynaptic is in the rows, postsynaptic in the columns.
        model_input (numpy.ndarray): A 2D array representing input to the network. Neurons are in the rows, timepoints in the columns. Only the first timepoint is used, since `out` is expected to have activity of all neurons, including input neurons.
        out (numpy.ndarray): A 2D array representing the output from the network. The second dimension represents timepoints.
        sensory_indices (list of int): A list of indices corresponding to sensory neurons in `inprop`.
        inidx_mapping (dict, optional): A dictionary mapping indices in `inprop` to new indices (e.g. cell type). If None, indices are not remapped.
                                       Defaults to None.
        outidx_mapping (dict, optional): A dictionary mapping indices in `out` to new indices. If None, `inidx_mapping` is used for
                                         mapping. Defaults to None.
        activation_threshold (float, optional): A threshold value for activation. Neurons with activations below this threshold are
                                                not considered. Defaults to 0.
        connectivity_threshold (float, optional): A threshold for filtering connections. Connections with weights below this threshold
                                                  are ignored. Defaults to 0.
        high_ram (bool, optional): Whether to use a high RAM implementation (which is slightly faster). This implementation gets direct connections between *all* relevant neurons at once, instead of within each layer. Defaults to True.

    Returns:
        pandas.DataFrame: A dataframe representing the paths in the network. Each row is a connection, with columns for 'pre' and
                          'post' neuron indices, 'layer', and their respective activations ('pre_activation', 'post_activation').
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
        out = out.cpu().numpy()

    print('Getting activations...')
    # get activations from input and out, based on the mappings provided by inidx_mapping and outidx_mapping
    sensory_act = get_activations(
        model_input, sensory_indices, inidx_mapping, threshold=activation_threshold)
    all_act = get_activations(
        out, all_indices, outidx_mapping, threshold=activation_threshold)

    print('Getting connectivity... If this takes a while, consider increasing the connectivity/activation threshold.')

    if high_ram:
        # get all pre and post indices across layers
        post_groups = {key for _, d in all_act.items() for key in d}
        pre_groups = set(sensory_act[0].keys()).union(post_groups)
        pre_indices = [idx for idx, key in inidx_mapping.items()
                       if key in pre_groups]
        post_indices = [idx for idx,
                        key in outidx_mapping.items() if key in post_groups]

        # if there are literally no neurons
        if len(pre_indices) == 0 or len(post_indices) == 0:
            raise ValueError(
                "No neurons found. Consider lowering the activation threshold.")

        # get connectivity
        conn = result_summary(inprop, inidx=pre_indices, outidx=post_indices,
                              inidx_map=inidx_mapping, outidx_map=outidx_mapping,
                              display_output=False)
        # turn to edgelist, and filter
        conn_el = adjacency_df_to_el(
            conn, threshold=connectivity_threshold)

    # make paths df
    paths = []
    for layer in range(out.shape[1]):
        if layer == 0:
            # the initial layer only has input neuron activations
            pre = sensory_act[layer]
            post = all_act[layer]
        else:
            # all the input, not just the optimised external input
            pre = all_act[layer-1]
            post = all_act[layer]

        pre_stringkeys = {str(key): val for key, val in pre.items()}
        post_stringkeys = {str(key): val for key, val in post.items()}

        if high_ram:
            # index the big connectivity matrix for this layer
            connections = conn_el[conn_el.pre.isin(
                pre_stringkeys.keys()) & conn_el.post.isin(post_stringkeys.keys())]

        else:
            # pre and post are already grouped by inidx and outidx_mapping
            # so need to recover the indices using pre, inidx_map, post, outidx_map
            pre_indices = [idx for idx,
                           val in inidx_mapping.items() if val in pre.keys()]
            post_indices = [idx for idx,
                            val in outidx_mapping.items() if val in post.keys()]

            conn = result_summary(inprop, inidx=pre_indices,
                                  outidx=post_indices, inidx_map=inidx_mapping, outidx_map=outidx_mapping, display_output=False)
            # turn to edgelist, and filter
            connections = adjacency_df_to_el(
                conn, threshold=connectivity_threshold)

        # so that direct connectivity is layer 1
        connections.loc[:, ['layer']] = layer+1
        connections.loc[:, ['pre_activation']
                        ] = connections.pre.map(pre_stringkeys)
        connections.loc[:, ['post_activation']
                        ] = connections.post.map(post_stringkeys)
        if connections.shape[0] > 0:
            paths.append(connections)
        else:
            print(f"Warning: No connections found in layer {layer+1}.")
    paths = pd.concat(paths)
    return paths


def activations_to_df_batched(inprop, opt_in: np.ndarray, out: np.ndarray, sensory_indices: List[int],
                              inidx_mapping: dict = None, outidx_mapping: dict = None,
                              activation_threshold: float = 0, connectivity_threshold: float = 0,
                              high_ram: bool = True) -> pd.DataFrame:
    """
    Generates a dataframe representing the paths in a layered plot, filtering by activation and connectivity thresholds.

    This function takes the direct connectivity matrix (inprop), optimal input neuron activity, output neuron activity, indices for sensory neurons, and mapping between input and output indices to groups. It generates a dataframe that represents the paths through the network layers.

    Args:
        inprop (scipy.sparse matrix or numpy.ndarray): Matrix representing the synaptic strengths
            between neurons, can be dense or sparse. Presynaptic is in the rows, postsynaptic in the columns.
        opt_in (numpy.ndarray): A **3D** array representing optimal input to the network. The first dimension represents the batch size, the second dimension represents input neurons, and the third dimension represents timepoints. Only the first timepoint is used, since `out` is expected to have activity of all neurons, including input neurons.
        out (numpy.ndarray): A **3D** array representing the activation of all neurons. The first dimension represents the batch size, the second dimension represents all neurons, and the third dimension represents timepoints.
        sensory_indices (list of int): A list of indices corresponding to sensory neurons in `inprop`.
        inidx_mapping (dict, optional): A dictionary mapping indices in `inprop` to new indices. If None, indices are not remapped.
                                       Defaults to None.
        outidx_mapping (dict, optional): A dictionary mapping indices in `out` to new indices. If None, `inidx_mapping` is used for
                                         mapping. Defaults to None.
        activation_threshold (float, optional): A threshold value for activation. Neurons with activations below this threshold are
                                                not considered. Defaults to 0.
        connectivity_threshold (float, optional): A threshold for filtering connections. Connections with weights below this threshold
                                                  are ignored. Defaults to 0.
        high_ram (bool, optional): Whether to use a high RAM implementation (which is slightly faster). This implementation gets direct connections between *all* relevant neurons at once, instead of within each layer. Defaults to True.

    Returns:
        pandas.DataFrame: A dataframe representing the paths in the network. Each row is a connection, with columns for 'pre' and
                          'post' neuron indices, 'layer', and their respective activations ('pre_activation', 'post_activation'). 
    """
    # use activations_to_df for each batch
    paths = []
    # assume the first dimnesion of opt_in and out is the batch size
    for i in tqdm(range(opt_in.shape[0])):
        path = activations_to_df(inprop, opt_in[i], out[i], sensory_indices,
                                 inidx_mapping, outidx_mapping, activation_threshold, connectivity_threshold, high_ram)
        path.loc[:, ['batch']] = i
        paths.append(path)
    return pd.concat(paths)


def input_from_df(df: pd.DataFrame, sensory_indices: list, idx_to_group: dict, num_layers: int, timepoints: Union[int, List[int], np.ndarray, set] = 0) -> np.ndarray:
    '''
    Make well-formatted input for the model, based on defined vectors of input neuron activation (df). The function returens a 3D tensor, with the first dimension being the batch size, the second dimension being the number of input neurons of the model, and the third dimension being the number of layers. 

    Args:
        df : pd.DataFrame
            Rows correspond to the values in `idx_to_group`, and columns correspond to the batches (number of columns = number of batch). For instance, rows of df can be olfactory glomeruli, and columns can be odours. df is the reaction of each glomerulus to each odour.
        sensory_indices : list
            The indices of sensory neurons. For instance, these could be the indices of individual olfactory receptor neurons.
        idx_to_group : dict
            A dictionary that maps indices to group. For instance, this could map from indices of indivudal olfactory receptor neuron to the glomerulus they innervate (rows of df).
        num_layers : int
            The number of layers in the model.

    Returns:
        np.ndarray: The input for the model.

    '''
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
        actvec = [grp2act[idx_to_group[idx]] if idx_to_group[idx]
                  in grp2act else 0 for idx in sensory_indices]
        for t in timepoints:
            inarray[l, :, t] = actvec
    return inarray


def get_neuron_activation(output,
                          neuron_indices: arrayable,
                          batch_names: arrayable = None,
                          idx_to_group: dict = None) -> pd.DataFrame:
    '''
    Get the activations for specified indices across timepoints, include batch name and group information when available.

    Args:
        output (torch.Tensor): The output tensor from the model. Shape should be (batch_size, num_neurons, num_timepoints) or (num_neurons, num_timepoints).
        neuron_indices (arrayable): The indices of the neurons to get activations for.
        batch_names (arrayable, optional): The names of the batches. Defaults to None. If output.ndim == 3, then this should be supplied. If not, batch names will be e.g. 'batch_0', 'batch_1', etc.
        idx_to_group (dict, optional): A dictionary mapping indices to groups. Defaults to None.

    Returns:
        pd.DataFrame: The activations for the neurons, with the first columns being batch_names, neuron_indices, and group. The rest are the timesteps. 
    '''
    batch_names = list(to_nparray(batch_names, unique=False))
    neuron_indices = list(to_nparray(neuron_indices))

    if output.ndim == 2:
        # message if batch_names is not None
        if batch_names is not None:
            print('batch_names is ignored for 2D output.')

        data = output[neuron_indices, :]
        df = pd.DataFrame(data, columns=[f'time_{i}' for i in range(
            output.shape[1])], index=neuron_indices)
        df.index.name = 'idx'
        if idx_to_group is not None:
            df['group'] = [idx_to_group[idx] for idx in neuron_indices]
        df.reset_index(inplace=True)
        # re-order columns
        if 'group' in df.columns:
            df = df[['idx', 'group'] +
                    [f'time_{i}' for i in range(output.shape[1])]]
        else:
            df = df[['idx'] + [f'time_{i}' for i in range(output.shape[1])]]
        return df

    if output.ndim == 3:
        if batch_names is None:
            # make batch names
            batch_names = [f'batch_{i}' for i in range(output.shape[0])]

        # make sure length matches
        if output.shape[0] != len(batch_names):
            raise ValueError(
                'Length of batch_names has to be the same as output.shape[0].')

        data = output.view(-1, output.shape[2]).cpu().numpy()

        # Create indices for the first two dimensions
        # in the end, we want output.shape[0] * output.shape[1] rows
        # neuron_indices are the second dimension
        # so multiply by output.shape[0]
        neuron_indices = neuron_indices * output.shape[0]
        batch_names = [o for o in batch_names for _ in range(output.shape[1])]

        # Combine indices and data into a DataFrame
        df = pd.DataFrame(
            data, columns=[f'time_{i}' for i in range(output.shape[2])])
        df['batch_name'] = batch_names
        df['idx'] = neuron_indices
        if idx_to_group is not None:
            df['group'] = [idx_to_group[idx] for idx in neuron_indices]

        # Reorder columns so indices are first
        if 'group' in df.columns:
            df = df[['batch_name', 'idx', 'group'] +
                    [f'time_{i}' for i in range(output.shape[2])]]
        else:
            df = df[['batch_name', 'idx'] +
                    [f'time_{i}' for i in range(output.shape[2])]]

    return df
