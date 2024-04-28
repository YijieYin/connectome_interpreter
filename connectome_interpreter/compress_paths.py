"""This module is about compressing the paths of different lengths (either taking into account the sign of connections or not), and looking at the result (`result_summary()`).
"""

import torch
from tqdm import tqdm
import pandas as pd
import plotly.express as px
from scipy.sparse import issparse
import ipywidgets as widgets
import seaborn as sns
import matplotlib.pyplot as plt

from .utils import dynamic_representation, torch_sparse_where, to_nparray, tensor_to_csc


def compress_paths(inprop, step_number, threshold=0, output_threshold=1e-4):
    """
    Performs iterative multiplication of a sparse matrix `inprop` for a specified number of steps, 
    applying thresholding to filter out values below a certain `threshold` to optimize memory usage 
    and computation speed. The function is optimized to run on GPU if available. 

    This function multiplies the connectivity matrix (input in rows; output in columns) `inprop` with itself `step_number` times, 
    with each step's result being thresholded to zero out elements below a given threshold. 
    The function stores each step's result in a list, where each result is further 
    processed to drop values below the `output_threshold` to save memory.

    Args:
        inprop (scipy.sparse.matrix): The connectivity matrix as a scipy sparse matrix.
        step_number (int): The number of iterations to perform the matrix multiplication.
        threshold (float, optional): The threshold value to apply after each multiplication. 
                                     Values below this threshold are set to zero. Defaults to 0.
        output_threshold (float, optional): The threshold value to apply to the final output, 
                                            used to reduce memory footprint. Defaults to 1e-4.

    Returns:
        list: A list of scipy.sparse.csc_matrix objects, each representing connectivity from all neurons to all neurons n steps away. 

    Note:
        This function requires PyTorch and is designed to automatically utilize CUDA-enabled GPU devices 
        if available to accelerate computations. The input matrix `inprop` is converted to a dense tensor 
        before processing. 

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> import numpy as np
        >>> inprop = csr_matrix(np.array([[0.1, 0.2], [0.3, 0.4]]))
        >>> step_number = 2
        >>> compressed_paths = compress_paths(inprop, step_number, threshold=0.1, output_threshold=0.01)
        >>> print(compressed_paths)
    """
    steps_fast = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inprop_tensor = torch.tensor(inprop.toarray(), device=device)

    for i in tqdm(range(step_number)):
        if i == 0:
            out_tensor = inprop_tensor
        else:
            out_tensor = torch.matmul(out_tensor, inprop_tensor)

            # # Downgrade the previous one to save memory
            # steps_fast[-1] = torch.where(steps_fast[-1] >= output_threshold,
            #                              steps_fast[-1], torch.tensor(0.0, device=device))
            # steps_fast[-1] = steps_fast[-1].to_sparse()

        # Thresholding during matmul
        if threshold != 0:
            out_tensor = torch.where(
                out_tensor >= threshold, out_tensor, torch.tensor(0.0, device=device))

        # Thresholding for output
        out_csc = torch.where(out_tensor >= output_threshold,
                              out_tensor, torch.tensor(0.0, device=device))
        out_csc = tensor_to_csc(out_csc.to('cpu'))
        steps_fast.append(out_csc)

    torch.cuda.empty_cache()
    return steps_fast


def compress_paths_signed(inprop, idx_to_sign, target_layer_number, threshold=0, output_threshold=1e-4):
    """
    Calculates the excitatory and inhibitory influences across specified layers of a neural network, using PyTorch for GPU acceleration. This function processes a connectivity matrix (where presynaptic neurons are represented by rows and postsynaptic neurons by columns) to distinguish and compute the influence of excitatory and inhibitory neurons at each layer.

    Args:
        inprop (scipy.sparse.csc_matrix): The initial connectivity matrix representing direct connections between adjacent layers. 
        idx_to_sign (dict): A dictionary mapping neuron indices to their types (1 for excitatory, -1 for inhibitory), used to differentiate between excitatory and inhibitory influences.
        target_layer_number (int): The number of layers through which to calculate influences, starting from the second layer (with the first layer's influence, the direct connectivity, being defined by `inprop`).
        threshold (float, optional): A value to threshold the influences; influences below this value are set to zero, and not passed on in future layers. Defaults to 0.
        output_threshold (float, optional): A threshold for the final output to reduce memory usage, with values below this threshold set to zero. Defaults to 1e-4.

    Returns:
        Tuple[List[scipy.sparse.csc_matrix], List[scipy.sparse.csc_matrix]]: Two lists of sparse matrices representing the excitatory and inhibitory influences, respectively, up to the specified target layer. 

    Note:
        This function is ideal with GPU support. Ensure your environment supports CUDA and that PyTorch is correctly installed.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inprop_tensor = torch.tensor(
        inprop.toarray(), device=device, dtype=torch.float32)

    # Create masks for excitatory and inhibitory neurons
    n_neurons = inprop_tensor.shape[0]
    excitatory_mask = torch.tensor([1 if idx_to_sign[i] == 1 else 0 for i in range(
        n_neurons)], dtype=torch.float32, device=device)
    inhibitory_mask = torch.tensor(
        [1 if idx_to_sign[i] == -1 else 0 for i in range(n_neurons)], dtype=torch.float32, device=device)
    print('Number of excitatory neurons: ' +
          str(excitatory_mask.unique(return_counts=True)[1][1].cpu().numpy()))
    print('Number of inhibitory neurons: ' +
          str(inhibitory_mask.unique(return_counts=True)[1][1].cpu().numpy()))

    lne = torch.matmul(torch.diag(excitatory_mask), inprop_tensor)
    lni = torch.matmul(torch.diag(inhibitory_mask), inprop_tensor)

    steps_excitatory = []
    steps_inhibitory = []

    for layer in tqdm(range(target_layer_number)):
        if layer != 0:
          # if layer ==0, return the direct excitatory and inhibitory connections separately
            lne_new = torch.matmul(
                lne, steps_excitatory[0]) + torch.matmul(lni, steps_inhibitory[0])
            lni_new = torch.matmul(
                lne, steps_inhibitory[0]) + torch.matmul(lni, steps_excitatory[0])

            # Apply thresholding
            if threshold > 0:
                lne_new = torch_sparse_where(lne_new, threshold)
                lni_new = torch_sparse_where(lni_new, threshold)

            # Dynamic representation based on density
            lne = dynamic_representation(lne_new)
            lni = dynamic_representation(lni_new)
            torch.cuda.empty_cache()

        # steps_excitatory.append(lne)
        # steps_inhibitory.append(lni)

        # Thresholding for output
        stepe_csc = torch_sparse_where(lne, output_threshold)
        stepe_csc = tensor_to_csc(stepe_csc.to('cpu'))
        steps_excitatory.append(stepe_csc)

        stepi_csc = torch_sparse_where(lni, output_threshold)
        stepi_csc = tensor_to_csc(stepi_csc.to('cpu'))
        steps_inhibitory.append(stepi_csc)

    #     # Downgrade previous matrices to save memory
    #     if steps_excitatory:
    #         steps_excitatory[-1] = torch_sparse_where(
    #             steps_excitatory[-1], output_threshold).to_sparse()
    #         steps_inhibitory[-1] = torch_sparse_where(
    #             steps_inhibitory[-1], output_threshold).to_sparse()

    # # Downgrade the last one to save memory
    # steps_excitatory[-1] = torch_sparse_where(
    #     steps_excitatory[-1], output_threshold).to_sparse()
    # steps_inhibitory[-1] = torch_sparse_where(
    #     steps_inhibitory[-1], output_threshold).to_sparse()
    torch.cuda.empty_cache()

    return steps_excitatory, steps_inhibitory


def add_first_n_matrices(matrices, n):
    """
    Adds the first N connectivity matrices from a list, supporting different path lengths. This function is designed to work with scipy sparse matrices, and dense numpy matrices. Each matrix in the list represents connectivity information for a specific path length.

    Args:
        matrices (list): A list of connectivity matrices of different path lengths. The matrices can be scipy sparse matrices, or dense numpy arrays. The function expects all matrices in the list to be of the same type and shape.
        n (int): The number of initial matrices in the list to be summed. This number must not exceed the length of the matrices list.

    Returns:
        matrix: The resulting matrix after summing the first N matrices. The type of the returned matrix matches the type of the input matrices (scipy sparse matrix, or numpy array).

    Raises:
        ValueError: If the list of matrices is empty or if n is larger than the number of matrices available in the list.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> matrices = [csr_matrix([[1, 2], [3, 4]]), csr_matrix([[5, 6], [7, 8]]), csr_matrix([[9, 10], [11, 12]])]
        >>> n = 2
        >>> result_matrix = add_first_n_matrices(matrices, n)
        >>> print(result_matrix.toarray())
        [[ 6  8]
        [10 12]]

    Note:
        Ensure that all matrices in the list are of compatible types and shapes before using this function.
    """

    if not matrices:
        raise ValueError("The list of matrices is empty")
    if n > len(matrices):
        raise ValueError("n is larger than the number of matrices available")

    sum_matrix = matrices[0].copy()
    for i in range(1, n):
        sum_matrix += matrices[i]

    return sum_matrix


def result_summary(stepsn, inidx, outidx, inidx_map=None, outidx_map=None, display_output=True, sort_by_column=None, pre_in_column=False, display_threshold=0):
    """
    Generates a summary of connections between different types of neurons, 
    represented by their input and output indexes. The function calculates 
    the total synaptic input from presynaptic neuron groups to an average neuron in each 
    postsynaptic neuron group.

    Args:
        stepsn (scipy.sparse matrix or numpy.ndarray): Matrix representing the synaptic strengths 
            between neurons, can be dense or sparse.
        inidx (int, float, list, set, numpy.ndarray, or pandas.Series): Array of indices representing the input (presynaptic) neurons, used to subset stepsn. nan values are removed.
        outidx (int, float, list, set, numpy.ndarray, or pandas.Series): Array of indices representing the output (postsynaptic) neurons.
        inidx_map (dict, optional): Mapping from indices to neuron groups for the input neurons. Defaults to None, in which case neurons are not grouped. 
        outidx_map (dict, optional): Mapping from indices to neuron groups for the output neurons.
            Defaults to None, in which case it is set to be the same as inidx_map.
        display_output (bool, optional): Whether to display the output in a coloured dataframe. Defaults to True.
        sort_by_column (str or list, optional): the column name(s) to sort the result by. If none is provided, then sort by the first column. 
        pre_in_column (bool, optional): Whether to have the presynaptic neuron groups as columns. Defaults to False (pre in rows, post in columns).
        display_threshold (float, optional): The threshold for displaying the output. Defaults to 0.

    Returns:
        pd.DataFrame: A dataframe representing the summed synaptic input from presynaptic neuron groups 
            to an average neuron in each postsynaptic neuron group. This dataframe is always returned, regardless of the
            value of display_output.

    Displays:
        If display_output is True, the function will display a styled version of the resulting dataframe.
    """
    if inidx_map is None:
        inidx_map = {idx: idx for idx in inidx}
    if outidx_map is None:
        outidx_map = inidx_map

    # remove nan values in inidx and outidx
    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    if issparse(stepsn):
        matrix = stepsn[:, outidx][inidx, :].toarray()
    else:
        matrix = stepsn[inidx, :][:, outidx]

    # Create the dataframe
    df = pd.DataFrame(data=matrix,
                      # choose what to group by here
                      index=[inidx_map[key] for key in inidx],
                      columns=[outidx_map[key] for key in outidx])

    # Sum across rows: presynaptic neuron is in the rows
    # summing across neurons of the same type: total amount of input from that type for the postsynaptic neuron
    summed_df = df.groupby(df.index).sum()

    # Average across columns and transpose back
    # averaging across columns of the same type:
    # on average, a neuron of that type receives x% input from a presynaptic type
    result_df = summed_df.T.groupby(level=0).mean().T

    if pre_in_column:
        result_df = result_df.T

    if display_threshold > 0:
        # only display rows where any value >= display_threshold
        result_df = result_df[(result_df >= display_threshold).any(axis=1)]

    if sort_by_column is None:
        # sort result_df by the values in the first column, in descending order
        result_df = result_df.sort_values(
            by=result_df.columns[0], ascending=False)
    elif isinstance(sort_by_column, str):
        sort_by_column = [sort_by_column]

    if sort_by_column is not None:
        if set(sort_by_column).issubset(result_df.columns):
            result_df = result_df.sort_values(
                by=sort_by_column, ascending=False)
        else:
            raise ValueError(
                "sort_by_column must be present in the values of outidx_map.")

    if display_output:
        result_dp = result_df.style.background_gradient(cmap='Blues', vmin=result_df.min().min(),
                                                        vmax=result_df.max().max())
        display(result_dp)
    return result_df


def contribution_by_path_lengths(steps, inidx, outidx, outidx_map):
    """
    Analyzes the contribution of presynaptic neurons to postsynaptic neuron groups across 
    different path lengths. This function calculates and visualizes the average input 
    received by a neuron in each postsynaptic neuron group from presynaptic ones, aggregated over specified 
    path lengths. Direct connections are in path_length 1.

    Args:
        steps (list of scipy.sparse matrices): List of sparse matrices, each representing 
            synaptic strengths for a specific path length.
        inidx (int, float, list, set, numpy.ndarray, or pandas.Series): Array of indices representing input (presynaptic) neurons.
        outidx (int, float, list, set, numpy.ndarray, or pandas.Series): Array of indices representing output (postsynaptic) neurons.
        outidx_map (dict): Mapping from indices to postsynaptic neuron groups.

    Returns:
        None: Displays a line plot with path lengths on the x-axis, the average input 
            received on the y-axis, and lines differentiated by postsynaptic neuron groups.
            The plot represents how different postsynaptic neuron groups are influenced by 
            presynaptic neurons over various path lengths.

    The function iterates through each path length (each 'step'), computes the sum of inputs 
    from presynaptic to each postsynaptic neuron. It then calculates the average for each postsynaptic group. It then generates a plot showing the relationship between path lengths and synaptic input 
    contribution for each postsynaptic neuron group.
    """

    # remove nan values in inidx and outidx
    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    rows = []
    for step in steps:
        df = pd.DataFrame(data=step[:, outidx][inidx, :].toarray(),
                          # all input are grouped togehter.
                          columns=[outidx_map[key] for key in outidx])

        # Sum across rows: presynaptic neuron is in the rows
        # summing across neurons of the same type: total amount of input for the postsynaptic neuron
        # this gives a dataframe of one column, where each row is a value from outidx_map
        # there is some secret transposing after summing
        summed_df = df.sum().to_frame()

        # Average across columns and transpose back
        # averaging across columns of the same type:
        # on average, a neuron of that type receives x% input from the inidx
        result_df = summed_df.groupby(level=0).mean().T
        rows.append(result_df)

    # first have a dataframe where: each row is a path length, each column is a postsynaptic cell type
    # then pivot_wide to long
    # index is the path length
    # variable is postynaptic cell type
    # value is y axis
    contri = pd.concat(rows, ignore_index=True).melt(
        ignore_index=False).reset_index()
    contri.columns = ['path_length', 'postsynaptic_type', 'value']
    contri.path_length = contri.path_length+1

    fig = px.line(contri, x="path_length", y="value",
                  color='postsynaptic_type')
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        )
    )
    fig.show()


def contribution_by_path_lengths_heatmap(steps, inidx, outidx, inidx_map=None, outidx_map=None,
                                         sort_by_index=True, sort_by_column=None,
                                         pre_in_column=False,
                                         display_threshold=0,
                                         cmap='viridis'):
    """
    Display the contribution from inidx to outidx, grouped by inidx_map and outidx_map, across different path lengths.

    Args:
        steps (list of scipy.sparse matrices): List of sparse matrices, each representing synaptic strengths for a specific path length.
        inidx (int, float, list, set, numpy.ndarray, or pandas.Series): Array of indices representing input (presynaptic) neurons.
        outidx (int, float, list, set, numpy.ndarray, or pandas.Series): Array of indices representing output (postsynaptic) neurons.
        inidx_map (dict, optional): Mapping from indices to input neuron groups. Defaults to None, in which case neurons are not grouped.
        outidx_map (dict, optional): Mapping from indices to output neuron groups. Defaults to None, in which case it is set to be the same as inidx_map.
        sort_by_index (bool, optional): Whether to sort the output by index. Defaults to True.
        sort_by_column (str or list, optional): the column name(s) to sort the result by. If none is provided, then sort by the first column.
        pre_in_column (bool, optional): Whether to have the presynaptic neuron groups as columns. Defaults to False (pre in rows, post in columns).
        display_threshold (float, optional): The threshold for displaying the output. Defaults to 0.
        cmap (str, optional): The colormap to use for the heatmap. Defaults to 'viridis'.

    Returns:
        None: Displays an interactive heatmap showing the contribution from inidx to outidx, grouped by inidx_map and outidx_map, across different path lengths.

    """

    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    if inidx_map is None:
        inidx_map = {idx: idx for idx in inidx}
    if outidx_map is None:
        outidx_map = inidx_map

    def plot_heatmap(index):
        plt.figure(figsize=(30, 15))
        # plt.imshow(heatmaps[index], cmap='viridis', aspect = 'auto')
        # Use seaborn's heatmap function which is a higher-level API for Matplotlib's imshow
        sns.heatmap(heatmaps[index-1], annot=True, fmt=".2f",
                    linewidths=.5, cmap=cmap)

        # Rotate the tick labels for the columns to show them better
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        # Show the heatmap
        plt.show()

    heatmaps = []

    for step in tqdm(steps):
        df = result_summary(step, inidx, outidx,
                            inidx_map, outidx_map,
                            display_output=False,
                            sort_by_column=sort_by_column,
                            pre_in_column=pre_in_column,
                            display_threshold=display_threshold)
        if sort_by_index:
            df.sort_index(inplace=True)
        heatmaps.append(df)

    slider = widgets.IntSlider(value=1, min=1, max=len(heatmaps), step=1,
                               description='Path length', continuous_update=True)

    # Link the slider to the plotting function
    widgets.interactive(plot_heatmap, index=slider)
