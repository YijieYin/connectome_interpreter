# Standard library imports
import math
from typing import List
import os
import gc

# Third-party package imports
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import scipy as sp
import seaborn as sns
import torch
from IPython.display import display
from scipy.sparse import csr_matrix, issparse, csc_matrix
from tqdm import tqdm

from .utils import (
    dynamic_representation,
    group_edge_by,
    tensor_to_csc,
    to_nparray,
    torch_sparse_where,
    arrayable,
)


def compress_paths(
    inprop: csr_matrix,
    step_number: int,
    threshold: float = 0,
    output_threshold: float = 1e-4,
    root: bool = False,
    chunkSize: int = 10000,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_to_disk: bool = False,
    save_path: str = "./",
    save_prefix: str = "step_",
) -> list:
    """Performs iterative multiplication of a sparse matrix `inprop` for a
    specified number of steps, applying thresholding to filter out values below
    a certain `threshold` to optimize memory usage and computation speed.

    The function is optimized to run on GPU if available. It needs >=
    size_of_inprop * 3 amount of GPU memory, for matrix multiplication, and
    thresholding.

    This function multiplies the connectivity matrix (input in rows; output in
    columns) `inprop` with itself `step_number` times, with each step's result
    being thresholded to zero out elements below a given threshold. The
    function stores each step's result in a list, where each result is further
    processed to drop values below the `output_threshold` to save memory.

    Args:
        inprop (scipy.sparse.matrix): The connectivity matrix as a scipy
            sparse matrix.
        step_number (int): The number of iterations to perform the matrix
            multiplication.
        threshold (float, optional): The threshold value to apply after each
            multiplication. Values below this threshold are set to zero.
            Defaults to 0.
        output_threshold (float, optional): The threshold value to apply to
            the final output, used to reduce output size. Defaults to 1e-4.
        root (bool, optional): Whether to take the nth root of the output.
            This can be understood as "the average direct connection strength"
            (when root=True), as opposed to "the proportion of influence among
            all partners n steps away" (when root=False). Defaults to False.
        chunkSize (int, optional): The size of the chunks to split the matrix
            into for matrix multiplication. Defaults to 10000.
        dtype (torch.dtype, optional): The data type to use for the tensor
            calculations. Defaults to torch.float32.
        device (torch.device, optional): The device to use for the tensor
            calculations. Defaults to torch.device("cuda" if
            torch.cuda.is_available() else "cpu").
        save_to_disk (bool, optional): Whether to save the output matrices to
            disk. Defaults to False.
        save_path (str, optional): The path to save the output matrices to.
            Defaults to "./" (the current folder).
        save_prefix (str, optional): The prefix to use for the output matrix
            filenames. Defaults to "step_".

    Returns:
        list: A list of scipy.sparse.csc_matrix objects, each representing
            connectivity from all neurons to all neurons n steps away.

    Note:
        This function requires PyTorch and is designed to automatically
        utilize CUDA-enabled GPU devices if available to accelerate
        computations. The input matrix `inprop` is converted to a dense tensor
        before processing.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> import numpy as np
        >>> inprop = csr_matrix(np.array([[0.1, 0.2], [0.3, 0.4]]))
        >>> step_number = 2
        >>> compressed_paths = compress_paths(inprop, step_number,
                                              threshold=0.1,
                                              output_threshold=0.01)
        >>> print(compressed_paths)
    """
    steps_fast: List[csc_matrix] = []

    if not isinstance(inprop, csc_matrix):
        inprop = inprop.tocsc()

    # check that step_number>0
    if step_number < 1:
        raise ValueError("step_number should be greater than 0")

    size = inprop.shape[0]

    chunks = math.ceil(size / chunkSize)

    with torch.no_grad():
        for i in tqdm(range(step_number)):

            if i == 0:
                out_tensor = torch.tensor(inprop.toarray(), dtype=dtype)
            else:
                out_tensor_new = torch.zeros(size, size, dtype=dtype)
                colLow = 0
                colHigh = chunkSize
                for colChunk in range(chunks):  # iterate chunks colwise
                    rowLow = 0
                    rowHigh = chunkSize

                    in_col = torch.tensor(
                        inprop[:, colLow:colHigh].toarray(), dtype=dtype
                    ).to(device)
                    # shape: size x chunkSize; on GPU

                    for rowChunk in range(chunks):  # iterate chunks rowwise
                        in_rows = out_tensor[rowLow:rowHigh, :].to(device)
                        # shape: chunkSize x size; on GPU
                        out_tensor_new[rowLow:rowHigh, colLow:colHigh] = torch.matmul(
                            in_rows, in_col
                        ).to("cpu")
                        # shape: chunkSize x chunkSize; on CPU

                        rowLow += chunkSize
                        rowHigh += chunkSize
                        rowHigh = min(rowHigh, size)

                        del in_rows
                    del in_col
                    torch.cuda.empty_cache()
                    colLow += chunkSize
                    colHigh += chunkSize
                    colHigh = min(colHigh, size)

                out_tensor = out_tensor_new
                del out_tensor_new
                # Clear PyTorch CUDA cache
                torch.cuda.empty_cache()

            # Thresholding during matmul
            if threshold != 0:
                out_tensor = torch.where(
                    out_tensor >= threshold,
                    out_tensor,
                    torch.tensor(0.0, dtype=dtype),
                )

            # Convert to csc for output
            out_csc = tensor_to_csc(out_tensor)
            out_csc.eliminate_zeros()

            if root:
                out_csc.data = np.power(out_csc.data, 1 / (i + 1))

            if output_threshold > 0:
                out_csc.data = np.where(
                    out_csc.data >= output_threshold, out_csc.data, 0
                )
                out_csc.eliminate_zeros()

            if save_to_disk:
                sp.sparse.save_npz(
                    os.path.join(save_path, f"{save_prefix}{i}.npz"), out_csc
                )
            else:
                steps_fast.append(out_csc)
            del out_csc

    # remove all variables
    del out_tensor
    torch.cuda.empty_cache()

    return steps_fast


# below: not chunked version
def compress_paths_not_chunked(
    inprop, step_number, threshold=0, output_threshold=1e-4, root=False
):
    """As above, but without chunking.

    This would be more demanding for GPU RAM.
    """
    steps_fast = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inprop_tensor = torch.tensor(inprop.toarray(), device=device)

    with torch.no_grad():
        for i in tqdm(range(step_number)):
            if i == 0:
                out_tensor = inprop_tensor.clone()
            else:
                out_tensor = torch.matmul(out_tensor, inprop_tensor)

            # Thresholding during matmul
            if threshold != 0:
                out_tensor = torch.where(
                    out_tensor >= threshold,
                    out_tensor,
                    torch.tensor(0.0, device=device),
                )

            # Convert to csc for output
            out_csc = tensor_to_csc(out_tensor.to("cpu"))
            out_csc.eliminate_zeros()

            if root:
                out_csc.data = np.power(out_csc.data, 1 / (i + 1))

            if output_threshold > 0:
                out_csc.data = np.where(
                    out_csc.data >= output_threshold, out_csc.data, 0
                )
                out_csc.eliminate_zeros()

            steps_fast.append(out_csc)
            torch.cuda.empty_cache()

    return steps_fast


def compress_paths_signed(
    inprop,
    idx_to_sign,
    target_layer_number,
    threshold=0,
    output_threshold=1e-4,
    root=False,
):
    """Calculates the excitatory and inhibitory influences across specified
    layers of a neural network, using PyTorch for GPU acceleration. This
    function processes a connectivity matrix (where presynaptic neurons are
    represented by rows and postsynaptic neurons by columns) to distinguish and
    compute the influence of excitatory and inhibitory neurons at each layer.

    Args:
        inprop (scipy.sparse.csc_matrix): The initial connectivity matrix
        representing direct connections between adjacent layers.
        idx_to_sign (dict): A dictionary mapping neuron indices to their types
        (1 for excitatory, -1 for inhibitory), used to differentiate between
        excitatory and inhibitory influences.
        target_layer_number (int): The number of layers through which to
        calculate influences, starting from the second layer (with the first
        layer's influence, the direct connectivity, being defined by `inprop`).
        threshold (float, optional): A value to threshold the influences;
        influences below this value are set to zero, and not passed on in
        future layers. Defaults to 0.
        output_threshold (float, optional): A threshold for the final output
        to reduce memory usage, with values below this threshold set to zero.
        Defaults to 1e-4.
        root (bool, optional): Whether to take the nth root of the output.
        This can be understood as "the average direct connection strength"
        (when root=True), as opposed to "the proportion of influence among all
        partners n steps away" (when root=False). Defaults to False.

    Returns:
        Tuple[List[scipy.sparse.csc_matrix], List[scipy.sparse.csc_matrix]]:
            Two lists of sparse matrices representing the excitatory and
            inhibitory influences, respectively, up to the specified target
            layer.

    Note:
        This function is ideal with GPU support. Ensure your environment
        supports CUDA and that PyTorch is correctly installed.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inprop_tensor = torch.tensor(inprop.toarray(), device=device, dtype=torch.float32)

    # Create masks for excitatory and inhibitory neurons
    n_neurons = inprop_tensor.shape[0]
    excitatory_mask = torch.tensor(
        [1 if idx_to_sign[i] == 1 else 0 for i in range(n_neurons)],
        dtype=torch.float32,
        device=device,
    )
    inhibitory_mask = torch.tensor(
        [1 if idx_to_sign[i] == -1 else 0 for i in range(n_neurons)],
        dtype=torch.float32,
        device=device,
    )
    print(
        "Number of excitatory neurons: "
        + str(excitatory_mask.unique(return_counts=True)[1][1].cpu().numpy())
    )
    print(
        "Number of inhibitory neurons: "
        + str(inhibitory_mask.unique(return_counts=True)[1][1].cpu().numpy())
    )

    lne = torch.matmul(torch.diag(excitatory_mask), inprop_tensor)
    lni = torch.matmul(torch.diag(inhibitory_mask), inprop_tensor)

    steps_excitatory = []
    steps_inhibitory = []

    for layer in tqdm(range(target_layer_number)):
        if layer != 0:
            # if layer ==0, return the direct excitatory and inhibitory
            # connections separately
            lne_new = torch.matmul(lne, steps_excitatory[0]) + torch.matmul(
                lni, steps_inhibitory[0]
            )
            lni_new = torch.matmul(lne, steps_inhibitory[0]) + torch.matmul(
                lni, steps_excitatory[0]
            )

            # Apply thresholding
            if threshold > 0:
                lne_new = torch_sparse_where(lne_new, threshold)
                lni_new = torch_sparse_where(lni_new, threshold)

            # Dynamic representation based on density
            lne = dynamic_representation(lne_new)
            lni = dynamic_representation(lni_new)
            torch.cuda.empty_cache()

        # possible alternative implementation: first root and threshold, then
        # move to CPU. But not sure if can root sparse tensor. So:
        stepe_csc = tensor_to_csc(lne.to("cpu"))
        stepi_csc = tensor_to_csc(lni.to("cpu"))

        if root:
            stepe_csc.data = np.power(stepe_csc.data, 1 / (layer + 1))
            stepi_csc.data = np.power(stepi_csc.data, 1 / (layer + 1))

        # then threshold
        if output_threshold > 0:
            stepe_csc.data = np.where(
                stepe_csc.data >= output_threshold, stepe_csc.data, 0
            )
            stepi_csc.data = np.where(
                stepi_csc.data >= output_threshold, stepi_csc.data, 0
            )

        stepe_csc.eliminate_zeros()
        stepi_csc.eliminate_zeros()

        steps_excitatory.append(stepe_csc)
        steps_inhibitory.append(stepi_csc)

    torch.cuda.empty_cache()

    return steps_excitatory, steps_inhibitory


def add_first_n_matrices(matrices, n):
    """Adds the first N connectivity matrices from a list, supporting different
    path lengths. This function is designed to work with scipy sparse matrices,
    and dense numpy matrices. Each matrix in the list represents connectivity
    information for a specific path length.

    Args:
        matrices (list): A list of connectivity matrices of different path
            lengths. The matrices can be scipy sparse matrices, or dense numpy
            arrays. The function expects all matrices in the list to be of the
            same type and shape.
        n (int): The number of initial matrices in the list to be summed. This
            number must not exceed the length of the matrices list.

    Returns:
        matrix: The resulting matrix after summing the first N matrices. The
        type of the returned matrix matches the type of the input matrices
        (scipy sparse matrix, or numpy array).

    Raises:
        ValueError: If the list of matrices is empty or if n is larger than
            the number of matrices available in the list.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> matrices = [csr_matrix([[1, 2], [3, 4]]),
                        csr_matrix([[5, 6], [7, 8]]),
                        csr_matrix([[9, 10], [11, 12]])]
        >>> n = 2
        >>> result_matrix = add_first_n_matrices(matrices, n)
        >>> print(result_matrix.toarray())
        [[ 6  8]
        [10 12]]

    Note:
        Ensure that all matrices in the list are of compatible types and
        shapes before using this function.
    """

    if not matrices:
        raise ValueError("The list of matrices is empty")
    if n > len(matrices):
        raise ValueError("n is larger than the number of matrices available")

    sum_matrix = matrices[0].copy()
    for i in range(1, n):
        sum_matrix += matrices[i]

    return sum_matrix


def result_summary(
    stepsn,
    inidx: arrayable,
    outidx: arrayable,
    inidx_map: dict | None = None,
    outidx_map: dict | None = None,
    display_output: bool = True,
    display_threshold: float = 1e-3,
    threshold_axis: str = "row",
    sort_within: str = "column",
    sort_names: str | List | None = None,
    pre_in_column: bool = False,
    include_undefined_groups: bool = False,
):
    """Generates a summary of connections between different types of neurons,
    represented by their input and output indexes. The function calculates the
    total synaptic input from presynaptic neuron groups to an average neuron in
    each postsynaptic neuron group.

    Args:
        stepsn (scipy.sparse matrix or numpy.ndarray): Matrix representing the
            synaptic strengths between neurons, can be dense or sparse.
        inidx (int, float, list, set, numpy.ndarray, or pandas.Series): Array
            of indices representing the input (presynaptic) neurons, used to
            subset stepsn. nan values are removed.
        outidx (int, float, list, set, numpy.ndarray, or pandas.Series): Array
            of indices representing the output (postsynaptic) neurons.
        inidx_map (dict, optional): Mapping from indices to neuron groups for
            the input neurons. Defaults to None, in which case neurons are not
            grouped.
        outidx_map (dict, optional): Mapping from indices to neuron groups for
            the output neurons. Defaults to None, in which case it is set to
            be the same as inidx_map.
        display_output (bool, optional): Whether to display the output in a
            coloured dataframe. Defaults to True.
        display_threshold (float, optional): The minimum threshold for
            displaying the output. Defaults to 0.
        threshold_axis (str, optional): The axis to apply the
            display_threshold to. Defaults to 'row' (removing entire rows if
            no value exceeds display_threshold).
        sort_within (str, optional): The axis to sort the output in. Defaults
            to 'column'.
        sort_names (str or list, optional): the column/row name(s) to sort the
            result by. If none is provided, then sort by the first column/row.
        pre_in_column (bool, optional): Whether to have the presynaptic neuron
            groups as columns. Defaults to False (pre in rows, post: columns).
        include_undefined_groups (bool, optional): Whether to include
            undefined groups in the output. Defaults to False.

    Returns:
        pd.DataFrame: A dataframe representing the summed synaptic input from
        presynaptic neuron groups to an average neuron in each postsynaptic
        neuron group. This dataframe is always returned, regardless of the
        value of display_output.

    Displays:
        If display_output is True, the function will display a styled version
        of the resulting dataframe.
    """
    if inidx_map is None:
        inidx_map = {idx: idx for idx in range(stepsn.shape[0])}
    if outidx_map is None:
        outidx_map = inidx_map

    # remove nan values in inidx and outidx
    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    if issparse(stepsn):
        # if stepsn is coo, turn into csc
        if stepsn.format == "coo":
            stepsn = stepsn.tocsc()
        matrix = stepsn[:, outidx][inidx, :].toarray()
    else:
        matrix = stepsn[inidx, :][:, outidx]

    if include_undefined_groups:
        # fill the nan values in inidx_map (e.g. 17726: nan) and outidx_map
        # with 'undefined'
        inidx_map = {k: v if pd.notna(v) else "undefined" for k, v in inidx_map.items()}
        outidx_map = {
            k: v if pd.notna(v) else "undefined" for k, v in outidx_map.items()
        }

    # Create the dataframe
    df = pd.DataFrame(
        data=matrix,
        # choose what to group by here
        # if idx is mapped to root_id, if root_id is kept as int64, the
        # root_ids seem a bit messed up
        index=[str(inidx_map[key]) for key in inidx],
        columns=[str(outidx_map[key]) for key in outidx],
    )

    # Sum across rows: presynaptic neuron is in the rows
    # summing across neurons of the same type: total amount of input from that
    # type for the postsynaptic neuron
    summed_df = df.groupby(df.index).sum()

    # Average across columns and transpose back
    # averaging across columns of the same type:
    # on average, a neuron of that type receives x% input from a presynaptic
    # type
    result_df = summed_df.T.groupby(level=0).mean().T

    if pre_in_column:
        result_df = result_df.T

    if display_threshold > 0:
        if threshold_axis == "row":
            # only display rows where any value >= display_threshold
            result_df = result_df[(result_df >= display_threshold).any(axis=1)]
        elif threshold_axis == "column":
            # only display columns where any value >= display_threshold
            result_df = result_df.loc[:, (result_df >= display_threshold).any(axis=0)]
        else:
            raise ValueError("threshold_axis must be either 'column' or 'row'.")

    if sort_within == "column":
        if sort_names is None:
            # sort result_df by the values in the first column, in descending
            # order
            result_df = result_df.sort_values(by=result_df.columns[0], ascending=False)
        elif isinstance(sort_names, str):
            sort_names = [sort_names]

        if sort_names is not None:
            if set(sort_names).issubset(result_df.columns):
                result_df = result_df.sort_values(by=sort_names, ascending=False)
            else:
                raise ValueError(
                    "sort_names must be present in the values of outidx_map."
                )
    elif sort_within == "row":
        # first sort rows by average row value in descending order
        result_df = result_df.loc[
            result_df.mean(axis=1).sort_values(ascending=False).index
        ]

        if sort_names is None:
            # sort result_df by the values in the first column, in descending
            # order
            result_df = result_df.sort_values(
                by=result_df.index[0], axis=1, ascending=False
            )
        elif isinstance(sort_names, str):
            sort_names = [sort_names]

        if sort_names is not None:
            if set(sort_names).issubset(result_df.index):
                result_df = result_df.sort_values(
                    by=sort_names, axis=1, ascending=False
                )
            else:
                raise ValueError(
                    "sort_names must be present in the values of inidx_map."
                )
    else:
        raise ValueError("sort_within must be either 'column' or 'row'.")

    if display_output:
        result_dp = result_df.style.background_gradient(
            cmap="Blues",
            vmin=result_df.min().min(),
            vmax=result_df.max().max(),
        )
        display(result_dp)
    return result_df


def contribution_by_path_lengths(
    steps,
    inidx: arrayable,
    outidx: arrayable,
    outidx_map: dict | None = None,
    inidx_map: dict | None = None,
    width: int = 800,
    height: int = 400,
):
    """Plots the connection strength from all of inidx (grouped by inidx_map)
    to an average outidx (grouped by outidx_map) over different path lengths.
    Either inidx_map or outidx_map, but not both, should be provided. If
    neither is provided, presynaptic neurons are grouped together. Direct
    connections are in path_length 1.

    Args:
        steps (list of scipy.sparse matrices): List of sparse matrices, each
            representing synaptic strengths for a specific path length.
        inidx (int, float, list, set, numpy.ndarray, or pandas.Series): Array
            of indices representing input (presynaptic) neurons.
        outidx (int, float, list, set, numpy.ndarray, or pandas.Series): Array
            of indices representing output (postsynaptic) neurons.
        outidx_map (dict): Mapping from indices to postsynaptic neuron groups.
            Only one of inidx_map and outidx_map should be specified.
        inidx_map (dict): Mapping from indices to presynaptic neuron groups.
            Only one of inidx_map and outidx_map should be specified.
        width (int, optional): The width of the plot. Defaults to 800.
        height (int, optional): The height of the plot. Defaults to 400.

    Returns:
        None: Displays an interactive line plot showing the connection strength
            from all of inidx to an average outidx over different path lengths.

    """

    # remove nan values in inidx and outidx
    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    # check if both inidx_map and outidx_map are provided
    if inidx_map is not None and outidx_map is not None:
        raise ValueError(
            "Only one of inidx_map and outidx_map should be specified. "
            "If you want to keep both, use "
            "contribution_by_path_lengths_heatmap()."
        )

    if inidx_map is None and outidx_map is None:
        outidx_map = {idx: idx for idx in outidx}
        # give message that pres are grouped together
        print(
            "Neither inidx_map nor outidx_map provided. By default "
            "presynaptic neurons are grouped together."
        )

    rows = []
    for step in steps:
        if inidx_map is not None:
            df = pd.DataFrame(
                data=step[:, outidx][inidx, :].toarray(),
                index=[inidx_map[key] for key in inidx],
            )
            # average of all columns
            # then groupby index, and take the sum
            # average post from all pre
            df = df.mean(axis=1).groupby(level=0).sum().to_frame().T
            rows.append(df)
        elif outidx_map is not None:
            df = pd.DataFrame(
                data=step[:, outidx][inidx, :].toarray(),
                columns=[outidx_map[key] for key in outidx],
            )
            # sum all rows
            # then groupby column, and take the mean
            # average post from all pre
            df = df.sum().groupby(level=0).mean().to_frame().T
            rows.append(df)

    # first have a dataframe where: each row is a path length, each column is
    # a postsynaptic cell type
    # then pivot_wide to long
    # index is the path length
    # variable is postynaptic cell type
    # value is y axis
    contri = pd.concat(rows, ignore_index=True).melt(ignore_index=False).reset_index()
    if inidx_map is not None:
        contri.columns = ["path_length", "presynaptic_type", "value"]
    else:
        contri.columns = ["path_length", "postsynaptic_type", "value"]
    contri.path_length = contri.path_length + 1

    fig = px.line(
        contri,
        x="path_length",
        y="value",
        color=contri.columns[1],
        width=width,
        height=height,
    )
    fig.update_layout(xaxis=dict(tickmode="linear", tick0=1, dtick=1))
    fig.show()


def contribution_by_path_lengths_heatmap(
    steps,
    inidx,
    outidx,
    inidx_map=None,
    outidx_map=None,
    sort_by_index=True,
    sort_names=None,
    pre_in_column=False,
    display_threshold=0,
    cmap="viridis",
    figsize=(30, 15),
):
    """Display the contribution from inidx to outidx, grouped by inidx_map and
    outidx_map, across different path lengths.

    Args:
        steps (list of scipy.sparse matrices): List of sparse matrices, each
            representing synaptic strengths for a specific path length.
        inidx (int, float, list, set, numpy.ndarray, or pandas.Series): Array
            of indices representing input (presynaptic) neurons.
        outidx (int, float, list, set, numpy.ndarray, or pandas.Series): Array
            of indices representing output (postsynaptic) neurons.
        inidx_map (dict, optional): Mapping from indices to input neuron
            groups. Defaults to None, in which case neurons are not grouped.
        outidx_map (dict, optional): Mapping from indices to output neuron
            groups. Defaults to None, in which case it is set to be the same
            as inidx_map.
        sort_by_index (bool, optional): Whether to sort the output by index.
            Defaults to True.
        sort_names (str or list, optional): the column name(s) to sort the
            result by. If none is provided, then sort by the first column.
        pre_in_column (bool, optional): Whether to have the presynaptic neuron
            groups as columns. Defaults to False (pre in rows, post: columns).
        display_threshold (float, optional): The threshold for displaying the
            output. Defaults to 0.
        cmap (str, optional): The colormap to use for the heatmap. Defaults to
            'viridis'.
        figsize (tuple, optional): The size of the figure to display. Defaults
            to (30, 15).

    Returns:
        None: Displays an interactive heatmap showing the contribution from
            inidx to outidx, grouped by inidx_map and outidx_map, across
            different path lengths.
    """

    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    if inidx_map is None:
        inidx_map = {idx: idx for idx in inidx}
    if outidx_map is None:
        outidx_map = inidx_map

    def plot_heatmap(index):
        plt.figure(figsize=figsize)
        # plt.imshow(heatmaps[index], cmap='viridis', aspect = 'auto')
        # Use seaborn's heatmap function which is a higher-level API for
        # Matplotlib's imshow
        sns.heatmap(
            heatmaps[index - 1],
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cmap=cmap,
        )

        # Rotate the tick labels for the columns to show them better
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        # Show the heatmap
        plt.show()

    heatmaps = []

    for step in tqdm(steps):
        df = result_summary(
            step,
            inidx,
            outidx,
            inidx_map,
            outidx_map,
            display_output=False,
            sort_names=sort_names,
            pre_in_column=pre_in_column,
            display_threshold=display_threshold,
        )
        if sort_by_index:
            df.sort_index(inplace=True)
        heatmaps.append(df)

    slider = widgets.IntSlider(
        value=1,
        min=1,
        max=len(heatmaps),
        step=1,
        description="Path length",
        continuous_update=True,
    )

    # Link the slider to the plotting function
    display(widgets.interactive(plot_heatmap, index=slider))


def effective_conn_from_paths(paths, group_dict=None, wide=True):
    """Calculate the effective connectivity between (groups of) neurons based
    only on the provided `paths` between neurons. This function runs on CPU,
    and doesn't expect a big connectivity matrix as input.

    Args:
        paths (pd.DataFrame): A dataframe representing the paths between
            neurons, with columns 'pre', 'post', 'weight', and 'layer'.
        group_dict (dict, optional): A dictionary mapping neuron indices
            (values in columns `pre` and `post`) to groups. Defaults to None.
        wide (bool, optional): Whether to pivot the output dataframe to a wide
            format. Defaults to True.

    Returns:
        pd.DataFrame: A dataframe representing the effective connectivity
            between groups of neurons.
    """

    # it will get confusing if we didn't use the same mapping for all layers
    # though it is true that this would increase the size of the matrix being
    # multiplied
    local_idx_dict = {
        idx: i for i, idx in enumerate(set(paths.pre).union(set(paths.post)))
    }  # give one index for each element in path; map from element to index
    # map from index to element
    local_to_global_idx = {i: idx for idx, i in local_idx_dict.items()}
    paths.loc[:, ["pre_idx"]] = paths.pre.map(local_idx_dict)
    paths.loc[:, ["post_idx"]] = paths.post.map(local_idx_dict)

    # matmul with sparse matrices
    for i, layer in enumerate(sorted(paths.layer.unique())):
        if i == 0:
            initial_el = paths[paths.layer == layer]  # edgelist
            csr = csr_matrix(
                (
                    initial_el.weight,
                    (initial_el.pre_idx.values, initial_el.post_idx.values),
                ),
                shape=(len(local_idx_dict), len(local_idx_dict)),
            )  # make sparse matrix of the shape all_elements, all_elements

        else:
            el = paths[paths.layer == layer]
            csr = csr @ csr_matrix(
                (el.weight, (el.pre_idx.values, el.post_idx.values)),
                shape=(len(local_idx_dict), len(local_idx_dict)),
            )

    coo = csr.tocoo()
    result_el = pd.DataFrame(
        {"pre_idx": coo.row, "post_idx": coo.col, "weight": coo.data}
    )
    result_el.loc[:, ["pre"]] = result_el.pre_idx.map(local_to_global_idx)
    result_el.loc[:, ["post"]] = result_el.post_idx.map(local_to_global_idx)

    if group_dict is not None:
        result_el = group_edge_by(result_el, group_dict)
        # result_el.loc[:, ['group_pre']] = result_el.pre.map(group_dict)
        # result_el.loc[:, ['group_post']] = result_el.post.map(group_dict)
        # # group by pre, sum; group by post, average
        # # e.g. if group is cell type: the input proportion from type A to an
        # average neuron in type B
        # result_el = result_el.groupby( # sum across pre
        #     ['group_pre', 'group_post', 'post']).weight.sum().reset_index()
        # result_el = result_el.groupby( # average across post
        #     ['group_pre', 'group_post']).weight.mean().reset_index()
        # result_el.columns = ['pre', 'post', 'weight']

    # pivot wider
    if wide:
        result_el = result_el.pivot(index="pre", columns="post", values="weight")
        result_el.fillna(0, inplace=True)

    return result_el


def signed_effective_conn_from_paths(paths, group_dict=None, wide=True, idx_to_nt=None):
    """Calculate the *signed* effective connectivity between (groups of)
    neurons based only on the provided `paths` between neurons. This function
    runs on CPU, and doesn't expect a big connectivity matrix as input.

    Args:
        paths (pd.DataFrame): A dataframe representing the paths between
            neurons, with columns 'pre', 'post', 'weight', 'layer', and
            optionally 'sign'.
        group_dict (dict, optional): A dictionary mapping neuron indices
            (values in columns `pre` and `post`) to groups. Defaults to None.
        wide (bool, optional): Whether to pivot the output dataframe to a wide
            format. Defaults to True.
        idx_to_nt (dict, optional): A dictionary mapping neuron indices
            (values in columns `pre` and `post`) to 1 (excitatory) / -1
            (inhibitory). Defaults to None.

    Returns:
        list: A list of two dataframes representing the effective connectivity
            between groups of neurons, one for effective excitation, the other
            inhibition.
    """

    if ("sign" not in paths.columns) & (idx_to_nt is None):
        raise ValueError(
            "Either 'sign' column must be present in paths or "
            "idx_to_nt must be provided."
        )

    # setting local indices
    # it will get confusing if we didn't use the same mapping for all layers
    # though it is true that this would increase the size of the matrix being
    # multiplied
    local_idx_dict = {
        idx: i for i, idx in enumerate(set(paths.pre).union(set(paths.post)))
    }
    local_to_global_idx = {i: idx for idx, i in local_idx_dict.items()}
    paths.loc[:, ["pre_idx"]] = paths.pre.map(local_idx_dict)
    paths.loc[:, ["post_idx"]] = paths.post.map(local_idx_dict)

    # make sure sign is in the column:
    if "sign" not in paths.columns:
        if any(~paths.pre.isin(idx_to_nt)):
            print(
                "Warning: some neurons are not in idx_to_nt. Their outputs "
                "will be ignored"
            )
        paths.loc[:, "sign"] = paths.pre.map(idx_to_nt)

    # matmul with sparse matrices
    for i, layer in enumerate(sorted(paths.layer.unique())):
        if i == 0:
            initial_el_e = paths[(paths.layer == layer) & (paths.sign == 1)]
            initial_el_i = paths[(paths.layer == layer) & (paths.sign == -1)]
            csr_e = csr_matrix(
                (
                    initial_el_e.weight,
                    (
                        initial_el_e.pre_idx.values,
                        initial_el_e.post_idx.values,
                    ),
                ),
                shape=(len(local_idx_dict), len(local_idx_dict)),
            )
            csr_i = csr_matrix(
                (
                    initial_el_i.weight,
                    (
                        initial_el_i.pre_idx.values,
                        initial_el_i.post_idx.values,
                    ),
                ),
                shape=(len(local_idx_dict), len(local_idx_dict)),
            )

        else:
            el_e = paths[(paths.layer == layer) & (paths.sign == 1)]
            el_i = paths[(paths.layer == layer) & (paths.sign == -1)]
            this_csr_e = csr_matrix(
                (el_e.weight, (el_e.pre_idx.values, el_e.post_idx.values)),
                shape=(len(local_idx_dict), len(local_idx_dict)),
            )
            this_csr_i = csr_matrix(
                (el_i.weight, (el_i.pre_idx.values, el_i.post_idx.values)),
                shape=(len(local_idx_dict), len(local_idx_dict)),
            )
            # e = ee + ii
            # make sure csr_e is not modified in place, so that we can use it
            # for csr_i
            csr_e_new = csr_e @ this_csr_e + csr_i @ this_csr_i
            # i = ie + ei
            csr_i = csr_e @ this_csr_i + csr_i @ this_csr_e
            # now modify csr_e
            csr_e = csr_e_new

    coo_e = csr_e.tocoo()
    coo_i = csr_i.tocoo()

    # make dataframe based on the connectivity matrix
    result_el_e = pd.DataFrame(
        {"pre_idx": coo_e.row, "post_idx": coo_e.col, "weight": coo_e.data}
    )
    result_el_i = pd.DataFrame(
        {"pre_idx": coo_i.row, "post_idx": coo_i.col, "weight": coo_i.data}
    )
    # change back to the global names
    result_el_e.loc[:, ["pre"]] = result_el_e.pre_idx.map(local_to_global_idx)
    result_el_e.loc[:, ["post"]] = result_el_e.post_idx.map(local_to_global_idx)
    result_el_i.loc[:, ["pre"]] = result_el_i.pre_idx.map(local_to_global_idx)
    result_el_i.loc[:, ["post"]] = result_el_i.post_idx.map(local_to_global_idx)

    if group_dict is not None:
        result_el_e = group_edge_by(result_el_e, group_dict)
        result_el_i = group_edge_by(result_el_i, group_dict)

    if wide:
        result_el_e = result_el_e.pivot(index="pre", columns="post", values="weight")
        result_el_e.fillna(0, inplace=True)
        result_el_i = result_el_i.pivot(index="pre", columns="post", values="weight")
        result_el_i.fillna(0, inplace=True)

    return result_el_e, result_el_i


def read_precomputed(
    prefix: str, file_path: str = None, first_n: int | None = None
) -> List:
    """Reads the precomputed compressed paths.

    Args:
        prefix (str): The prefix/folder name (expected to be the same) of the
            files to read.
        file_path (str, optional): The path to the files. Defaults to None. If
            None, checks if running in Google Colab, and sets the path: if
            running in Colab, sets the path to "/content/"; otherwise, sets the
            path to "".

    Returns:
        List: A list of sparse matrices representing the steps.
    """
    if file_path is None:
        # Check if running in Google Colab
        if "COLAB_GPU" in os.environ:
            # Running in Colab
            file_path = "/content/"
        else:
            # Running locally
            file_path = ""

    steps_cpu = []
    # get npz files
    files = os.listdir(f"{file_path}{prefix}")
    files = [f for f in files if f.endswith(".npz")]

    if first_n is None:
        first_n = len(files)

    for i in range(first_n):
        steps_cpu.append(sp.sparse.load_npz(f"{file_path}{prefix}/{prefix}_{i}.npz"))
    return steps_cpu
