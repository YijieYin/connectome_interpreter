import random

import torch
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse
import matplotlib.colors as mcl
import matplotlib.pyplot as plt
from tqdm import tqdm


def dynamic_representation(tensor, density_threshold=0.2):
    """Convert tensor to sparse if density is below threshold, otherwise to dense."""
    nonzero_elements = torch.nonzero(tensor).size(0)
    total_elements = tensor.numel()
    density = nonzero_elements / total_elements

    if density < density_threshold:
        return tensor.to_sparse()
    else:
        return tensor.to_dense()


def torch_sparse_where(x, threshold):
    """
    Apply a threshold to a tensor, setting values below the threshold to zero. This function allows the tensor to be sparse. torch.where() does not.

    Args:
      x (torch.Tensor): The input tensor to apply the threshold to.
      threshold (float): The threshold value.

    Returns:
      torch.Tensor: A new tensor with values below the threshold set to zero.
    """
    if x.is_sparse:
        values = x._values()
        indices = x._indices()

        # Apply the threshold to the values
        thresholded_values = torch.where(
            values >= threshold, values, torch.tensor(0.0, device=x.device, dtype=x.dtype))

        # Create a new sparse tensor with the same indices but updated values
        result = torch.sparse_coo_tensor(
            indices, thresholded_values, x.size(), device=x.device, dtype=x.dtype).coalesce()
        # It might be beneficial to remove zero values that were newly created by thresholding
        # .coalesce() combines duplicates and removes zeros if present
    else:
        result = torch.where(
            x >= threshold, x, torch.tensor(0.0, device=x.device))

    return result


def tensor_to_csc(tensor):
    """Turn torch.Tensor into scipy Compressed Sparse Column matrix.

    Args:
      tensor (torch.Tensor): A sparse tensor.

    Returns:
      scipy.sparse.csc_matrix: A Scipy sparse Compressed Sparse Column matrix.
    """
    tensor = tensor.to('cpu').coalesce()
    # Extract indices and values
    indices = tensor.indices().numpy()
    values = tensor.values().numpy()
    # Calculate the shape of the original tensor
    shape = tensor.shape

    # Create a SciPy COO matrix
    coo = coo_matrix((values, (indices[0], indices[1])), shape=shape)

    # Convert COO matrix to CSC matrix
    csc = coo.tocsc()

    return csc


def coo_tensor_to_el(coo_tensor):
    """
    Convert a PyTorch sparse COO tensor to a DataFrame representing an edge list.

    This function checks if the input tensor is sparse. If not, it converts it to a sparse COO tensor.
    It then extracts the indices and values, and creates a DataFrame with columns 'row_idx', 'col_idx', and 'value'.
    Each row in the DataFrame represents an edge in the graph, where 'row_idx' and 'col_idx' are the nodes connected by the edge,
    and 'value' is the weight of the edge.

    Args:
      coo_tensor (torch.Tensor): A PyTorch tensor, either already in sparse COO format or dense.

    Returns:
      pd.DataFrame: A DataFrame with columns 'row_idx', 'col_idx', and 'value', representing the edge list of the graph.
    """

    if not coo_tensor.is_sparse:
        coo_tensor = coo_tensor.to_sparse_coo()

    # Transpose and convert to numpy array
    indices = coo_tensor.indices().t().cpu().numpy()
    values = coo_tensor.values().cpu().numpy()

    # Split the indices to row and column
    row_idx, col_idx = indices[:, 0], indices[:, 1]

    edge_list_df = pd.DataFrame(
        {'row_idx': row_idx, 'col_idx': col_idx, 'value': values})
    return edge_list_df


def coo_to_el(coo):
    """
    Convert a SciPy sparse COO matrix to a DataFrame representing an edge list.

    Extracts the row indices, column indices, and data from a COO matrix to create a DataFrame.
    Each row in the DataFrame represents an edge in the graph, where 'row_idx' and 'col_idx' are the nodes connected by the edge,
    and 'value' is the weight of the edge.

    Args:
      coo (scipy.sparse.coo_matrix): A COO matrix from SciPy representing a sparse matrix.

    Returns:
      pd.DataFrame: A DataFrame with columns 'row_idx', 'col_idx', and 'value', representing the edge list of the graph.
    """
    row = coo.row
    col = coo.col
    data = coo.data
    all_el = pd.DataFrame({'row_idx': row, 'col_idx': col, 'value': data})

    return all_el


def modify_coo_matrix(sparse_matrix, input_idx=None, output_idx=None, value=None, updates_df=None, re_normalize=True):
    """
    Modify the values of a sparse matrix (either COO or CSR format) at specified indices.

    There are two modes of operation:
    1. Single update mode: where `input_idx`, `output_idx`, and `value` are provided as individual arguments. In this case all combinations of input_idx and output_idx are updated.
    2. Batch update mode: where `updates_df` is provided, a DataFrame with columns ['input_idx', 'output_idx', 'value'].

    Args:
      sparse_matrix (coo_matrix or csr_matrix): The sparse matrix to modify.
      input_idx (int, list, numpy.ndarray, set, optional): Row indices for updates.
      output_idx (int, list, numpy.ndarray, set, optional): Column indices for updates.
      value (numeric, list, numpy.ndarray, optional): New values for updates. If it is a number, then it is used for all updates. Else, it needs to be of length equal to the product of the lengths of input_idx and output_idx.
      updates_df (DataFrame, optional): The DataFrame containing batch updates.
      re_normalize (bool, optional): Whether to re-normalize the matrix after updating, such that all values in each column sum up to 1. Default to True.

    Note:
      In single update mode, `input_idx`, `output_idx`, and `value` must all be provided.
      In batch update mode, only `updates_df` should be provided.
      If both modes are provided, the function will prioritize the single update mode.

    Returns:
      coo_matrix or csr_matrix: The updated sparse matrix, in the same format as the input.
    """
    if not issparse(sparse_matrix):
        raise TypeError("The provided matrix is not a sparse matrix.")

    # Ensure we're working with CSR for efficiency, remember original format
    original_format = sparse_matrix.getformat()
    csr = sparse_matrix.tocsr() if original_format == 'coo' else sparse_matrix

    updated_cols = set()  # Track columns that are updated or added
    print("Updating matrix...")
    if input_idx is not None and output_idx is not None and value is not None:
        # Convert inputs to arrays and ensure compatibility
        input_idx = to_nparray(input_idx)
        output_idx = to_nparray(output_idx)
        if np.isscalar(value) or (isinstance(value, np.ndarray) and value.size == 1):
            value = np.full(len(input_idx) * len(output_idx), value)
        elif len(value) == len(input_idx) * len(output_idx):
            value = np.atleast_1d(value)
        else:
            raise ValueError(
                "The length of 'value' is incorrect. It should either be a single value or match the length of 'input_idx' * 'output_idx'.")

        for (i, j, val) in tqdm(zip(np.repeat(input_idx, len(output_idx)),
                                    np.tile(output_idx, len(input_idx)),
                                    value)):
            # Update the matrix, note the changes in indexing for csr
            if csr[i, j] != 0:
                csr[i, j] = val  # Efficient for CSR format
                updated_cols.add(j)

    elif updates_df is not None:
        # Batch updates from a DataFrame
        for _, row in tqdm(updates_df.iterrows()):
            i, j, val = row['input_idx'], row['output_idx'], row['value']
            if csr[i, j] != 0:
                csr[i, j] = val  # Efficient for CSR format
                updated_cols.add(j)

    if re_normalize and updated_cols:
        print("Re-normalizing updated columns...")
        # Normalize updated columns
        for col in updated_cols:
            col_sum = csr[:, col].sum()
            if col_sum != 0:
                csr[:, col] /= col_sum
    # remove 0 entries
    csr.eliminate_zeros()
    return csr.asformat(original_format)


def to_nparray(input_data):
    """
    Converts the input data into a numpy array, filtering out any NaN values and duplicates. 
    The input can be a single number, a list, a set, a numpy array, or a pandas Series.

    Args:
        input_data: The input data to convert. Can be of type int, float, list, set, numpy.ndarray, or pandas.Series.

    Returns:
        numpy.ndarray: A unique numpy array created from the input data, with all NaN values removed.
    """
    # First, ensure the input is in array form and convert pd.Series to np.ndarray
    if isinstance(input_data, (int, float)):
        input_array = np.array([input_data])
    elif isinstance(input_data, (list, set, np.ndarray, pd.Series)):
        input_array = np.array(list(input_data))
    else:
        raise TypeError(
            "Input data must be an int, float, list, set, numpy.ndarray, or pandas.Series")

    # Then, remove NaN values
    cleaned_array = input_array[~pd.isna(input_array)]

    # Finally, return unique values
    return np.unique(cleaned_array)


def get_ngl_link(df, no_connection_invisible=True, colour_saturation=0.4, scene=None, normalise_within_column=True, include_postsynaptic_neuron=False):
    """
    Generates a Neuroglancer link with layers, corresponding to each column in the df. The function
    processes a dataframe, adding colour information to each neuron,
    and then creates layers in a Neuroglancer scene for visualization. Separate layers
    are created for each column in the DataFrame, with unique segment colors based on
    the values in the DataFrame.

    Args:
        df (pandas.DataFrame): A DataFrame containing neuron metadata. The index should contain neuron identifiers (root_ids), and columns should represent different
      attributes or categories.
        no_connection_invisible (bool, optional): Whether to make invisible neurons that are not connected. Default to True (invisible). 
        colour_saturation (float, optional): The saturation of the colours. Default to 0.4.
        scene (ngl.Scene, optional): A Neuroglancer scene object from nglscenes package. You can read a scene from clipboard like `scene = Scene.from_clipboard()`. 
        normalise_within_column (bool, optional): Whether to normalise the values within each column (the alternative is normalising by the min and max value in the entire dataframe). Default to True.
        include_postsynaptic_neuron (bool, optional): Whether to include the postsynaptic neuron (column names of `df`) in the visualisation. Default to False.

    Returns:
        str: A URL to the generated Neuroglancer scene.

    Note:
        The function assumes that the 'scene1' variable is defined in the global scope and is an instance of ngl.Scene.
        The function creates separate Neuroglancer layers for each column in the DataFrame, using the column name as the layer name.
        The root_ids are colored based on the values in the DataFrame, with a color scale ranging from white (minimum value) to a specified color (maximum value).
        The function relies on the Neuroglancer library for layer creation and scene manipulation.
    """

    try:
        import nglscenes as ngl
    except ImportError:
        raise ImportError(
            "To use this function, please install the package by running 'pip3 install git+https://github.com/schlegelp/nglscenes@main]'")

    # define a scene if not given:
    if scene == None:
        # Initialize a scene
        scene = ngl.Scene()
        scene['layout'] = '3d'
        scene['position'] = [527216.1875, 208847.125, 84774.0625]
        scene['projectionScale'] = 400000
        scene['dimensions'] = {"x": [1e-9, "m"],
                               "y": [1e-9, "m"], "z": [1e-9, "m"]}

        # and another for FAFB mesh
        fafb_layer = ngl.SegmentationLayer(source='precomputed://https://spine.itanna.io/files/eric/jfrc_mesh_test',
                                           name='jfrc_mesh_test1')
        fafb_layer['segments'] = ['1']
        fafb_layer['objectAlpha'] = 0.17
        fafb_layer['selectedAlpha'] = 0.55
        fafb_layer['segmentColors'] = {'1': '#cacdd8'}
        fafb_layer['colorSeed'] = 778769298
        scene.add_layers(fafb_layer)

        # and the neuropil layer with names
        np_layer = ngl.SegmentationLayer(source='precomputed://gs://neuroglancer-fafb-data/elmr-data/FAFBNP.surf/mesh#type=mesh',
                                         name='neuropil_regions_surface_named')
        np_layer['segments'] = [str(num) for num in range(0, 79)]
        np_layer['visible'] = False
        np_layer['objectAlpha'] = 0.17
        scene.add_layers(np_layer)

    # Define a list of colors optimized for human perception on a dark background
    colors = ['#ff6b6b', '#f06595', '#cc5de8', '#845ef7', '#5c7cfa', '#339af0', '#22b8cf', '#20c997', '#51cf66', '#94d82d', '#fcc419', '#4ecdc4',
              '#ffe66d', '#7bed9f', '#a9def9', '#f694c1', '#c7f0bd', '#ffc5a1', '#ff8c94', '#ffaaa6', '#ffd3b5', '#a8e6cf', '#a6d0e4', '#c1beff', '#f5b0cb']

    # Normalize the values in the DataFrame
    if normalise_within_column:
        df_norm = (df - df.min()) / (df.max() - df.min())
    else:
        df_norm = (df - df.min().min()) / (df.max().max() - df.min().min())

    scene['layout'] = '3d'

    source = 'precomputed://gs://flywire_v141_m783'

    for column in df.columns:
        color = random.choice(colors)
        cmap = mcl.LinearSegmentedColormap.from_list(
            "custom_cmap", ["white", color])

        df_group = df_norm[[column]]
        if no_connection_invisible:
            df_group = df_group[df_group.iloc[:, 0] > 0]

        layer = ngl.SegmentationLayer(source=source, name=str(column))

        layer['segments'] = list(df_group.index.astype(str))
        layer['segmentColors'] = {
            # trick from Claud to make the colours more saturated
            str(root_id): mcl.to_hex(cmap(colour_saturation + (1-colour_saturation)*value.values[0]))
            for root_id, value in df_group.iterrows()
        }
        if include_postsynaptic_neuron:
            layer['segments'].append(str(df_group.columns[0]))
            layer['segmentColors'][str(df_group.columns[0])] = '#43A7FF'

        scene.add_layers(layer)

    return scene.url


def top_n_activated(array, idx_map, model, sensory, n=None, threshold=None):
    """
    Identifies the top activated neurons for each column in the array, 
    either by number (top n) or by a minimum activation threshold, or both.

    Args:
        array (np.ndarray): 2D array of neuron activations, where rows represent neurons 
            and columns represent different time steps.
        idx_map (dict): Mapping from neuron index to neuron identifier.
        model: Model object containing `sensory_indices` and `non_sensory_indices` attributes.
        sensory (bool): If True, considers only sensory neurons; otherwise, considers non-sensory neurons.
        n (int, optional): Number of top activations to return for each column. If None, 
            all activations above the threshold are returned. Defaults to None.
        threshold (float, optional): Minimum activation level to consider. If None, 
            the top n activations are returned regardless of their magnitude. Defaults to None.

    Returns:
        dict: A dictionary where each key is a column index and each value is a nested dictionary 
            of neuron identifiers and their activations, for those activations that are either 
            in the top n, above the threshold, or both.

    Note:
        If both `n` and `threshold` are provided, the function returns up to top n activations 
        that are also above the threshold for each column.
    """
    result = {}
    sensory_indices = model.sensory_indices.cpu().numpy()
    non_sensory_indices = model.non_sensory_indices.cpu().numpy()

    indices = sensory_indices if sensory else non_sensory_indices
    global_to_local_map = {global_idx: num for num,
                           global_idx in enumerate(indices)}

    for col in range(array.shape[1]):
        # Determine which indices to use based on the 'sensory' flag
        # these are global indices in the all-to-all connectivity

        column_values = array[:, col]

        # Filter activations by threshold if provided
        if threshold is not None:
            # these are local indices
            filtered_indices = np.where(column_values >= threshold)[0]
            # these are global indices
            threshold_indices = indices[filtered_indices]
        else:
            threshold_indices = indices

        # Sort the filtered activations
        # these are the local indices corresponding to only sensory/non-sensory neurons, but not both
        sorted_indices = np.argsort(
            column_values)[-n:] if n is not None else np.argsort(column_values)
        # these are global indices
        topn_indices = indices[sorted_indices]

        selected = np.intersect1d(threshold_indices, topn_indices)

        # Build the result dictionary
        result[col] = {idx_map[idx]: column_values[global_to_local_map[idx]]
                       for idx in selected}

    return result


def plot_column_changes(input_snapshots, column_index=0):
    """
    Plot the changes in a specific column of input_tensor across training iterations.

    Args:
        input_snapshots: List of 2D tensor snapshots collected during training.
        column_index: Index of the column to plot.
    """
    # Assuming input_snapshots is a list of 2D arrays, extract the specific column across snapshots
    column_values = np.array([snapshot[:, column_index]
                             for snapshot in input_snapshots])

    # Plotting
    plt.figure(figsize=(10, 6))
    # Use 'imshow' for a 2D heatmap-like visualization
    plt.imshow(column_values.T, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Value')
    plt.xlabel('Snapshot Index')
    plt.ylabel(f'Elements in Column {column_index}')
    plt.title(f'Changes in Column {column_index} During Training')
    plt.show()
