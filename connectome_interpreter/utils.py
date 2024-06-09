import random
import warnings

import torch
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse
import matplotlib.colors as mcl
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx


def dynamic_representation(tensor, density_threshold=0.2):
    """Convert tensor to sparse if density is below threshold, otherwise to dense. This might be memory intensive."""
    # nonzero_elements = torch.nonzero(tensor).size(0)
    # .nonzero() doesn't work if the tensor has more than INT_MAX elements
    # Calculate the number of non-zero elements without using torch.nonzero()
    nonzero_elements = tensor.count_nonzero().item()

    # to_sparse() doesn't support operations on tensors with more than INT_MAX (2,147,483,647) elements
    if nonzero_elements < 2147483647:
        total_elements = tensor.numel()
        density = nonzero_elements / total_elements

        if density < density_threshold:
            return tensor.to_sparse()
        else:
            return tensor.to_dense()
    else:
        print("Tensor is too large to convert to sparse. Returning dense tensor.")
        return tensor


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
      tensor (torch.Tensor): A (sparse) tensor.

    Returns:
      scipy.sparse.csc_matrix: A Scipy sparse Compressed Sparse Column matrix.
    """
    if tensor.is_sparse:
        tensor = tensor.coalesce()
        indices = tensor.indices().numpy()
        values = tensor.values().numpy()
        shape = tensor.shape
        coo = coo_matrix((values, (indices[0], indices[1])), shape=shape)
    else:
        coo = coo_matrix(tensor.numpy())
        coo.eliminate_zeros()
        # .nonzero() is too memory consuming for large tensors
        # memory usage can be further reduced by first spliting the tensor/COO matrix into smaller chunks, for future work

    return coo.tocsc()


def coo_tensor_to_el(coo_tensor):
    """
    Convert a PyTorch sparse COO tensor to a DataFrame representing an edge list.

    This function checks if the input tensor is sparse. If not, it converts it to a sparse COO tensor.
    It then extracts the indices and values, and creates a DataFrame with columns 'pre', 'post', and 'weight'.
    Each row in the DataFrame represents an edge in the graph, where 'pre' and 'post' are the nodes connected by the edge,
    and 'weight' is the weight of the edge.

    Args:
      coo_tensor (torch.Tensor): A PyTorch tensor, either already in sparse COO format or dense.

    Returns:
      pd.DataFrame: A DataFrame with columns 'pre', 'post', and 'weight', representing the edge list of the graph.
    """

    if not coo_tensor.is_sparse:
        coo_tensor = coo_tensor.to_sparse_coo()

    # Transpose and convert to numpy array
    indices = coo_tensor.indices().t().cpu().numpy()
    values = coo_tensor.values().cpu().numpy()

    # Split the indices to row and column
    pre, post = indices[:, 0], indices[:, 1]

    edge_list_df = pd.DataFrame(
        {'pre': pre, 'post': post, 'weight': values})
    return edge_list_df


def coo_to_el(coo, row_indices=None, col_indices=None):
    """
    Extracts an edgelist from a COO matrix, optionally using pre-specified row and/or column indices.
    If row_indices or col_indices are None, all rows or columns are considered, respectively.
    Each row in the resulting DataFrame represents an edge in the graph, where 'pre' and 'post' are the nodes connected by the edge,
    and 'value' is the weight of the edge.

    Args:
      coo: A scipy.sparse.coo_matrix instance.
      row_indices: Optional; A list or array of row indices of interest.
      col_indices: Optional; A list or array of column indices of interest.

    Returns:
      pd.DataFrame: A DataFrame with columns 'pre', 'post', and 'value', representing the edge list of the graph.
    """
    # Determine whether to filter on rows/columns based on provided indices
    row_mask = np.full(coo.shape[0], True) if row_indices is None else np.isin(
        coo.row, row_indices)
    col_mask = np.full(coo.shape[1], True) if col_indices is None else np.isin(
        coo.col, col_indices)

    # Combine row and column masks
    mask = row_mask & col_mask

    # Filter rows, cols, and data based on the combined mask
    rows = coo.row[mask]
    cols = coo.col[mask]
    data = coo.data[mask]

    all_el = pd.DataFrame({'pre': rows, 'post': cols, 'value': data})

    return all_el


def adjacency_df_to_el(adjacency, threshold=None):
    """
    Convert a DataFrame representing an adjacency matrix to an edge list.

    This function takes a DataFrame where the index and columns represent nodes in the graph,
    and the values represent the weights of the edges between them. It converts this DataFrame to an edge list,
    where each row represents an edge in the graph, with columns 'pre', 'post', and 'weight'.

    Args:
      adjacency (pd.DataFrame): A DataFrame representing an adjacency matrix.
      threshold (float, optional): If provided, edges with values below this threshold are removed. Default to None.

    Returns:
      pd.DataFrame: A DataFrame with columns 'pre', 'post', and 'weight', representing the edge list of the graph.
    """
    if threshold is not None:
        adjacency = adjacency.where(adjacency >= threshold, 0)

    # Stack the DataFrame to create a long format
    adjacency = adjacency.stack().reset_index()
    adjacency.columns = ['pre', 'post', 'weight']
    adjacency = adjacency[adjacency['weight'] != 0]

    return adjacency


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

    # Convert sparse matrix to COO format for direct edge manipulation
    original_format = sparse_matrix.getformat()
    original_dtype = sparse_matrix.dtype
    if original_format != 'coo':
        sparse_matrix = sparse_matrix.tocoo()

    # Extract edges as a DataFrame
    edges = pd.DataFrame(
        {'input_idx': sparse_matrix.row, 'output_idx': sparse_matrix.col, 'value': sparse_matrix.data})

    print("Updating matrix...")
    if input_idx is not None and output_idx is not None and value is not None:
        # Create updates DataFrame from provided indices and values
        input_idx = to_nparray(input_idx)
        output_idx = to_nparray(output_idx)
        if np.isscalar(value):
            value = np.full(len(input_idx) * len(output_idx), value)
        elif len(value) != len(input_idx) * len(output_idx):
            raise ValueError(
                "Length of 'value' must match the product of the lengths of 'input_idx' and 'output_idx', or be one single value.")

        updates_df = pd.DataFrame({
            'input_idx': np.repeat(input_idx, len(output_idx)),
            'output_idx': np.tile(output_idx, len(input_idx)),
            'value': value
        })

    elif updates_df is not None:
        # check column names
        if not all(col in updates_df.columns for col in ['input_idx', 'output_idx', 'value']):
            raise ValueError(
                "The DataFrame must contain columns ['input_idx', 'output_idx', 'value'].")

    else:
        raise ValueError(
            "Either ['input_idx', 'output_idx', and 'value'], or 'updates_df' must be provided.")

    # Concatenate edges and updates_df, ensuring that updates_df is last so its values take precedence
    combined_edges = pd.concat([edges, updates_df])

    # Remove duplicates, keeping the last occurrence (from updates_df)
    # We specify the subset to be the columns 'input_idx' and 'output_idx'
    edges = combined_edges.drop_duplicates(
        subset=['input_idx', 'output_idx'], keep='last')

    if re_normalize:
        print("Re-normalizing updated columns...")
        edge_to_normalise = edges[edges.output_idx.isin(updates_df.output_idx)]
        col_sums = edge_to_normalise.groupby('output_idx').value.sum()
        col_sums[col_sums == 0] = 1

        edge_to_normalise.loc[:, ['value']] = edge_to_normalise['value'] / \
            edge_to_normalise['output_idx'].map(col_sums)

        edges = pd.concat([edges[~edges.output_idx.isin(updates_df.output_idx)],
                           edge_to_normalise])

    # Convert back to original sparse matrix format
    updated_matrix = coo_matrix(
        (edges['value'].astype(original_dtype), (edges['input_idx'], edges['output_idx'])), shape=sparse_matrix.shape)

    # if re_normalize:
    #     print("Re-normalizing updated columns...")
    #     # Convert updated_matrix back to CSR format for efficient column operations
    #     updated_matrix = updated_matrix.tocsr()

    #     # Identify the unique columns to re-normalize based on the updates_df
    #     unique_cols = updates_df.reset_index()['output_idx'].unique()

    #     # Compute sum of each column that has been updated
    #     col_sums = np.array(
    #         updated_matrix[:, unique_cols].sum(axis=0)).flatten()

    #     for idx, col in enumerate(unique_cols):
    #         if col_sums[idx] != 0:
    #             updated_matrix[:, col] /= col_sums[idx]

    if issubclass(updated_matrix.dtype.type, np.integer):
        warnings.warn(
            "Matrix data is of integer type instead of float32. Is the sparse matrix normalised connectivity?",
            UserWarning
        )
    updated_matrix.eliminate_zeros()
    return updated_matrix.asformat(original_format)

    # if not issparse(sparse_matrix):
    #     raise TypeError("The provided matrix is not a sparse matrix.")

    # # Ensure we're working with CSR for efficiency, remember original format
    # original_format = sparse_matrix.getformat()
    # csr = sparse_matrix.tocsr() if original_format == 'coo' else sparse_matrix

    # updated_cols = set()  # Track columns that are updated or added
    # print("Updating matrix...")
    # if input_idx is not None and output_idx is not None and value is not None:
    #     # Convert inputs to arrays and ensure compatibility
    #     input_idx = to_nparray(input_idx)
    #     output_idx = to_nparray(output_idx)
    #     if np.isscalar(value) or (isinstance(value, np.ndarray) and value.size == 1):
    #         value = np.full(len(input_idx) * len(output_idx), value)
    #     elif len(value) == len(input_idx) * len(output_idx):
    #         value = np.atleast_1d(value)
    #     else:
    #         raise ValueError(
    #             "The length of 'value' is incorrect. It should either be a single value or match the length of 'input_idx' * 'output_idx'.")

    #     for (i, j, val) in tqdm(zip(np.repeat(input_idx, len(output_idx)),
    #                                 np.tile(output_idx, len(input_idx)),
    #                                 value)):
    #         # Update the matrix, note the changes in indexing for csr
    #         if csr[i, j] != 0:
    #             csr[i, j] = val  # Efficient for CSR format
    #             updated_cols.add(j)

    # elif updates_df is not None:
    #     # Batch updates from a DataFrame
    #     for _, row in tqdm(updates_df.iterrows()):
    #         i, j, val = row['input_idx'], row['output_idx'], row['value']
    #         if csr[i, j] != 0:
    #             csr[i, j] = val  # Efficient for CSR format
    #             updated_cols.add(j)

    # else:
    #     raise ValueError("Either ['input_idx', 'output_idx', and 'value'], or 'updates_df' must be provided.")

    # if re_normalize and updated_cols:
    #     print("Re-normalizing updated columns...")
    #     # Normalize updated columns
    #     for col in updated_cols:
    #         col_sum = csr[:, col].sum()
    #         if col_sum != 0:
    #             csr[:, col] /= col_sum
    # # remove 0 entries
    # csr.eliminate_zeros()
    # return csr.asformat(original_format)


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


def get_ngl_link(df, no_connection_invisible=True, colour_saturation=0.4, scene=None, normalise_within_column=True, include_postsynaptic_neuron=False, diff_colours_per_layer=False, colors=None, colormap='viridis'):
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
        diff_colours_per_layer (bool, optional): Whether to use different colours for each layer. Default to False.
        colors (list, optional): A list of colours to use for the neurons in each layer, if `diff_colours_per_layer` is True. If None, a default list of colours is used. Default to None.
        colormap (str, optional): The name of the colormap to use for colouring the neurons in every layer, if `diff_colours_per_layer` is False. Default to 'viridis'.

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
            "To use this function, please install the package by running 'pip3 install git+https://github.com/schlegelp/nglscenes@main'")

    # define a scene if not given:
    if scene == None:
        no_scene_provided = True
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

        # and the neuropil layer with names
        np_layer = ngl.SegmentationLayer(source='precomputed://gs://neuroglancer-fafb-data/elmr-data/FAFBNP.surf/mesh#type=mesh',
                                         name='neuropil_regions_surface_named')
        np_layer['segments'] = [str(num) for num in range(0, 79)]
        np_layer['visible'] = False
        np_layer['objectAlpha'] = 0.17

    # Define a list of colors optimized for human perception on a dark background
    if colors is None:
        colors = ['#ff6b6b', '#f06595', '#cc5de8', '#845ef7', '#5c7cfa', '#339af0', '#22b8cf', '#20c997', '#51cf66', '#94d82d', '#fcc419', '#4ecdc4',
                  '#ffe66d', '#7bed9f', '#a9def9', '#f694c1', '#c7f0bd', '#ffc5a1', '#ff8c94', '#ffaaa6', '#ffd3b5', '#a8e6cf', '#a6d0e4', '#c1beff', '#f5b0cb']

    # Normalize the values in the DataFrame
    if normalise_within_column:
        df_norm = (df - df.min()) / (df.max() - df.min())
    else:
        df_norm = (df - df.min().min()) / (df.max().max() - df.min().min())

    scene['layout'] = '3d'

    source = ['precomputed://gs://flywire_v141_m783',
              'precomputed://https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/dynann/flytable-info-783-all']

    for i, column in enumerate(df.columns):
        if diff_colours_per_layer:
            color = random.choice(colors)
            cmap = mcl.LinearSegmentedColormap.from_list(
                "custom_cmap", ["white", color])
        else:
            cmap = plt.get_cmap(colormap)

        df_group = df_norm[[column]]
        if no_connection_invisible:
            df_group = df_group[df_group.iloc[:, 0] > 0]

        layer = ngl.SegmentationLayer(source=source, name=str(column))

        layer['segments'] = list(df_group.index.astype(str))
        if diff_colours_per_layer:
            layer['segmentColors'] = {
                # trick from Claud to make the colours more saturated
                str(root_id): mcl.to_hex(cmap(colour_saturation + (1-colour_saturation)*value.values[0]))
                for root_id, value in df_group.iterrows()
            }
        else:
            layer['segmentColors'] = {
                str(root_id): mcl.to_hex(cmap(value.values[0]))
                for root_id, value in df_group.iterrows()
            }
        # only the first layer is visible
        if i == 0:
            layer['visible'] = True
        else:
            layer['visible'] = False

        if include_postsynaptic_neuron:
            layer['segments'].append(str(df_group.columns[0]))
            layer['segmentColors'][str(df_group.columns[0])] = '#43A7FF'

        scene.add_layers(layer)

    if no_scene_provided:
        # add FAFB and NP layers at the end
        scene.add_layers(fafb_layer)
        scene.add_layers(np_layer)

    return scene.url


def get_activations(array, global_indices, idx_map=None, top_n=None, threshold=None):
    """
    Get activation for neurons (in rows) in `array` for each time step (in columns). Optionally group neurons by `idx_map`, and filter by `top_n` or `threshold`.

    Args:
        array (np.ndarray): 2D array of neuron activations, where rows represent neurons
            and columns represent different time steps.
        global_indices (int, list, set, np.ndarray, pd.Series): Array of global neuron indices corresponding to keys in `idx_map`. 
        idx_map (dict, optional): Mapping from neuron index (`global_indices`) to neuron identifier. If not None, and if multiple neurons map to the same identifier, the activations are averaged. Defaults to None.
        top_n (int, optional): Number of top activations to return for each column. If None,
            all activations above the threshold are returned. Defaults to None.
        threshold (float, dict, optional): Minimum activation level to consider. If a dictionary is provided, the threshold for each column is specified by the column index. Defaults to None.

    Returns:
        dict: A dictionary where each key is a column index and each value is a nested dictionary
            of neuron identifiers and their activations, for those activations that are either
            in the top n, above the threshold, or both.

    Note:
        The global_indices have to be in the same order as the indices in defining the original model. 
        If both `n` and `threshold` are provided, the function returns up to top n activations
        that are also above the threshold for each column.
    """
    result = {}
    indices = to_nparray(global_indices)
    if array.shape[0] != len(indices):
        raise ValueError(
            "The length of 'global_indices' should match the number of rows in 'array'.")

    global_to_local_map = {global_idx: num for num,
                           global_idx in enumerate(indices)}

    for col in range(array.shape[1]):
        # Determine which indices to use based on the 'sensory' flag
        # these are global indices in the all-to-all connectivity

        column_values = array[:, col]
        local_indices = np.asarray(range(len(column_values)))

        # Filter activations by threshold if provided
        if threshold is not None:
            if isinstance(threshold, dict):
                if col not in threshold:
                    thresh = 0
                else:
                    thresh = threshold[col]
            else:
                thresh = threshold

            if thresh > 0:
                # these are local indices
                local_indices = np.setdiff1d(
                    local_indices, np.where(column_values < thresh)[0])

        # Sort the filtered activations
        # these are the local indices
        sorted_indices = np.argsort(
            column_values)[-top_n:] if top_n is not None else np.argsort(column_values)

        # get intersection of sorted_indices and local_indices, in the same order as sorted_indices
        selected_local = [
            index for index in sorted_indices if index in local_indices]
        selected_global = indices[selected_local]

        # Build the result dictionary
        if idx_map is None:
            result[col] = {idx: column_values[global_to_local_map[idx]]
                           for idx in selected_global}
        else:
            # initialise a dict of empty lists
            result[col] = {idx_map[idx]: [] for idx in selected_global}

            for idx in selected_global:
                result[col][idx_map[idx]].append(
                    column_values[global_to_local_map[idx]])
            # grouped indices by idx_map
            new_indices = result[col].keys()
            # calculate the average
            result[col] = {idx: np.mean(result[col][idx])
                           for idx in new_indices}

    return result


def plot_layered_paths(path_df, figsize=(10, 8), priority_indices=None, sort_by_activation=False,
                       fraction=0.03, pad=0.02, weight_decimals=2,
                       neuron_to_sign=None, sign_color_map={1: 'red', -1: 'blue'}):
    """
    Plots a directed graph of layered paths with optional node coloring based on activation values.

    This function creates a visualization of a directed graph with nodes placed in layers. Nodes can be optionally
    colored based on 'pre_activation' and 'post_activation' columns present in the dataframe. If these columns are
    missing, a default color is used for all nodes. The edges are weighted, and their labels represent the weight values.

    Args:
        path_df (pandas.DataFrame): A dataframe containing the columns 'pre', 'post', 'layer', 'weight', and optionally 
                                    'pre_activation', 'post_activation', 'pre_layer', 'post_layer'. Each row represents an
                                    edge in the graph. The 'pre' and 'post' columns refer to the source and target nodes, respectively.
                                    The 'layer' column is used to place nodes in layers, and 'weight' indicates the edge weight.
                                    If present, 'pre_activation' and 'post_activation' are used to color the nodes based on
                                    their activation values.
        figsize (tuple, optional): A tuple indicating the size of the matplotlib figure. Defaults to (10, 8).
        priority_indices (list, optional): A list of indices to prioritize when creating the layered positions. Nodes with these
                                    indices will be placed at the top of their respective layers. Defaults to None. 
        sort_by_activation (bool, optional): A flag to sort the nodes based on their activation values (after grouping by priority). Defaults to False.
        fraction (float, optional): The fraction of the figure width to use for the colorbar. Defaults to 0.03.
        pad (float, optional): The padding between the colorbar and the plot. Defaults to 0.02.
        weight_decimals (int, optional): The number of decimal places to display for edge weights. Defaults to 2.
        neuron_to_sign (dict, optional): A dictionary mapping neuron names (as they appear in path_df) to their signs (e.g. {'KCg-m': 1, 'Delta7': -1}). Defaults to None.
        sign_color_map (dict, optional): A dictionary mapping neuron signs to colors. Defaults to red for excitatory, and blue for inhibitory. Edges are in lightgrey by default.

    Returns:
        None: This function does not return a value. It generates a plot using matplotlib.

    Note:
        If 'pre_layer' and 'post_layer' columns are not in the dataframe, they will be created within the function
        to uniquely identify the nodes based on their 'pre'/'post' values and 'layer'.
        The function automatically checks for the presence of 'pre_activation' and 'post_activation' columns to
        determine whether to color the nodes based on activation values.
        The positions of the nodes are determined by a custom positioning function (`connectome_interpreter.path_finding.create_layered_positions`).
        This function requires the networkx library for graph operations and matplotlib for plotting.
    """
    if path_df.shape[0] == 0:
        raise ValueError("The provided DataFrame is empty.")

    # Create a 'post_layer' column to use as unique identifiers
    if 'post_layer' not in path_df.columns:
        path_df['post_layer'] = path_df['post'].astype(
            str) + '_' + path_df['layer'].astype(str)
    if 'pre_layer' not in path_df.columns:
        path_df['pre_layer'] = path_df['pre'].astype(
            str) + '_' + (path_df.layer-1).astype(str)

    # Create the graph using the new 'post_layer' identifiers
    G = nx.from_pandas_edgelist(path_df, 'pre_layer', 'post_layer', [
                                'weight'], create_using=nx.DiGraph())

    # Labels for nodes
    labels = dict(zip(path_df.post_layer, path_df.post))
    labels.update(dict(zip(path_df.pre_layer, path_df.pre)))

    # Determine the width of the edges
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    widths = [max(0.1, w * 5) for w in weights]  # Scale weights for visibility

    # Generate positions
    from .path_finding import create_layered_positions
    if sort_by_activation:
        node_activation_dict = dict(
            zip(path_df.post_layer, path_df.post_activation))
        node_activation_dict.update(
            dict(zip(path_df.pre_layer, path_df.pre_activation)))
        positions = create_layered_positions(
            path_df, priority_indices, sort_dict=node_activation_dict)
    else:
        positions = create_layered_positions(path_df, priority_indices)

    # Node colors based on activation values
    if ('pre_activation' in path_df.columns) & ('post_activation' in path_df.columns):
        activations = np.concatenate(
            [path_df['pre_activation'].values, path_df['post_activation'].values])
        norm = plt.Normalize(vmin=activations.min(), vmax=activations.max())
        color_map = plt.get_cmap('viridis')
        # Update graph with activation data
        nx.set_node_attributes(
            G, dict(zip(path_df.pre_layer, path_df.pre_activation)), 'activation')
        nx.set_node_attributes(
            G, dict(zip(path_df.post_layer, path_df.post_activation)), 'activation')

        node_colors = [color_map(norm(G.nodes[node]['activation']))
                       for node in G.nodes()]

    # Edge colors based on neuron signs
    default_color = 'lightgrey'  # Default edge color

    # specify edge colour based on pre-neuron sign, if available
    edge_colors = []
    for u, _ in G.edges():
        pre_neuron = labels[u]
        if neuron_to_sign and pre_neuron in neuron_to_sign:
            edge_colors.append(sign_color_map.get(
                neuron_to_sign[pre_neuron], default_color))
        else:
            edge_colors.append(default_color)

    # Plot the graph
    _, ax = plt.subplots(figsize=figsize)
    if ('pre_activation' in path_df.columns) & ('post_activation' in path_df.columns):
        nx.draw(G, pos=positions,
                labels=labels,
                with_labels=True, node_size=100,
                node_color=node_colors,
                arrows=True, arrowstyle='-|>', arrowsize=10,
                font_size=8, width=widths, edge_color=edge_colors, ax=ax)
        plt.colorbar(plt.cm.ScalarMappable(
            norm=norm, cmap=color_map), ax=ax, label='Activation', fraction=fraction, pad=pad)
    else:
        nx.draw(G, pos=positions,
                labels=labels,
                with_labels=True, node_size=100,
                node_color='lightblue', arrows=True, arrowstyle='-|>', arrowsize=10, font_size=8, width=widths, ax=ax, edge_color=edge_colors)

    edge_labels = {(u, v): f'{data["weight"]:.{weight_decimals}f}' for u,
                   v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels,
                                 #  font_color='red'
                                 ax=ax)
    ax.set_ylim(0, 1)
    plt.show()


# def change_model_weights(model, df, mode, coefficient=0.1):
#     """
#     Change the weights of the model based on the provided DataFrame.
#     The DataFrame should contain columns 'pre' and 'post', which contain indices of the connectivity weights in model. The weights are modified proportional to:
#     1. The similarity of pre and post activations (i.e. the sum of element-wise multiplication of activations across time), and
#     2. The coefficient provided.
#     The 'mode' column should specify whether the weight change is ltp or ltd.

#     Args:
#         model (torch.nn.Module): The model containing the weights to change.
#         df (pd.DataFrame): A DataFrame containing the columns 'pre' and 'post'.
#         mode (str): The mode of weight change, either 'ltp' or 'ltd'.

#     Returns:
#         None: This function modifies the model weights in place.
#     """

#     df.loc[:, ['pre_local_idx']] = pd.factorize(df.pre)[0]
#     df.loc[:, ['post_local_idx']] = pd.factorize(df.post)[0]

#     # piggy back on matrix multiplication, and calculate between each pair of neurons:
#     # sum of unitary product of activity across timesteps
#     # then multiply this sum with the coefficient
#     weight_changes = torch.matmul(model.activations[df.pre.unique(), :],
#                                   model.activations.t()[:, df.post.unique()]) * coefficient
#     # only select the pairs present in the original ltd/ltp_df
#     mask = torch.zeros_like(
#         weight_changes, dtype=torch.bool, device=weight_changes.device)
#     mask[df.pre_local_idx.values[:, None], df.post_local_idx.values] = True
#     # the rest have no change
#     weight_changes[~mask] = 0

#     # take out the weights to change
#     # NOTE: all_weights have pre in the columns, post in the rows
#     weights_subset = model.all_weights[df.post.unique()[
#         :, None], df.pre.unique()]
#     # change the weights: keep the signs separate, strengthen (ltp) / weaken (ltd) the connection by using the absolute values
#     if mode == 'ltp':
#         # strengthen connectivity
#         weights_subset_abs = torch.abs(weights_subset) + weight_changes.t()
#     elif mode == 'ltd':
#         # weaken connectivity
#         weights_subset_abs = torch.abs(weights_subset) - weight_changes.t()
#     weights_subset = torch.sign(weights_subset) * \
#         torch.clamp(weights_subset_abs, 0)
#     # finally modify in the big weights matrix.
#     model.all_weights[df.post.unique()[:, None], df.pre.unique()
#                       ] = weights_subset


def change_model_weights(model, df, mode, coefficient=0.1, offset=0, normalise=True):
    """
    Change the weights of the model based on the provided DataFrame.
    The DataFrame should contain columns 'pre' and 'post', and optionally 'conditional', which contain indices of the connectivity weights in model. The weights are modified proportional to: 
    1. The similarity of pre and post activations (i.e. the sum of element-wise multiplication of activations across time), and 
    2. The coefficient provided.
    3. The similarity of conditional activations (if provided). `offset` specifies the time lag between the conditional neurons and the pre and post neurons.
    The 'mode' column should specify whether the weight change is ltp or ltd.

    Args:
        model (torch.nn.Module): The model containing the weights to change.
        df (pd.DataFrame): A DataFrame containing the columns 'pre' and 'post'.
        mode (str): The mode of weight change, either 'ltp' or 'ltd'.
        coefficient (float): The coefficient to multiply the weight change by. Can be thought of as the 'strength of plasticity'. Default to 0.1.
        offset (int): Offset between the conditional neurons and the pre and post neurons. Default to 0. -1 means the conditional neurons' activity precede the pre and post neurons, and the activity of conditional neurons at time t is related to the activity of pre and post neurons at time t+1.
        normalise (bool): Whether to normalise the weights after changing them, such that absolute weights to postsynaptic neuron sums to 1. Default to True.

    Returns:
        None: This function modifies the model weights in place.
    """

    if 'conditional' in df.columns:
        # first group by unique combinations of pre and post
        # Create a pivot table to get the mean conditional activations for each unique pre and post combination
        pivot_table = df.pivot_table(
            index=['pre', 'post'], values='conditional', aggfunc=list).reset_index()
        # get the average activation of the conditionals across timepoints
        conditional_acts = torch.stack([torch.mean(
            model.activations[torch.tensor(cond)], dim=0) for cond in pivot_table.conditional.values])
        # then multiply with the pre and post activations, and the coefficeint
        if offset == 0:
            weight_change = torch.mean(
                # the activation of the conditional neurons relates positively to the plasticity
                model.activations[pivot_table.pre, :] * \
                model.activations[pivot_table.post, :] * conditional_acts,
                dim=1) * coefficient
        else:
            weight_change = torch.mean(
                # the activation of the conditional neurons relates positively to the plasticity
                model.activations[pivot_table.pre, -offset:] * \
                model.activations[pivot_table.post, - \
                                  offset:] * conditional_acts[:, :offset],
                dim=1) * coefficient

    else:
        weight_change = torch.mean(
            model.activations[df.pre, :] * model.activations[df.post, :], dim=1) * coefficient
    # shape of weight_change is the same as the number of rows in the df

    # take out the weights to change
    # NOTE: all_weights have pre in the columns, post in the rows
    weights_subset = model.all_weights[df.post, df.pre]
    # change the weights: keep the signs separate, strengthen (ltp) / weaken (ltd) the connection by using the absolute values
    if mode == 'ltp':
        # strengthen connectivity
        weights_subset_abs = torch.abs(weights_subset) + weight_change.t()
    elif mode == 'ltd':
        # weaken connectivity
        weights_subset_abs = torch.abs(weights_subset) - weight_change.t()
    else:
        raise ValueError("The mode should be either 'ltp' or 'ltd'.")

    # add the sign back
    weights_subset = torch.sign(weights_subset) * \
        torch.clamp(weights_subset_abs, 0, 1)
    # finally modify in the big weights matrix.
    model.all_weights[df.post, df.pre] = weights_subset

    if normalise:
        # make sure all the rows modified (df.post) have their absolute values sum up to 1
        model.all_weights[df.post, :] = model.all_weights[df.post,
                                                          :] / model.all_weights[df.post, :].abs().sum(dim=1)[:, None]
