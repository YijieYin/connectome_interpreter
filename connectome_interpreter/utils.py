import torch
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import random 
import matplotlib.colors as mcl

import nglscenes as ngl


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


def modify_coo_matrix(coo, input_idx=None, output_idx=None, value=None, updates_df=None):
    """
    Modify the values of a COO sparse matrix at specified indices.

    There are two modes of operation:
    1. Single update mode: where `input_idx`, `output_idx`, and `value` are provided as individual arguments.
    2. Batch update mode: where `updates_df` is provided, a DataFrame with columns ['input_idx', 'output_idx', 'value'].

    Args:
      coo (coo_matrix): The COO sparse matrix to modify.
      input_idx (int, optional): The row index for the single update mode.
      output_idx (int, optional): The column index for the single update mode.
      value (numeric, optional): The new value for the single update mode.
      updates_df (DataFrame, optional): The DataFrame containing batch updates.

    Note:
      In single update mode, `input_idx`, `output_idx`, and `value` must all be provided.
      In batch update mode, only `updates_df` should be provided.
      If both modes are provided, the function will prioritize the single update mode.

    Returns:
      coo_matrix: The updated COO sparse matrix.
    """
    if input_idx is not None and output_idx is not None and value is not None:
        # Single update mode
        update_index = np.where((coo.row == input_idx)
                                & (coo.col == output_idx))[0]
        if update_index.size > 0:
            coo.data[update_index] = value
    elif updates_df is not None:
        # Batch update mode
        for idx, row in updates_df.iterrows():
            update_index = np.where((coo.row == row['input_idx']) & (
                coo.col == row['output_idx']))[0]
            if update_index.size > 0:
                coo.data[update_index] = row['value']
    else:
        # No valid update mode provided
        raise ValueError(
            "Invalid parameters: provide either single indices and value or an updates dataframe.")

    # Remove zero entries by reconstructing the matrix
    nonzero_indices = coo.data != 0
    coo = coo_matrix((coo.data[nonzero_indices], (coo.row[nonzero_indices],
                     coo.col[nonzero_indices])), shape=coo.shape)
    return coo


def silence_connections(coo, input_idx, output_idx):
    """
    Sets the values of specified connections in a COO sparse matrix to zero, effectively "silencing" these connections.

    Args:
      coo (coo_matrix): The COO sparse matrix to modify.
      input_idx (int, list, numpy.ndarray, set): Row indices of the connections to silence. Can be a single index or a collection of indices.
      output_idx (int, list, numpy.ndarray, set): Column indices of the connections to silence. Can be a single index or a collection of indices.

    Note:
      `input_idx` and `output_idx` should be the same type and length if they are lists, arrays, or sets.
      This function leverages `modify_coo_matrix` for applying updates.

    Returns:
      coo_matrix: The updated COO sparse matrix with specified connections silenced.
    """
    # Convert input_idx and output_idx to numpy arrays if they are not already
    input_idx = np.array(list(input_idx)) if isinstance(
        input_idx, (list, set)) else np.array([input_idx])
    output_idx = np.array(list(output_idx)) if isinstance(
        output_idx, (list, set)) else np.array([output_idx])

    # Check if input_idx and output_idx are single values and create all combinations if they are arrays
    if input_idx.size > 1 or output_idx.size > 1:
        input_idx_combinations, output_idx_combinations = np.meshgrid(
            input_idx, output_idx, indexing='ij')
        input_idx_flattened = input_idx_combinations.ravel()
        output_idx_flattened = output_idx_combinations.ravel()
    else:
        # If single values, just use them directly
        input_idx_flattened = input_idx
        output_idx_flattened = output_idx

    # Create updates DataFrame with 'value' column set to 0 for silencing
    updates_df = pd.DataFrame({
        'input_idx': input_idx_flattened,
        'output_idx': output_idx_flattened,
        'value': 0
    })

    # Update the COO matrix using the modify_coo_matrix function
    updated_coo = modify_coo_matrix(coo, updates_df=updates_df)
    return updated_coo


def to_nparray(input_data):
    """
    Converts the input data into a numpy array, filtering out any NaN values. 
    The input can be a single number, a list, a set, or a numpy array.

    Args:
        input_data: The input data to convert. Can be of type int, float, list, set, or numpy.ndarray.

    Returns:
        numpy.ndarray: A numpy array created from the input data, with all NaN values removed.
    """
    # First, ensure the input is in array form
    if isinstance(input_data, (int, float)):
        input_array = np.array([input_data])
    elif isinstance(input_data, (list, set, np.ndarray)):
        input_array = np.array(list(input_data))
    else:
        raise TypeError(
            "Input data must be an int, float, list, set, or numpy.ndarray")

    # Then, remove NaN values
    cleaned_array = input_array[~pd.isna(input_array)]

    return cleaned_array

def get_ngl_link(df, no_connection_invisible = True, colour_saturation = 0.4, scene = None):
    """
    Generates a Neuroglancer link with layers, corresponding to each column in the df. The function
    processes a dataframe, adding colour information to each neuron,
    and then creates layers in a Neuroglancer scene for visualization. Separate layers
    are created for each column in the DataFrame, with unique segment colors based on
    the values in the DataFrame.

    Parameters:
    - df (pandas.DataFrame): A DataFrame containing neuron metadata. The index should
      contain neuron identifiers (root_ids), and columns should represent different
      attributes or categories.
    - no_connection_invisible (bool, optional): Whether to make invisible neurons that are not connected. Default to True (invisible). 
    - colour_saturation (float, optional): The saturation of the colours. Default to 0.4.
    - scene (ngl.Scene, optional): A Neuroglancer scene object from nglscenes package. You can read a scene from clipboard like `scene = Scene.from_clipboard()`. 

    Returns:
    - str: A URL to the generated Neuroglancer scene.

    Notes:
    - The function assumes that the 'scene1' variable is defined in the global scope and
      is an instance of ngl.Scene.
    - The function creates separate Neuroglancer layers for each column in the DataFrame,
      using the column name as the layer name.
    - The root_ids are colored based on the values in the DataFrame, with a color scale
      ranging from white (minimum value) to a specified color (maximum value).
    - The function relies on the Neuroglancer library for layer creation and scene manipulation.
    """

    try:
        import nglscenes as ngl
    except ImportError:
        raise ImportError("To use this function, please install the niche package by running 'pip3 install git+https://github.com/schlegelp/nglscenes@main]'")
    
    # define a scene if not given: 
    if scene == None: 
        # Initialize a scene
        scene = ngl.Scene()
        scene['layout'] = '3d'
        scene['position'] = [527216.1875, 208847.125, 84774.0625]
        scene['projectionScale'] = 400000
        scene['dimensions'] = {"x": [1e-9,"m"], "y": [1e-9,"m"], "z": [1e-9, "m"]}

        # and another for FAFB mesh
        fafb_layer = ngl.SegmentationLayer(source = 'precomputed://https://spine.itanna.io/files/eric/jfrc_mesh_test',
                                          name = 'jfrc_mesh_test1')
        fafb_layer['segments'] = ['1']
        fafb_layer['objectAlpha'] = 0.17
        fafb_layer['selectedAlpha'] = 0.55
        fafb_layer['segmentColors'] = {'1':'#cacdd8'}
        fafb_layer['colorSeed'] = 778769298
        scene.add_layers(fafb_layer)

        # and the neuropil layer with names
        np_layer = ngl.SegmentationLayer(source = 'precomputed://gs://neuroglancer-fafb-data/elmr-data/FAFBNP.surf/mesh#type=mesh',
                                        name = 'neuropil_regions_surface_named')
        np_layer['segments'] = [str(num) for num in range(0, 79)]
        np_layer['visible'] = False
        np_layer['objectAlpha'] = 0.17
        scene.add_layers(np_layer)


    # Define a list of colors optimized for human perception on a dark background
    colors = [
        "#ff6b6b", "#f06595", "#cc5de8", "#845ef7", "#5c7cfa", "#339af0", "#22b8cf", "#20c997", "#51cf66", "#94d82d", "#fcc419",
        "#4ecdc4", "#ffe66d", "#7bed9f", "#a9def9", "#e2c1f9", "#f694c1", "#ead5dc", "#c7f0bd", "#f0e5cf", "#ffc5a1",
        "#ff8c94", "#ffaaa6", "#ffd3b5", "#dcedc1", "#a8e6cf", "#a6d0e4", "#c1beff", "#f5b0cb", "#ffcfdf", "#fde4cf", "#f1e3d3"
    ]

    # Normalize the values in the DataFrame
    df_norm = (df - df.min().min()) / (df.max().max() - df.min().min())

    scene['layout'] = '3d'

    source = 'precomputed://gs://flywire_v141_m783'

    for column in df.columns:
        color = random.choice(colors)
        cmap = mcl.LinearSegmentedColormap.from_list("custom_cmap", ["white", color])

        df_group = df_norm[[column]]
        if no_connection_invisible: 
          df_group = df_group[df_group.iloc[:,0]>0]

        layer = ngl.SegmentationLayer(source=source, name=str(column))

        layer['segments'] = list(df_group.index.astype(str))
        layer['segmentColors'] = {
            # trick from Claud to make the colours more saturated 
            str(root_id): mcl.to_hex(cmap(colour_saturation + (1-colour_saturation)*value.values[0]))
            for root_id, value in df_group.iterrows()
        }

        scene.add_layers(layer)

    return scene.url