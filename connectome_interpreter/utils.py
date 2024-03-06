import torch
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import random 
import matplotlib.colors as mcl
import itertools

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

def modify_coo_matrix(coo, input_idx=None, output_idx=None, value=None, updates_df=None, re_normalize=True):
    """
    Modify the values of a COO sparse matrix at specified indices.

    There are two modes of operation:
    1. Single update mode: where `input_idx`, `output_idx`, and `value` are provided as individual arguments. In this case all combinations of input_idx and output_idx are updated. 
    2. Batch update mode: where `updates_df` is provided, a DataFrame with columns ['input_idx', 'output_idx', 'value'].

    Args:
      coo (coo_matrix): The COO sparse matrix to modify.
      input_idx (int, list, numpy.ndarray, set, optional): Row indices for updates.
      output_idx (int, list, numpy.ndarray, set, optional): Column indices for updates.
      value (numeric, list, numpy.ndarray, optional): New values for updates. If it is a number, then it is used for all updates. Else, it needs to be of length len(input_idx) * len(output_idx).
      updates_df (DataFrame, optional): The DataFrame containing batch updates.
      re_normalize (bool, optional): Whether to re-normalize the matrix after updating, such that all values in each column sum up to 1. Default to True.

    Note:
      In single update mode, `input_idx`, `output_idx`, and `value` must all be provided.
      In batch update mode, only `updates_df` should be provided.
      If both modes are provided, the function will prioritize the single update mode.

    Returns:
      coo_matrix: The updated COO sparse matrix.
    """
    updated_cols = set()  # Track columns that are updated or added
    if input_idx is not None and output_idx is not None and value is not None:
        # Convert sets to numpy arrays and ensure all inputs are numpy arrays
        if isinstance(input_idx, set): input_idx = np.array(list(input_idx))
        else: input_idx = np.atleast_1d(input_idx)
        if isinstance(output_idx, set): output_idx = np.array(list(output_idx))
        else: output_idx = np.atleast_1d(output_idx)
        
        value = np.atleast_1d(value)

        # Ensure 'value' can be a single value or an array
        value = np.full(len(input_idx) * len(output_idx), value[0]) if len(value) == 1 else value

        # Adjust index to reflect combination of input_idx and output_idx
        combination_index = 0

        for i, j in itertools.product(input_idx, output_idx):
            val = value[combination_index]
            idx = np.where((coo.row == i) & (coo.col == j))[0]
            if idx.size > 0:
                coo.data[idx] = val
            elif val != 0:  # Add new value if it doesn't exist
                coo.row = np.append(coo.row, i)
                coo.col = np.append(coo.col, j)
                coo.data = np.append(coo.data, val)
            updated_cols.add(j)
            combination_index += 1  # Move to the next value for the next combination
    
    elif updates_df is not None:
        for _, row in updates_df.iterrows():
            i, j, val = row['input_idx'], row['output_idx'], row['value']
            idx = np.where((coo.row == i) & (coo.col == j))[0]
            if idx.size > 0:
                coo.data[idx] = val
            elif val != 0:
                coo.row = np.append(coo.row, i)
                coo.col = np.append(coo.col, j)
                coo.data = np.append(coo.data, val)
            updated_cols.add(j)
    
    # Remove zero entries by reconstructing the matrix
    nonzero_indices = coo.data != 0
    coo = coo_matrix((coo.data[nonzero_indices], (coo.row[nonzero_indices],
                     coo.col[nonzero_indices])), shape=coo.shape)
    
    if re_normalize and updated_cols:
        csr = coo.tocsr()  # Convert for efficient column operations
        for col in updated_cols:
            col_sum = np.sum(csr[:, col])
            if col_sum != 0:
                csr[:, col] /= col_sum
        coo = csr.tocoo()  # Convert back to COO

    return coo



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