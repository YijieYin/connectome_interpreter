import torch
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix


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
