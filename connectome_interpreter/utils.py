import torch
import pandas as pd
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

    Parameters:
    - x: The input tensor to apply the threshold to.
    - threshold: The threshold value.

    Returns:
    - A new tensor with values below the threshold set to zero.
    """
    if x.is_sparse:
      values = x._values()
      indices = x._indices()

      # Apply the threshold to the values
      thresholded_values = torch.where(values >= threshold, values, torch.tensor(0.0, device=x.device, dtype=x.dtype))

      # Create a new sparse tensor with the same indices but updated values
      result = torch.sparse_coo_tensor(indices, thresholded_values, x.size(), device=x.device, dtype=x.dtype).coalesce()
      # It might be beneficial to remove zero values that were newly created by thresholding
      # .coalesce() combines duplicates and removes zeros if present
    else:
      result = torch.where(x >= threshold, x, torch.tensor(0.0, device=x.device))

    return result

def tensor_to_csc(tensor): 
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

  Parameters:
  - coo_tensor (torch.Tensor): A PyTorch tensor, either already in sparse COO format or dense.

  Returns:
  - pd.DataFrame: A DataFrame with columns 'row_idx', 'col_idx', and 'value', representing the edge list of the graph.
  """

  if not coo_tensor.is_sparse: 
    coo_tensor = coo_tensor.to_sparse_coo()

  indices = coo_tensor.indices().t().cpu().numpy()  # Transpose and convert to numpy array
  values = coo_tensor.values().cpu().numpy()
  
  # Split the indices to row and column
  row_idx, col_idx = indices[:, 0], indices[:, 1]
  
  edge_list_df = pd.DataFrame({'row_idx': row_idx, 'col_idx': col_idx, 'value': values})
  return edge_list_df

def coo_to_el(coo): 
  """
  Convert a SciPy sparse COO matrix to a DataFrame representing an edge list.

  Extracts the row indices, column indices, and data from a COO matrix to create a DataFrame.
  Each row in the DataFrame represents an edge in the graph, where 'row_idx' and 'col_idx' are the nodes connected by the edge,
  and 'value' is the weight of the edge.

  Parameters:
  - coo (scipy.sparse.coo_matrix): A COO matrix from SciPy representing a sparse matrix.

  Returns:
  - pd.DataFrame: A DataFrame with columns 'row_idx', 'col_idx', and 'value', representing the edge list of the graph.
  """
  row = coo.row
  col = coo.col
  data = coo.data
  all_el = pd.DataFrame({'row_idx': row, 'col_idx': col, 'value': data})

  return all_el 
