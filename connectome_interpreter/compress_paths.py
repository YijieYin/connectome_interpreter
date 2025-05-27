# Standard library imports
import math
from typing import List
import os
import gc

# Third-party package imports
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import scipy as sp
import seaborn as sns
import torch
from IPython.display import display
from scipy.sparse import csr_matrix, issparse, csc_matrix, coo_matrix
from tqdm import tqdm

from .utils import (
    dynamic_representation,
    group_edge_by,
    tensor_to_csc,
    to_nparray,
    torch_sparse_where,
    arrayable,
    scipy_sparse_to_pytorch,
)


def compress_paths(
    A: sp.sparse.spmatrix,
    step_number: int,
    threshold: float = 0,
    output_threshold: float = 1e-4,
    root: bool = False,
    chunkSize=5000,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_to_disk: bool = False,
    save_path: str = "./",
    save_prefix: str = "step_",
    return_results: bool = True,
    high_cpu_ram: bool = True,
    output_dtype: np.dtype = np.float32,
    density_threshold: float = 0.2,  # Save as dense if density exceeds this
):
    """
    Computes A^0 to A^n by chunking the computation to save memory.
    Results are thresholded and returned as a list of sparse scipy matrices or numpy
    arrays.

    Args:
        A (scipy.sparse.matrix): The connectivity matrix as a scipy
            sparse matrix.
        step_number (int): Power to raise A to.
        threshold (float): Threshold to apply to the matrix after each multiplication.
            If 0, no thresholding is applied.
        output_threshold (float): Threshold to apply to the output (>=), but not during
            matrix multiplication.
        root (bool): If True, take the n-th root of the result in output.
        chunkSize (int): Size of the chunks to process at a time. This determines
            the memory usage (on GPU if available). e.g. it might need A + (A.shape[0] *
            chunkSize) * float32 / 8 btes.
        device (torch.device): Device to use for computation. Uses GPU if available.
        save_to_disk (bool): If True, save the results to disk.
        save_path (str): Path to save the results.
        save_prefix (str): Prefix to use for the saved files.
        return_results (bool, optional): Whether to return the results as a list
            of sparse matrices. Defaults to True. If False, returns an empty list.
        high_cpu_ram (bool): if high_cpu_ram, keep all the resulting chunked sparse
            matrices in memory, before combining and writing to disk altogether. if
            False, write each chunk to disk to a temporary directory as soon as it is
            computed, and combine them at the end.
        output_dtype (np.dtype): Data type to use for the output matrices. Defaults
            to np.float32.
        density_threshold (float): If the density of the matrix exceeds this threshold,
            the matrix is saved as dense. Defaults to 0.2.

    Returns:
        List[scipy.sparse.csr_matrix or numpy.ndarray]: List of matrices representing
        A^0 to A^n.
    """
    assert A.shape[0] == A.shape[1], "Matrix A must be square"
    assert step_number > 0, "Power n must be positive"

    # Turn A into a torch sparse tensor
    A = scipy_sparse_to_pytorch(A).to(device)

    matrix_size = A.shape[0]
    num_chunks = (matrix_size + chunkSize - 1) // chunkSize  # Ceiling division

    # Create save directory if needed
    if save_to_disk and not os.path.exists(save_path):
        os.makedirs(save_path)

    # Prepare temporary storage method
    if high_cpu_ram:
        # Initialize dictionaries to collect COO data
        rows: dict[int, list[npt.NDArray]] = {idx: [] for idx in range(step_number)}
        cols: dict[int, list[npt.NDArray]] = {idx: [] for idx in range(step_number)}
        data: dict[int, list[npt.NDArray]] = {idx: [] for idx in range(step_number)}
    else:
        # Create temporary directory for chunks
        temp_dir = os.path.join(os.getcwd(), "temp_chunks")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

    # Process each chunk
    for chunk_idx in tqdm(range(num_chunks)):
        # Define the range for this chunk
        start_col = chunk_idx * chunkSize
        end_col = min((chunk_idx + 1) * chunkSize, matrix_size)
        current_chunk_size = end_col - start_col

        for power in range(step_number):
            # Compute the result for this chunk and power
            if power == 0:
                # For A^1, extract the relevant columns from A
                indices = A._indices()
                values = A._values()

                # Filter to include only columns in current chunk
                col_mask = (indices[1, :] >= start_col) & (indices[1, :] < end_col)
                result_indices = indices[:, col_mask].clone()
                result_values = values[col_mask].clone()

                # Adjust column indices to be relative to start_col
                result_indices[1, :] -= start_col

                # Create sparse tensor for this column chunk
                result = torch.sparse_coo_tensor(
                    result_indices,
                    result_values,
                    (matrix_size, current_chunk_size),
                    device=device,
                ).to_dense()

                if threshold > 0:
                    result = torch.where(
                        torch.abs(result) < threshold,
                        torch.tensor(0.0).to(device),
                        result,
                    )
            else:
                # Matrix multiplication for higher powers
                result = torch.sparse.mm(A, result)

                if threshold > 0:
                    result = torch.where(
                        torch.abs(result) < threshold,
                        torch.tensor(0.0).to(device),
                        result,
                    )

                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Apply root if requested
            if root:
                result_root = torch.sign(result) * torch.abs(result) ** (
                    1 / (power + 1)
                )

            # Apply output threshold and collect non-zero entries
            mask = torch.abs(result) >= output_threshold
            chunk_rows, chunk_cols = torch.nonzero(mask, as_tuple=True)

            # Get values based on whether root was applied
            chunk_data: npt.NDArray
            if root:
                chunk_data = (
                    result_root[chunk_rows, chunk_cols]
                    .cpu()
                    .numpy()
                    .astype(output_dtype)
                )
            else:
                chunk_data = (
                    result[chunk_rows, chunk_cols].cpu().numpy().astype(output_dtype)
                )

            # Convert to CPU
            chunk_cols: npt.NDArray = chunk_cols.cpu().numpy()
            chunk_rows: npt.NDArray = chunk_rows.cpu().numpy()

            # Store the data based on RAM usage preference
            if high_cpu_ram:
                # Adjust column indices to global coordinates
                global_cols = chunk_cols + start_col
                # Store in memory
                rows[power].append(chunk_rows)
                cols[power].append(global_cols)
                data[power].append(chunk_data)
            else:
                # Save chunk to disk
                sparse_chunk = sp.sparse.coo_matrix(
                    (chunk_data, (chunk_rows, chunk_cols)),
                    shape=(matrix_size, current_chunk_size),
                )
                sparse_chunk.eliminate_zeros()
                sp.sparse.save_npz(
                    os.path.join(temp_dir, f"step_{power}_chunk_{chunk_idx}.npz"),
                    sparse_chunk,
                )

        # Clear memory
        del result, mask, chunk_rows, chunk_cols, chunk_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Combine chunks and save results
    print("Combining data from all chunks")
    results = []

    for power in tqdm(range(step_number)):
        # Get all data for current power
        if high_cpu_ram:
            # Combine data from memory
            if rows[power]:  # Check if data was collected
                all_rows = np.concatenate(rows[power])
                all_cols = np.concatenate(cols[power])
                all_data = np.concatenate(data[power])
            else:
                all_rows = np.array([])
                all_cols = np.array([])
                all_data = np.array([])
        else:
            # Combine data from disk
            all_rows, all_cols, all_data = [], [], []

            for chunk_idx in range(num_chunks):
                # Load chunk from disk
                sparse_chunk = sp.sparse.load_npz(
                    os.path.join(temp_dir, f"step_{power}_chunk_{chunk_idx}.npz")
                )
                # Adjust column indices
                sparse_chunk.col += chunk_idx * chunkSize
                # Append data
                all_rows.append(sparse_chunk.row)
                all_cols.append(sparse_chunk.col)
                all_data.append(sparse_chunk.data)
                # Clean up
                del sparse_chunk
                gc.collect()

            # Combine all chunks
            all_rows = np.concatenate(all_rows)
            all_cols = np.concatenate(all_cols)
            all_data = np.concatenate(all_data)

        # Calculate density from the data directly
        nnz = len(all_data)
        density = nnz / (matrix_size**2)
        print(
            f"Matrix {power} has {nnz} non-zero elements ({density*100:.6f}% of full matrix)"
        )

        # Choose the appropriate format based on density
        if density > density_threshold:
            print(
                f"Creating matrix {power} in dense format (density: {density*100:.6f}%)"
            )
            # Create dense matrix directly instead of converting from sparse
            dense_result = np.zeros((matrix_size, matrix_size), dtype=output_dtype)
            dense_result[all_rows, all_cols] = all_data

            # Save to disk if requested
            if save_to_disk:
                np.save(
                    os.path.join(save_path, f"{save_prefix}{power}.npy"), dense_result
                )

            # Add to results if requested
            if return_results:
                results.append(dense_result)

            # Clean up
            del dense_result
        else:
            # Create sparse matrix
            sparse_result = sp.sparse.coo_matrix(
                (all_data, (all_rows, all_cols)), shape=(matrix_size, matrix_size)
            ).tocsc()
            sparse_result.eliminate_zeros()

            # Save sparse matrix if requested
            if save_to_disk:
                sp.sparse.save_npz(
                    os.path.join(save_path, f"{save_prefix}{power}.npz"), sparse_result
                )

            # Add to results if requested
            if return_results:
                results.append(sparse_result)

            # Clean up
            del sparse_result

        # Clean up
        del all_rows, all_cols, all_data
        gc.collect()

    # Clean up temporary files if using disk storage
    if not high_cpu_ram:
        for power in range(step_number):
            for chunk_idx in range(num_chunks):
                os.remove(os.path.join(temp_dir, f"step_{power}_chunk_{chunk_idx}.npz"))
        os.rmdir(temp_dir)

    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()

    return results


def compress_paths_dense_chunked(
    inprop: csc_matrix,
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
    """Performs iterative multiplication of a sparse matrix `inprop` for a specified
    number of steps, applying thresholding to filter out values below a certain
    `threshold` to optimize memory usage and computation speed.

    The function is optimized to run on GPU if available. It needs >=
    size_of_inprop.to_dense() * 3 amount of GPU memory, for matrix multiplication, and
    thresholding.

    This function multiplies the connectivity matrix (input in rows; output in columns)
    `inprop` with itself `step_number` times, with each step's result being thresholded
    to zero out elements below a given threshold. The function stores each step's result
    in a list, where each result is further processed to drop values below the
    `output_threshold` to save memory.

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
            filenames. Defaults to ``"step_"``.

    Returns:
        list: A list of scipy.sparse.csc_matrix objects, each representing
            connectivity from all neurons to all neurons n steps away.

    Note:
        This function requires PyTorch and is designed to automatically
        utilize CUDA-enabled GPU devices if available to accelerate
        computations. The input matrix `inprop` is converted to a dense tensor
        before processing.

    Example:
        >>> from scipy.sparse import csc_matrix
        >>> import numpy as np
        >>> inprop = csc_matrix(np.array([[0.1, 0.2], [0.3, 0.4]]))
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
    idx_to_sign: dict,
    target_layer_number: int,
    threshold: float = 0,
    output_threshold: float = 1e-4,
    root: bool = False,
    chunkSize: int = 2000,
    save_path: str = "./",
    return_results: bool = True,
):
    """Calculates the excitatory and inhibitory influences across specified layers of a
    neural network, using PyTorch for GPU acceleration. Even numbers of inhibition is
    treated as excitation.

    Args:
        inprop (scipy.sparse.csc_matrix): The initial connectivity matrix representing
            direct connections between all neurons, shape (n_neurons, n_neurons).
        idx_to_sign (dict): A dictionary mapping neuron indices to the sign of output
            (1 for excitatory, -1 for inhibitory).
        target_layer_number (int): The number of layers through which to calculate
            influences. 1 for direct connections, 2 for one step away, etc.
        threshold (float, optional): A value to threshold the influences during
            calculation; influences below this value are set to zero, and not passed on.
            Defaults to 0.
        output_threshold (float, optional): A threshold for the final output to reduce
            output file size, with values below this threshold set to zero. Defaults to
            1e-4.
        root (bool, optional): Whether to take the nth root of the output. This can be
            understood as "the average direct connection strength" (when root=True), as
            opposed to "the proportion of influence among all partners n steps away"
            (when root=False). Defaults to False.
        chunkSize (int, optional): The size of the chunks to split the matrix into for
            matrix multiplication. Defaults to 5000. A chunk is a dense matrix of size
            (chunkSize, n_neurons). This determines the memory usage: e.g. 2000 * 160000
            * float32 / 8 bytes / 1e9 = 1.28 GB. We need two copies (one excitation, one
            inhibition), plus the sparse representations of the (n_neurons, n_neurons)
            matrix.
        save_path (str, optional): Path to save the results. Defaults to "./".
        return_results (bool, optional): Whether to return the results as a list
            of sparse matrices. Defaults to True.

    Returns:
        Tuple[List[scipy.sparse.csc_matrix], List[scipy.sparse.csc_matrix]]:
            Two lists of sparse matrices representing the excitatory and
            inhibitory influences, respectively, up to the specified target
            layer.
    """
    # if inprop is not a coo matrix, convert it
    if not issparse(inprop):
        # raise error
        raise ValueError("inprop must be a scipy sparse matrix")
    if not isinstance(inprop, coo_matrix):
        inprop = inprop.tocoo()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e_idx = [i for i in range(len(idx_to_sign)) if idx_to_sign[i] == 1]
    i_idx = [i for i in range(len(idx_to_sign)) if idx_to_sign[i] == -1]

    matrix_size = inprop.shape[0]
    num_chunks = (matrix_size + chunkSize - 1) // chunkSize  # Ceiling division

    # make a temporary directory to save the chunks
    temp_dir = os.path.join(os.getcwd(), "temp_chunks")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Filter out e/i rows directly in COO format
    e_mask = np.isin(inprop.row, e_idx)
    i_mask = np.isin(inprop.row, i_idx)

    direct_e_sender = sp.sparse.coo_matrix(
        (
            inprop.data[e_mask],
            (inprop.row[e_mask], inprop.col[e_mask]),
        ),
        shape=inprop.shape,
    ).tocsr()
    direct_i_sender = sp.sparse.coo_matrix(
        (
            inprop.data[i_mask],
            (inprop.row[i_mask], inprop.col[i_mask]),
        ),
        shape=inprop.shape,
    ).tocsr()

    if target_layer_number == 1:
        return [direct_e_sender.tocsc()], [direct_i_sender.tocsc()]

    # shape: (n_neurons, n_neurons)
    direct_e_sender_tensor = scipy_sparse_to_pytorch(direct_e_sender, device=device)
    # shape: (n_neurons, n_neurons)
    direct_i_sender_tensor = scipy_sparse_to_pytorch(direct_i_sender, device=device)

    for chunk_idx in tqdm(range(num_chunks)):
        # Define the range for this chunk
        start_row = chunk_idx * chunkSize
        end_row = min((chunk_idx + 1) * chunkSize, matrix_size)
        current_chunk_size = end_row - start_row

        for layer in range(target_layer_number):
            if layer == 0:
                # For inprop^1, just take the corresponding rows, and turn into tensor
                # shape: (chunkSize, n_neurons)
                dense_e = torch.tensor(
                    direct_e_sender[start_row:end_row, :].toarray(),
                    dtype=torch.float32,
                    device=device,
                )
                dense_i = torch.tensor(
                    direct_i_sender[start_row:end_row, :].toarray(),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                past_dense_e = dense_e.clone()
                # E = ee and ii added
                # shape: (chunkSize, n_neurons)
                dense_e = torch.sparse.mm(
                    dense_e, direct_e_sender_tensor
                ) + torch.sparse.mm(dense_i, direct_i_sender_tensor)

                # I = E * I + I * E
                # shape: (chunkSize, n_neurons)
                dense_i = torch.sparse.mm(
                    past_dense_e, direct_i_sender_tensor
                ) + torch.sparse.mm(dense_i, direct_e_sender_tensor)

            # Apply thresholding
            if threshold > 0:
                dense_e = torch.where(
                    dense_e >= threshold,
                    dense_e,
                    torch.tensor(0.0, device=device),
                )
                dense_i = torch.where(
                    dense_i >= threshold,
                    dense_i,
                    torch.tensor(0.0, device=device),
                )

            # force garbage collection after each multiplication
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Convert to csc for output
            mask_e = torch.abs(dense_e) >= output_threshold
            mask_i = torch.abs(dense_i) >= output_threshold
            chunk_rows_e, chunk_cols_e = torch.nonzero(mask_e, as_tuple=True)
            chunk_rows_i, chunk_cols_i = torch.nonzero(mask_i, as_tuple=True)
            if root:
                chunk_data_e = (
                    (
                        torch.sign(dense_e[chunk_rows_e, chunk_cols_e])
                        * torch.abs(dense_e[chunk_rows_e, chunk_cols_e])
                        ** (1 / (layer + 1))
                    )
                    .cpu()
                    .numpy()
                )
                chunk_data_i = (
                    (
                        torch.sign(dense_i[chunk_rows_i, chunk_cols_i])
                        * torch.abs(dense_i[chunk_rows_i, chunk_cols_i])
                        ** (1 / (layer + 1))
                    )
                    .cpu()
                    .numpy()
                )
            else:
                chunk_data_e = dense_e[chunk_rows_e, chunk_cols_e].cpu().numpy()
                chunk_data_i = dense_i[chunk_rows_i, chunk_cols_i].cpu().numpy()

            # save the chunked data to disk as scipy sparse matrices
            sparse_e = sp.sparse.coo_matrix(
                (
                    chunk_data_e,
                    (chunk_rows_e.cpu().numpy(), chunk_cols_e.cpu().numpy()),
                ),
                shape=(current_chunk_size, matrix_size),
            )
            sparse_i = sp.sparse.coo_matrix(
                (
                    chunk_data_i,
                    (chunk_rows_i.cpu().numpy(), chunk_cols_i.cpu().numpy()),
                ),
                shape=(current_chunk_size, matrix_size),
            )
            sparse_e.eliminate_zeros()
            sparse_i.eliminate_zeros()

            # save the sparse matrices to disk
            sp.sparse.save_npz(
                os.path.join(temp_dir, f"step_{layer}_e_{chunk_idx}.npz"), sparse_e
            )
            sp.sparse.save_npz(
                os.path.join(temp_dir, f"step_{layer}_i_{chunk_idx}.npz"), sparse_i
            )
            del sparse_e, sparse_i
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Combine all data -----------------
    print("Combining data from all chunks")
    stepse = []
    stepsi = []
    # if save_path isn't already a directory, make it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for layer in tqdm(range(target_layer_number)):
        # first combine the es ----
        rows = []
        cols = []
        data = []
        for chunk_idx in range(num_chunks):
            # load the chunked data from disk
            sparse_e = sp.sparse.load_npz(
                os.path.join(temp_dir, f"step_{layer}_e_{chunk_idx}.npz")
            )
            # adjust the row indices to account for the chunk offset
            sparse_e.row += chunk_idx * chunkSize
            # append the data to the lists
            rows.append(sparse_e.row)
            cols.append(sparse_e.col)
            data.append(sparse_e.data)
            # delete the sparse matrix to save memory
            del sparse_e
            gc.collect()

        # combine the data
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        # create the sparse matrix
        sparse_e = sp.sparse.coo_matrix(
            (data, (rows, cols)), shape=(matrix_size, matrix_size)
        ).tocsc()
        sparse_e.eliminate_zeros()
        # save the sparse matrix to disk
        sp.sparse.save_npz(os.path.join(save_path, f"step_{layer}_e.npz"), sparse_e)
        if return_results:
            stepse.append(sparse_e)
        # delete the sparse matrix to save memory
        del sparse_e
        gc.collect()

        # now combine the is ----
        rows = []
        cols = []
        data = []
        for chunk_idx in range(num_chunks):
            # load the chunked data from disk
            sparse_i = sp.sparse.load_npz(
                os.path.join(temp_dir, f"step_{layer}_i_{chunk_idx}.npz")
            )
            # adjust the row indices to account for the chunk offset
            sparse_i.row += chunk_idx * chunkSize
            # append the data to the lists
            rows.append(sparse_i.row)
            cols.append(sparse_i.col)
            data.append(sparse_i.data)
            # delete the sparse matrix to save memory
            del sparse_i
            gc.collect()
        # combine the data
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        # create the sparse matrix
        sparse_i = sp.sparse.coo_matrix(
            (data, (rows, cols)), shape=(matrix_size, matrix_size)
        ).tocsc()
        sparse_i.eliminate_zeros()
        if return_results:
            stepsi.append(sparse_i)
        # save the sparse matrix to disk
        sp.sparse.save_npz(os.path.join(save_path, f"step_{layer}_i.npz"), sparse_i)
        # delete the sparse matrix to save memory
        del sparse_i
        gc.collect()

    # remove the temporary directory
    for layer in range(target_layer_number):
        for chunk_idx in range(num_chunks):
            os.remove(os.path.join(temp_dir, f"step_{layer}_e_{chunk_idx}.npz"))
            os.remove(os.path.join(temp_dir, f"step_{layer}_i_{chunk_idx}.npz"))
    os.rmdir(temp_dir)
    # remove all variables
    del direct_e_sender, direct_i_sender
    torch.cuda.empty_cache()
    # clear the cache
    gc.collect()

    return stepse, stepsi


def compress_paths_signed_no_chunking(
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
    lne_initial = lne.clone()
    lni_initial = lni.clone()

    steps_excitatory = []
    steps_inhibitory = []

    for layer in tqdm(range(target_layer_number)):
        if layer != 0:
            # if layer ==0, return the direct excitatory and inhibitory
            # connections separately
            lne_new = torch.matmul(lne, lne_initial) + torch.matmul(lni, lni_initial)
            lni = torch.matmul(lne, lni_initial) + torch.matmul(lni, lne_initial)

            # Apply thresholding
            if threshold > 0:
                lne_new = torch_sparse_where(lne_new, threshold)
                lni = torch_sparse_where(lni, threshold)

            # Dynamic representation based on density
            lne = dynamic_representation(lne_new)
            lni = dynamic_representation(lni)
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
        >>> from scipy.sparse import csc_matrix
        >>> matrices = [csc_matrix([[1, 2], [3, 4]]),
                        csc_matrix([[5, 6], [7, 8]]),
                        csc_matrix([[9, 10], [11, 12]])]
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
    outprop: bool = False,
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
        threshold_axis (str, optional): The axis to apply the display_threshold to.
            Defaults to 'row' (removing entire rows if no value exceeds
            display_threshold).
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

    if not outprop:
        # Sum across rows: presynaptic neuron is in the rows
        # summing across neurons of the same type: total amount of input from that
        # type for the postsynaptic neuron
        summed_df = df.groupby(df.index).sum()

        # Average across columns and transpose back
        # averaging across columns of the same type:
        # on average, a neuron of that type receives x% input from a presynaptic type
        result_df = summed_df.T.groupby(level=0).mean().T
    else:
        # calculate the output proportion from an average source neuron to a target type
        # so first sum across columns
        summed_df = df.T.groupby(level=0).sum().T
        # then average across rows
        result_df = summed_df.groupby(summed_df.index).mean()

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
    """Plots the connection strength from all of inidx (grouped by inidx_map) to an
    average outidx (grouped by outidx_map) over different path lengths. Either inidx_map
    or outidx_map, but not both, should be provided. If neither is provided, presynaptic
    neurons are grouped together. Direct connections are in path_length 1.

    Args:
        steps (list of scipy.sparse matrices or numpy.array): List of sparse matrices,
            each representing synaptic strengths for a specific path length.
        inidx (int, float, list, set, numpy.ndarray, or pandas.Series): Array of indices
            representing input (presynaptic) neurons.
        outidx (int, float, list, set, numpy.ndarray, or pandas.Series): Array of
            indices representing output (postsynaptic) neurons.
        outidx_map (dict): Mapping from indices to postsynaptic neuron groups. Only one
            of inidx_map and outidx_map should be specified.
        inidx_map (dict): Mapping from indices to presynaptic neuron groups. Only one of
            inidx_map and outidx_map should be specified.
        width (int, optional): The width of the plot. Defaults to 800.
        height (int, optional): The height of the plot. Defaults to 400.

    Returns:
        None: Displays an interactive line plot showing the connection strength from all
            of inidx to an average outidx over different path lengths.

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
        if hasattr(step, "toarray"):
            data = step[:, outidx][inidx, :].toarray()
        else:
            data = step[inidx, :][:, outidx]
        if inidx_map is not None:
            df = pd.DataFrame(
                data=data,
                index=[inidx_map[key] for key in inidx],
            )
            # average of all columns
            # then groupby index, and take the sum
            # average post from all pre
            df = df.mean(axis=1).groupby(level=0).sum().to_frame().T
            rows.append(df)
        elif outidx_map is not None:
            df = pd.DataFrame(
                data=data,
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
    fig.update_layout(
        xaxis=dict(tickmode="linear", tick0=1, dtick=1),
        yaxis=dict(tickmode="linear", tick0=0, dtick=contri.value.max() / 5),
    )
    # fig.show()

    return fig


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
    prefix: str, file_path: str | None = None, first_n: int | None = None
) -> List:
    """Reads the precomputed compressed paths.

    Args:
        prefix (str): The prefix/folder name (expected to be the same) of the
            files to read.
        file_path (str, optional): The path to the files. Defaults to None. If
            None, checks if running in Google Colab, and sets the path: if
            running in Colab, sets the path to "/content/"; otherwise, sets the
            path to "".
        first_n (int, optional): Number of files to read. If None, reads all files.

    Returns:
        List: A list of matrices (sparse or dense) representing the steps.
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

    # Get all files with the prefix pattern
    folder_path = os.path.join(file_path, prefix)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory {folder_path} does not exist")

    files = os.listdir(folder_path)

    # Filter files that match the pattern prefix_i.npz or prefix_i.npy
    step_files = []
    for f in files:
        if f.startswith(f"{prefix}_") and (f.endswith(".npz") or f.endswith(".npy")):
            # Extract the step number from the filename
            try:
                step_num = int(f.split("_")[-1].split(".")[0])
                step_files.append((step_num, f))
            except ValueError:
                # Skip files that don't match the expected pattern
                continue

    # Sort by step number to ensure correct order
    step_files.sort(key=lambda x: x[0])

    if first_n is None:
        first_n = len(step_files)
    else:
        first_n = min(first_n, len(step_files))

    # Load the first_n files
    for i in range(first_n):
        step_num, filename = step_files[i]
        file_full_path = os.path.join(folder_path, filename)

        if filename.endswith(".npz"):
            # Load sparse matrix
            matrix = sp.sparse.load_npz(file_full_path)
        elif filename.endswith(".npy"):
            # Load dense matrix
            matrix = np.load(file_full_path)
        else:
            # This shouldn't happen given our filtering, but just in case
            continue

        steps_cpu.append(matrix)

    if len(steps_cpu) == 0:
        raise FileNotFoundError(
            f"No files found with pattern {prefix}_*.np[yz] in {folder_path}"
        )

    return steps_cpu
