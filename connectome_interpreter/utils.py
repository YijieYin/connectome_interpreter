# Standard library imports
import random
import warnings
from collections import defaultdict
from numbers import Real
from typing import Collection, List, Tuple, Optional, Union
import os

import ipywidgets as widgets
import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from IPython.display import display, IFrame, HTML
from scipy.sparse import coo_matrix, issparse
import seaborn as sns

import plotly.graph_objects as go
import plotly.colors

# types that can be made into a numeric numpy array
arrayable = npt.ArrayLike


def dynamic_representation(tensor, density_threshold=0.2):
    """Convert tensor to sparse if density is below threshold, otherwise to
    dense. This might be memory intensive."""
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
    Apply a threshold to a tensor, setting values below the threshold to zero.
    This function allows the tensor to be sparse. torch.where() does not.

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
            values >= threshold,
            values,
            torch.tensor(0.0, device=x.device, dtype=x.dtype),
        )

        # Create a new sparse tensor with the same indices but updated values
        result = torch.sparse_coo_tensor(
            indices,
            thresholded_values,
            x.size(),
            device=x.device,
            dtype=x.dtype,
        ).coalesce()
        # It might be beneficial to remove zero values that were newly created
        # by thresholding .coalesce() combines duplicates and removes zeros if
        # present
    else:
        result = torch.where(x >= threshold, x, torch.tensor(0.0, device=x.device))

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
        # memory usage can be further reduced by first spliting the tensor/COO
        # matrix into smaller chunks, for future work

    return coo.tocsc()


def coo_tensor_to_el(coo_tensor):
    """
    Convert a PyTorch sparse COO tensor to a DataFrame representing an edge
    list.

    This function checks if the input tensor is sparse. If not, it converts it
    to a sparse COO tensor. It then extracts the indices and values, and
    creates a DataFrame with columns 'pre', 'post', and 'weight'. Each row in
    the DataFrame represents an edge in the graph, where 'pre' and 'post' are
    the nodes connected by the edge, and 'weight' is the weight of the edge.

    Args:
      coo_tensor (torch.Tensor): A PyTorch tensor, either already in sparse
        COO format or dense.

    Returns:
      pd.DataFrame: A DataFrame with columns 'pre', 'post', and 'weight',
        representing the edge list of the graph.
    """

    if not coo_tensor.is_sparse:
        coo_tensor = coo_tensor.to_sparse_coo()

    # Transpose and convert to numpy array
    indices = coo_tensor.indices().t().cpu().numpy()
    values = coo_tensor.values().cpu().numpy()

    # Split the indices to row and column
    pre, post = indices[:, 0], indices[:, 1]

    edge_list_df = pd.DataFrame({"pre": pre, "post": post, "weight": values})
    return edge_list_df


def coo_to_el(coo, row_indices=None, col_indices=None):
    """
    Extracts an edgelist from a COO matrix, optionally using pre-specified row
    and/or column indices. If row_indices or col_indices are None, all rows or
    columns are considered, respectively. Each row in the resulting DataFrame
    represents an edge in the graph, where 'pre' and 'post' are the nodes
    connected by the edge, and 'value' is the weight of the edge.

    Args:
      coo: A scipy.sparse.coo_matrix instance.
      row_indices: Optional; A list or array of row indices of interest.
      col_indices: Optional; A list or array of column indices of interest.

    Returns:
      pd.DataFrame: A DataFrame with columns 'pre', 'post', and 'value',
        representing the edge list of the graph.
    """
    # Determine whether to filter on rows/columns based on provided indices
    row_mask = (
        np.full(coo.shape[0], True)
        if row_indices is None
        else np.isin(coo.row, row_indices)
    )
    col_mask = (
        np.full(coo.shape[1], True)
        if col_indices is None
        else np.isin(coo.col, col_indices)
    )

    # Combine row and column masks
    mask = row_mask & col_mask

    # Filter rows, cols, and data based on the combined mask
    rows = coo.row[mask]
    cols = coo.col[mask]
    data = coo.data[mask]

    all_el = pd.DataFrame({"pre": rows, "post": cols, "value": data})

    return all_el


def adjacency_df_to_el(adjacency, threshold=None):
    """
    Convert a DataFrame representing an adjacency matrix to an edge list.

    This function takes a DataFrame where the index and columns represent
    nodes in the graph, and the values represent the weights of the edges
    between them. It converts this DataFrame to an edge list, where each row
    represents an edge in the graph, with columns 'pre', 'post', and 'weight'.

    Args:
      adjacency (pd.DataFrame): A DataFrame representing an adjacency matrix.
      threshold (float, optional): If provided, edges with values below this
        threshold are removed. Default to None.

    Returns:
      pd.DataFrame: A DataFrame with columns 'pre', 'post', and 'weight',
        representing the edge list of the graph.
    """
    if threshold is not None:
        adjacency = adjacency.where(adjacency >= threshold, 0)

    # Stack the DataFrame to create a long format
    adjacency = adjacency.stack().reset_index()
    adjacency.columns = ["pre", "post", "weight"]
    adjacency = adjacency[adjacency["weight"] != 0]

    return adjacency


def modify_coo_matrix(
    sparse_matrix,
    input_idx: arrayable | None = None,
    output_idx: arrayable | None = None,
    value=None,
    updates_df: pd.DataFrame | None = None,
    re_normalize: bool = True,
):
    """
    Modify the values of a sparse matrix (either COO or CSR format) at
    specified indices.

    There are two modes of operation:

    1. Single update mode: where `input_idx`, `output_idx`, and `value` are
    provided as individual arguments. In this case all combinations of
    input_idx and output_idx are updated.

    2. Batch update mode: where `updates_df` is provided, a DataFrame with
    columns ['input_idx', 'output_idx', 'value'].

    Args:
        sparse_matrix (coo_matrix or csr_matrix): The sparse matrix to modify.
        input_idx (int, list, numpy.ndarray, set, optional): Row indices for
            updates.
        output_idx (int, list, numpy.ndarray, set, optional): Column indices
            for updates.
        value (numeric, list, numpy.ndarray, optional): New values for updates.
            If it is a number, then it is used for all updates. Else, it needs
            to be of length equal to the product of the lengths of input_idx
            and output_idx.
        updates_df (DataFrame, optional): The DataFrame containing batch
            updates.
        re_normalize (bool, optional): Whether to re-normalize the matrix after
            updating, such that all values in each column sum up to 1. Default
            to True.

    Note:
        In single update mode, `input_idx`, `output_idx`, and `value` must all
        be provided.
        In batch update mode, only `updates_df` should be provided.
        If both modes are provided, the function will prioritize the single
        update mode.

    Returns:
        coo_matrix or csr_matrix: The updated sparse matrix, in the same format
        as the input.
    """

    if not issparse(sparse_matrix):
        raise TypeError("The provided matrix is not a sparse matrix.")

    # Convert sparse matrix to COO format for direct edge manipulation
    original_format = sparse_matrix.getformat()
    original_dtype = sparse_matrix.dtype
    if original_format != "coo":
        sparse_matrix = sparse_matrix.tocoo()

    # Extract edges as a DataFrame
    edges = pd.DataFrame(
        {
            "input_idx": sparse_matrix.row,
            "output_idx": sparse_matrix.col,
            "value": sparse_matrix.data,
        }
    )

    print("Updating matrix...")
    if input_idx is not None and output_idx is not None and value is not None:
        # Create updates DataFrame from provided indices and values
        input_idx = to_nparray(input_idx)
        output_idx = to_nparray(output_idx)

        if len(input_idx) == 0 or len(output_idx) == 0:
            # warning
            warnings.warn(
                "No updates were provided. Returning the original matrix.",
                UserWarning,
            )
            return sparse_matrix

        if np.isscalar(value):
            value = np.full(len(input_idx) * len(output_idx), value)
        elif len(value) != len(input_idx) * len(output_idx):
            raise ValueError(
                "Length of 'value' must match the product of the lengths of 'input_idx' and 'output_idx', or be one single value."
            )

        updates_df = pd.DataFrame(
            {
                "input_idx": np.repeat(input_idx, len(output_idx)),
                "output_idx": np.tile(output_idx, len(input_idx)),
                "value": value,
            }
        )

    elif updates_df is not None:
        # check column names
        if not all(
            col in updates_df.columns for col in ["input_idx", "output_idx", "value"]
        ):
            raise ValueError(
                "The DataFrame must contain columns ['input_idx', 'output_idx', 'value']."
            )

    else:
        raise ValueError(
            "Either ['input_idx', 'output_idx', and 'value'], or 'updates_df' must be provided."
        )

    # Concatenate edges and updates_df, ensuring that updates_df is last so
    # its values take precedence
    combined_edges = pd.concat([edges, updates_df])

    # Remove duplicates, keeping the last occurrence (from updates_df)
    # We specify the subset to be the columns 'input_idx' and 'output_idx'
    edges = combined_edges.drop_duplicates(
        subset=["input_idx", "output_idx"], keep="last"
    )

    if re_normalize:
        print("Re-normalizing updated columns...")
        edge_to_normalise = edges[edges.output_idx.isin(updates_df.output_idx)]
        col_sums = edge_to_normalise.groupby("output_idx").value.sum()
        col_sums[col_sums == 0] = 1

        edge_to_normalise.loc[:, ["value"]] = edge_to_normalise[
            "value"
        ] / edge_to_normalise["output_idx"].map(col_sums)

        edges = pd.concat(
            [
                edges[~edges.output_idx.isin(updates_df.output_idx)],
                edge_to_normalise,
            ]
        )

    # Convert back to original sparse matrix format
    updated_matrix = coo_matrix(
        (
            edges["value"].astype(original_dtype),
            (edges["input_idx"], edges["output_idx"]),
        ),
        shape=sparse_matrix.shape,
    )

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
            UserWarning,
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


def to_nparray(input_data: arrayable, unique: bool = True) -> npt.NDArray:
    """
    Converts the input data into a numpy array, filtering out any NaN values
    (and duplicates). The input can be a single number, a list, a set, a numpy
    array, or a pandas Series.

    Args:
        input_data: The input data to convert. Can be of type int (including
            numpy.int64 and numpy.int32), float, list, set, numpy.ndarray,
            pandas.Series, or pandas.Index.
        unique (bool, optional): Whether to return only unique values. Default
            to True. NOTE: np.unique() sorts the array.

    Returns:
        numpy.ndarray: A unique numpy array created from the input data, with
            all NaN values removed.
    """
    # First, ensure the input is in array form and convert pd.Series to
    # np.ndarray
    if isinstance(input_data, (int, float, np.int64, np.int32)):
        input_array = np.array([input_data])
    elif isinstance(input_data, (list, set, np.ndarray, pd.Series, pd.Index)):
        input_array = np.array(list(input_data))
    else:
        raise TypeError(
            "Input data must be an int, float, list, set, numpy.ndarray, or pandas.Series"
        )

    # Then, remove NaN values
    cleaned_array = input_array[~pd.isna(input_array)]

    # Finally, return unique values
    if unique:
        return np.unique(cleaned_array)
    else:
        return cleaned_array


def get_ngl_link(
    df: pd.DataFrame | pd.Series,
    no_connection_invisible: bool = True,
    group_by: dict | None = None,
    colour_saturation: float = 0.4,
    scene=None,
    source: list | None = None,
    normalise: str | None = None,
    include_postsynaptic_neuron: bool = False,
    diff_colours_per_layer: bool = False,
    colors: list | None = None,
    colormap: str = "viridis",
    df_format: str = "wide",
    open_here: bool = False,
    width: int = 1500,
    height: int = 800,
) -> str:
    """
    Generates a Neuroglancer link with layers based on the neuron ids and the
    values in `df`.

    Args:
        df (pandas.DataFrame or pandas.Series): A DataFrame containing neuron
            metadata. If `df_format` == `wide` (default), the index should
            contain neuron identifiers (bodyId/root_ids), and columns should
            represent different attributes, timesteps or categories. If
            `df_format` == `long`, the DataFrame should contain three columns:
            'neuron_id', 'layer', and 'activation'.
        no_connection_invisible (bool, optional): Whether to make invisible
            neurons that are not connected. Default to True (invisible).
        group_by (dict, optional): A dictionary mapping neuron identifiers to
            group names. Each group will have its own layer. Default to None.
        colour_saturation (float, optional): The saturation of the colours.
            Default to 0.4.
        scene (ngl.Scene, optional): A Neuroglancer scene object from
            nglscenes package. You can read a scene from clipboard like `scene
            = Scene.from_clipboard()`.
        source (list, optional): The source of the Neuroglancer layers.
            Default to None, in which case Full Adult Fly Brain neurons are
            used.
        normalise (str, optional): How to normalise the values. `layer` for
            normalising within each layer; `all` for normalising by the min and
            max value in the entire dataframe. Default to None.
        include_postsynaptic_neuron (bool, optional): Whether to include the
            postsynaptic neuron (column names of `df`) in the visualisation.
            Default to False. Only works if `df_format` is 'wide'.
        diff_colours_per_layer (bool, optional): Whether to use different
            colours for each layer. Default to False.
        colors (list, optional): A list of colours to use for the neurons in
            each layer, if `diff_colours_per_layer` is True. If None, a
            default list of colours is used. Default to None.
        colormap (str, optional): The name of the colormap to use for
            colouring the neurons in every layer, if `diff_colours_per_layer`
            is False. Default to 'viridis'.
        df_format (str, optional): The format of the DataFrame. Either 'wide'
            or 'long'. Default to 'wide'.
        open_here (bool, optional): Whether to display the Neuroglancer scene in the
            notebook. Default to False.
        width (int, optional): The width of the Neuroglancer scene. Default to 1500.
        height (int, optional): The height of the Neuroglancer scene. Default to 800.

    Returns:
        str: A URL to the generated Neuroglancer scene.

    Note:
        The function assumes that the 'scene1' variable is defined in the
            global scope and is an instance of ngl.Scene.
        The function creates separate Neuroglancer layers for each column in
            the DataFrame, using the column name as the layer name.
        The root_ids are colored based on the values in the DataFrame, with a
            color scale ranging from white (minimum value) to a specified
            color (maximum value).
        The function relies on the Neuroglancer library for layer creation and
            scene manipulation.
    """

    try:
        import nglscenes as ngl
    except ImportError as exc:
        raise ImportError(
            "To use this function, please install the package by running 'pip3 install git+https://github.com/schlegelp/nglscenes@main'"
        ) from exc

    # if df is a pandas series, turn into dataframe
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # define a scene if not given:
    if scene == None:
        no_scene_provided = True
        # Initialize a scene
        scene = ngl.Scene()
        scene["layout"] = "3d"
        scene["position"] = [527216.1875, 208847.125, 84774.0625]
        scene["projectionScale"] = 400000
        scene["dimensions"] = {
            "x": [1e-9, "m"],
            "y": [1e-9, "m"],
            "z": [1e-9, "m"],
        }

        # and another for FAFB mesh
        fafb_layer = ngl.SegmentationLayer(
            source="precomputed://https://spine.itanna.io/files/eric/jfrc_mesh_test",
            name="jfrc_mesh_test1",
        )
        fafb_layer["segments"] = ["1"]
        fafb_layer["objectAlpha"] = 0.17
        fafb_layer["selectedAlpha"] = 0.55
        fafb_layer["segmentColors"] = {"1": "#cacdd8"}
        fafb_layer["colorSeed"] = 778769298

        # and the neuropil layer with names
        np_layer = ngl.SegmentationLayer(
            source="precomputed://gs://neuroglancer-fafb-data/elmr-data/FAFBNP.surf/mesh#type=mesh",
            name="neuropil_regions_surface_named",
        )
        np_layer["segments"] = [str(num) for num in range(0, 79)]
        np_layer["visible"] = False
        np_layer["objectAlpha"] = 0.17
    else:
        no_scene_provided = False

    # Define a list of colors optimized for human perception on a dark
    # background
    if colors is None:
        colors = [
            "#ff6b6b",
            "#f06595",
            "#cc5de8",
            "#845ef7",
            "#5c7cfa",
            "#339af0",
            "#22b8cf",
            "#20c997",
            "#51cf66",
            "#94d82d",
            "#fcc419",
            "#4ecdc4",
            "#ffe66d",
            "#7bed9f",
            "#a9def9",
            "#f694c1",
            "#c7f0bd",
            "#ffc5a1",
            "#ff8c94",
            "#ffaaa6",
            "#ffd3b5",
            "#a8e6cf",
            "#a6d0e4",
            "#c1beff",
            "#f5b0cb",
        ]

    # Normalize the values in the DataFrame
    if normalise is not None:
        if normalise == "all":
            df = (df - df.min().min()) / (df.max().max() - df.min().min())

    scene["layout"] = "3d"

    if source is None:
        source = [
            "precomputed://gs://flywire_v141_m783",
            "precomputed://https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/dynann/flytable-info-783-all",
        ]

    if df_format == "wide":
        iterate_over = df.columns
    elif df_format == "long":
        iterate_over = df.layer.unique()
    else:
        raise ValueError("df_format should be either 'wide' or 'long'")

    for i, ite in enumerate(iterate_over):
        cmap: plt.Colormap
        if diff_colours_per_layer:
            color = random.choice(colors)
            cmap = mcl.LinearSegmentedColormap.from_list(
                "custom_cmap", ["white", color]
            )
        else:
            cmap = plt.get_cmap(colormap)

        if df_format == "wide":
            df_group = df[[ite]]
            if no_connection_invisible:
                df_group = df_group[df_group.iloc[:, 0] > 0]
        elif df_format == "long":
            df_group = df[df.layer == ite]
            # make df_group: neuron_id in index, activation in the first column
            df_group.set_index("neuron_id", inplace=True)
            df_group = df_group[["activation"]]

        if group_by is None:
            df_group.loc[:, ["group"]] = ""
        else:
            df_group.loc[:, ["group"]] = df_group.index.map(group_by)

        for grp in df_group.group.unique():
            df_group_grp = df_group[df_group.group == grp]
            df_group_grp = df_group_grp.drop("group", axis=1)

            layer = ngl.SegmentationLayer(source=source, name=str(ite) + " " + str(grp))

            if normalise is not None:
                if normalise == "layer":
                    # if there is only one row, then keep it as is
                    if df_group_grp.shape[0] > 1:
                        df_group_grp = (df_group_grp - df_group_grp.min()) / (
                            df_group_grp.max() - df_group_grp.min()
                        )

            layer["segments"] = list(df_group_grp.index.astype(str))
            if diff_colours_per_layer:
                layer["segmentColors"] = {
                    # trick from Claud to make the colours more saturated
                    str(root_id): mcl.to_hex(
                        cmap(
                            colour_saturation
                            + (1 - colour_saturation) * value.values[0]
                        )
                    )
                    for root_id, value in df_group_grp.iterrows()
                }
            else:
                layer["segmentColors"] = {
                    str(root_id): mcl.to_hex(cmap(value.values[0]))
                    for root_id, value in df_group_grp.iterrows()
                }

            # only the first layer is visible
            if i == 0:
                layer["visible"] = True
            else:
                layer["visible"] = False

            if include_postsynaptic_neuron:
                layer["segments"].append(str(df_group_grp.columns[0]))
                layer["segmentColors"][str(df_group_grp.columns[0])] = "#43A7FF"

            scene.add_layers(layer)

    if no_scene_provided:
        # add FAFB and NP layers at the end
        scene.add_layers(fafb_layer)
        scene.add_layers(np_layer)

    if open_here:
        frame = IFrame(scene.url, width=width, height=height)
        display(frame)
    else:
        return scene.url


def get_activations(
    array,
    global_indices: arrayable,
    idx_map: dict | None = None,
    top_n=None,
    threshold=None,
):
    """
    Get activation for neurons (in rows) in `array` for each time step (in
    columns). Optionally group neurons by `idx_map`, and filter by `top_n` or
    `threshold`.

    Args:
        array (np.ndarray): 2D array of neuron activations, where rows
            represent neurons and columns represent different time steps.
        global_indices (int, list, set, np.ndarray, pd.Series): Array of
            global neuron indices corresponding to keys in `idx_map`.
        idx_map (dict, optional): Mapping from neuron index (`global_indices`)
            to neuron identifier. If not None, and if multiple neurons map to
            the same identifier, the activations are averaged. Defaults to
            None.
        top_n (int, optional): Number of top activations to return for each
            column. If None, all activations above the threshold are returned.
            Defaults to None.
        threshold (float, dict, optional): Minimum activation level to
            consider. If a dictionary is provided, the threshold for each
            column is specified by the column index. Defaults to None.

    Returns:
        dict: A dictionary where each key is a column index and each value is
        a nested dictionary of neuron identifiers and their activations, for
        those activations that are either in the top n, above the threshold,
        or both.

    Note:
        The `global_indices` have to be in the same order as the indices in
        defining the original model.
        If both `n` and `threshold` are provided, the function returns up to
        top n activations that are also above the threshold for each column.
    """

    result = {}
    indices = to_nparray(global_indices)
    if array.shape[0] != len(indices):
        raise ValueError(
            "The length of 'global_indices' should match the number of rows in 'array'."
        )

    global_to_local_map = {global_idx: num for num, global_idx in enumerate(indices)}

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
                    local_indices, np.where(column_values < thresh)[0]
                )

        # Sort the filtered activations
        # these are the local indices
        sorted_indices = (
            np.argsort(column_values)[-top_n:]
            if top_n is not None
            else np.argsort(column_values)
        )

        # get intersection of sorted_indices and local_indices, in the same
        # order as sorted_indices
        selected_local = [index for index in sorted_indices if index in local_indices]
        selected_global = indices[selected_local]

        # Build the result dictionary
        if idx_map is None:
            result[col] = {
                idx: column_values[global_to_local_map[idx]] for idx in selected_global
            }
        else:
            # initialise a dict of empty lists
            result[col] = {idx_map[idx]: [] for idx in selected_global}

            for idx in selected_global:
                result[col][idx_map[idx]].append(
                    column_values[global_to_local_map[idx]]
                )
            # grouped indices by idx_map
            new_indices = result[col].keys()
            # calculate the average
            result[col] = {idx: np.mean(result[col][idx]) for idx in new_indices}

    return result


def plot_layered_paths(
    paths: pd.DataFrame,
    figsize: tuple = (10, 8),
    priority_indices=None,
    sort_dict: Optional[dict] = None,
    sort_by_activation: bool = False,
    fraction: float = 0.03,
    pad: float = 0.02,
    weight_decimals: int = 2,
    neuron_to_sign: dict | None = None,
    sign_color_map: dict = {1: "red", -1: "blue"},
    neuron_to_color: dict | None = None,
    node_activation_min: float | None = None,
    node_activation_max: float | None = None,
    edge_text: bool = True,
    node_text: bool = True,
    highlight_nodes: list[str] = [],
    interactive: bool = False,
    save_plot: bool = False,
    file_name: str = "layered_paths",
    label_pos: float = 0.7,
    default_neuron_color: str = "lightblue",
    default_edge_color: str = "lightgrey",
    node_size: int = 500,
):
    """
    Plots a directed graph of layered paths with optional node coloring based on
    activation values.

    This function creates a visualization of a directed graph with nodes placed in
    layers. Nodes can be optionally colored based on 'pre_activation' and
    'post_activation' columns present in the dataframe. If these columns are missing, a
    default color is used for all nodes. The edges are weighted, and their labels
    represent the weight values.

    Args:
        paths (pandas.DataFrame): A dataframe containing the columns 'pre', 'post',
            'layer', 'weight', and optionally 'pre_activation', 'post_activation',
            'pre_layer', 'post_layer'. Each row represents an edge in the graph. The
            'pre' and 'post' columns refer to the source and target nodes, respectively.
            The 'layer' column is used to place nodes in layers, and 'weight' indicates
            the edge weight. If present, 'pre_activation' and 'post_activation' are used
            to color the nodes based on their activation values.
        figsize (tuple, optional): A tuple indicating the size of the matplotlib figure.
            Defaults to (10, 8).
        priority_indices (list, optional): A list of indices to prioritize when creating
            the layered positions. Nodes with these indices will be placed at the top of
            their respective layers. Defaults to None.
        sort_dict (dict, optional): A dictionary mapping node names to their priority
            values (bigger values are higher in the plot). Nodes will be sorted based on
            these values before plotting. Defaults to None.
        sort_by_activation (bool, optional): A flag to sort the nodes based on their
            activation values (after grouping by priority). Defaults to False.
        fraction (float, optional): The fraction of the figure width to use for the
            colorbar. Defaults to 0.03.
        pad (float, optional): The padding between the colorbar and the plot. Defaults
            to 0.02.
        weight_decimals (int, optional): The number of decimal places to display for
            edge weights. Defaults to 2.
        neuron_to_sign (dict, optional): A dictionary mapping neuron names (as they
            appear in path_df) to their signs (e.g. {'KCg-m': 1, 'Delta7': -1}).
            Can also use a dictionary to map neuron names to their neurotransmitter name.
            Defaults to None.
        sign_color_map (dict, optional): A dictionary used to color edges. Defaults is
            lightgrey but if `neuron_to_sign` is provided, the default is to color edges
            red if the pre-neuron is excitatory, and blue if inhibitory.
            If the `neuron_to_sign` values are neurotransmitter names, then provide
            a dictionary that maps neurotransmitter names to colors.
            If the keys of `sign_color_map` do not match the values of `neuron_to_sign`,
            a warning is printed and the default color is used for the difference.
        neuron_to_color (dict, optional): A dictionary mapping neuron names to colors.
            If not provided, a default color is used for all nodes. The keys should
            match the node names ('pre' and 'post') in paths. The difference is given
            the default color.
        node_activation_min (float, optional): The minimum value for node activation. If
            not provided, the minimum value of node activations is used. Defaults to
            None.
        node_activation_max (float, optional): The maximum value for node activation. If
            not provided, the maximum value of node activations is used. Defaults to
            None.
        edge_text (bool, optional): Whether to display edge weights as text on the plot.
            Defaults to True.
        node_text (bool, optional): Whether to display node names as text on the plot.
            Defaults to True.
        highlight_nodes (list[str], optional): A list of node names to highlight bold in
            the plot. Defaults to an empty list.
        interactive (bool, optional): Whether to create an interactive plot using pyvis.
            Defaults to False. If False, a static matplotlib plot is created.
        save_plot (bool, optional): Whether to save the plot to a file. Defaults to False.
        file_name (str, optional): The name of the file to save the plot. Defaults to
            "layered_paths" in the local directory (.html if interactive and .pdf if
            static).
        label_pos (float, optional): The position of the edge labels. Defaults to 0.7.
            Bigger values move the labels closer to the left of the edge. 
            Only works if `interactive` is False.
        default_neuron_color (str, optional): The default color for nodes if no
            specific color is provided in `neuron_to_color`. Defaults to "lightblue".
        default_edge_color (str, optional): The default color for edges if no specific
            color is provided. Defaults to "lightgrey".
        node_size (int, optional): The size of the nodes in the plot. Defaults to 500.

    Returns:
        None: This function does not return a value. It generates a plot using
            matplotlib or pyvis.

    Note:
        If 'pre_layer' and 'post_layer' columns are not in the dataframe, they will be
            created within the function to uniquely identify the nodes based on their
            'pre'/'post' values and 'layer'.
        The function automatically checks for the presence of 'pre_activation' and
            'post_activation' columns to determine whether to color the nodes based on
            activation values.
        The positions of the nodes are determined by a custom positioning function
            (`create_layered_positions`).
        This function requires the networkx library for graph operations and matplotlib
            for plotting. For interactive plots, it requires the pyvis library (where
            the node label has to be underneath the node).
    """

    if paths.shape[0] == 0:
        raise ValueError("The provided DataFrame is empty.")

    path_df = paths.copy()

    # Create a 'post_layer' column to use as unique identifiers
    if "post_layer" not in path_df.columns:
        path_df["post_layer"] = (
            path_df["post"].astype(str) + "_" + path_df["layer"].astype(str)
        )
    if "pre_layer" not in path_df.columns:
        path_df["pre_layer"] = (
            path_df["pre"].astype(str) + "_" + (path_df.layer - 1).astype(str)
        )

    # Rescale weights to be between 1 and 10
    weight_min = path_df.weight.min()
    weight_max = path_df.weight.max()
    if weight_max == weight_min:
        path_df["weight"] = 1
    else:
        path_df["weight"] = 1 + 9 * (path_df["weight"].values - weight_min) / (
            weight_max - weight_min
        )

    # Create the graph using the new 'post_layer' identifiers
    G = nx.from_pandas_edgelist(
        path_df,
        "pre_layer",
        "post_layer",
        ["weight"],
        create_using=nx.DiGraph(),
    )

    # Labels for nodes
    labels = dict(zip(path_df.post_layer, path_df.post))
    labels.update(dict(zip(path_df.pre_layer, path_df.pre)))

    # Generate positions
    from .path_finding import create_layered_positions

    if sort_by_activation:
        node_activation_dict = dict(zip(path_df.post_layer, path_df.post_activation))
        node_activation_dict.update(
            dict(zip(path_df.pre_layer, path_df.pre_activation))
        )
    else:
        node_activation_dict = {}

    if sort_dict is not None:
        if sort_by_activation:
            max_act = max(node_activation_dict.values())
        else:
            max_act = 0

        # iterating through all layers, because the same cell type can be at multiple layers
        all_sort_dict = {}
        for l in path_df.layer.unique():
            this_path_df = path_df[path_df.layer == l]
            node_to_nodelayer = dict(zip(this_path_df.pre, this_path_df.pre_layer))

            this_sort_dict = {
                node_to_nodelayer[node]: (value + max_act)
                for node, value in sort_dict.items()
                if node in node_to_nodelayer
            }

            # add mapping from post to post_layer
            # doing this separately so that if the same node is in adjacent layers,
            # both are taken into account
            node_to_nodelayer.update(
                dict(zip(this_path_df.post, this_path_df.post_layer))
            )
            this_sort_dict.update(
                {
                    node_to_nodelayer[node]: (value + max_act)
                    for node, value in sort_dict.items()
                    if node in node_to_nodelayer
                }
            )
            all_sort_dict.update(this_sort_dict)
        sort_dict = all_sort_dict
    else:
        sort_dict = {}

    # put two sorting together
    # updating node_activation_dict with sort_dict means that sort_dict takes priority
    node_activation_dict.update(sort_dict)
    if len(node_activation_dict) == 0:
        sort_dict = None
    else:
        sort_dict = node_activation_dict

    positions = create_layered_positions(path_df, priority_indices, sort_dict=sort_dict)

    # Default color for nodes if not provided otherwise
    if neuron_to_color is None:
        neuron_to_color = {label: default_neuron_color for label in labels.values()}
    else:
        # Ensure neuron_to_color has all labels, even if not in the dictionary
        nodediff = set(labels.values()) - set(neuron_to_color.keys())
        if len(nodediff) > 0:
            neuron_to_color.update(dict.fromkeys(nodediff, default_neuron_color))

    # Node colors based on activation values or neuron_to_color
    if ("pre_activation" in path_df.columns) & ("post_activation" in path_df.columns):
        activations = np.concatenate(
            [
                path_df["pre_activation"].values,
                path_df["post_activation"].values,
            ]
        )
        if node_activation_min is None:
            node_activation_min = activations.min()
        if node_activation_max is None:
            node_activation_max = activations.max()
        norm = plt.Normalize(vmin=node_activation_min, vmax=node_activation_max)
        color_map = plt.get_cmap("viridis")
        # Update graph with activation data
        nx.set_node_attributes(
            G,
            dict(zip(path_df.pre_layer, path_df.pre_activation)),
            "activation",
        )
        nx.set_node_attributes(
            G,
            dict(zip(path_df.post_layer, path_df.post_activation)),
            "activation",
        )
        node_colors = [
            color_map(norm(G.nodes[node]["activation"])) for node in G.nodes()
        ]
    else:
        node_colors = []
        for v in G.nodes():
            node_colors.append(neuron_to_color[labels[v]])

    if neuron_to_sign is not None:
        signdiff = set(neuron_to_sign.values()) - set(sign_color_map.keys())
        if len(signdiff) > 0:
            print(
                "Warning: Some values in neuron_to_sign are not in the keys of sign_color_map. Using default color for those edges."
            )
            # this is taken care of below with sign_color_map.get(pre_neuron, default_edge_color)

    # Specify edge colour based on pre-neuron sign, if available
    edge_colors = []
    for u, _ in G.edges():
        pre_neuron = labels[u]
        if neuron_to_sign and pre_neuron in neuron_to_sign:
            edge_colors.append(
                sign_color_map.get(neuron_to_sign[pre_neuron], default_edge_color)
            )
        else:
            edge_colors.append(default_edge_color)

    if interactive:
        try:
            from pyvis.network import Network
        except ImportError as e:
            raise ImportError(
                "Please install pyvis for interactive plots: pip install pyvis"
            ) from e

        net2 = Network(
            directed=True, layout=False, notebook=True, cdn_resources="in_line"
        )
        canvas_h_px = int(figsize[1] * 80)  # tweak the multiplier if needed
        net2.height = f"{canvas_h_px}px"  # explicit px keeps Colab happy
        net2.width = "100%"  # stretch side-to-side
        net2.from_nx(G)

        node_colors_dict = dict(zip(G.nodes(), node_colors))
        edge_colors_dict = dict(zip(G.edges(), edge_colors))
        for v in net2.nodes:
            # coordinates in px
            v["x"], v["y"] = positions.get(v["id"])
            v["x"] = v["x"] * int(figsize[0] * 80)
            v["y"] = v["y"] * canvas_h_px
            v["label"] = labels[v["id"]]
            v["color"] = mpl.colors.rgb2hex(node_colors_dict[v["id"]], keep_alpha=True)
            v["size"] = node_size / 20  # bigger nodes

            v["font"] = {
                "face": (
                    "arial black" if labels[v["id"]] in highlight_nodes else "arial"
                ),
                "size": 26,
            }

        for edge in net2.edges:
            u, v = edge["from"], edge["to"]
            edge["color"] = edge_colors_dict[(u, v)]
            if edge_text:
                edge["label"] = (
                    f"{(weight_min+(weight_max - weight_min)*(edge['width']-1)/9):.{weight_decimals}f}"
                )
                edge["font"] = {"size": 18, "face": "arial"}

        # Set physics options for the network with high spring constant
        # to keep straighter arrows when nodes are moved around
        net2.set_options("""
        var options = {
        "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
            "springConstant": 0.1
            },
            "minVelocity": 0.1
        },
        "nodes": {
            "physics": false
        }
        }
        """)
        net2.toggle_physics(False)
        # net2.show_buttons(filter_=["node", "edge", "physics"])

        if "COLAB_GPU" in os.environ:
            # display of plot in ipynb only works in colab
            html = net2.generate_html(notebook=True)
            if save_plot:
                with open(f"{file_name}.html", "w") as f:
                    f.write(html)

            display(HTML(html))
        else:
            # if running locally, will just open in browser
            if save_plot:
                net2.write_html(str(file_name) + ".html")
                print(f"Interactive graph saved as {file_name}.html")

            net2.show(str(file_name) + ".html", notebook=False)

    else:

        fig, ax = plt.subplots(figsize=figsize)
        nx.draw(
            G,
            pos=positions,
            with_labels=False,
            node_size=node_size,
            node_color=node_colors,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=10,
            width=[G[u][v]["weight"] for u, v in G.edges()],
            edge_color=edge_colors,
            ax=ax,
        )
        if ("pre_activation" in path_df.columns) & (
            "post_activation" in path_df.columns
        ):
            plt.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=color_map),
                ax=ax,
                label="Activation",
                fraction=fraction,
                pad=pad,
            )

        if node_text:
            # label highlighted and normal nodes separately
            bold_nodes = [n for n in G.nodes() if labels[n] in highlight_nodes]
            normal_nodes = [n for n in G.nodes() if labels[n] not in highlight_nodes]

        nx.draw_networkx_labels(
            G,
            pos=positions,
            labels={n: labels[n] for n in normal_nodes},
            font_weight="normal",
            font_size=14,
            ax=ax,
        )
        nx.draw_networkx_labels(
            G,
            pos=positions,
            labels={n: labels[n] for n in bold_nodes},
            font_weight="bold",
            font_size=14,
            ax=ax,
        )

        if edge_text:
            edge_labels = {
                (
                    u,
                    v,
                    # transform back to the original weight values
                ): f"{(weight_min+(weight_max - weight_min)*(G[u][v]['weight']-1)/9):.{weight_decimals}f}"
                for u, v in G.edges()
            }
            nx.draw_networkx_edge_labels(
                G,
                pos=positions,
                edge_labels=edge_labels,
                label_pos=label_pos,
                font_size=14,
                rotate=False,
                ax=ax,
            )

        ax.set_ylim(0, 1)
        if save_plot:
            fig.savefig(file_name + ".pdf")
            print(f"Graph saved as {file_name}.pdf")
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
    The DataFrame should contain columns 'pre' and 'post', and optionally
    'conditional', which contain indices of the connectivity weights in model.
    The weights are modified proportional to:
    1. The similarity of pre and post activations (i.e. the sum of element-
    wise multiplication of activations across time), and
    2. The coefficient provided.
    3. The similarity of conditional activations (if provided). `offset`
    specifies the time lag between the conditional neurons and the pre and
    post neurons.

    The 'mode' column should specify whether the weight change is ltp or ltd.

    Args:
        model (torch.nn.Module): The model containing the weights to change.
        df (pd.DataFrame): A DataFrame containing the columns 'pre' and 'post'.
        mode (str): The mode of weight change, either 'ltp' or 'ltd'.
            coefficient (float): The coefficient to multiply the weight change
            by. Can be thought of as the 'strength of plasticity'. Default to
            0.1.
        offset (int): Offset between the conditional neurons and the pre and
            post neurons. Default to 0. -1 means the conditional neurons'
            activity precede the pre and post neurons, and the activity of
            conditional neurons at time t is related to the activity of pre
            and post neurons at time t+1.
        normalise (bool): Whether to normalise the weights after changing them,
            such that absolute weights to postsynaptic neuron sums to 1.
            Default to True.

    Returns:
        None: This function modifies the model weights in place.
    """

    if "conditional" in df.columns:
        # first group by unique combinations of pre and post
        # Create a pivot table to get the mean conditional activations for
        # each unique pre and post combination
        pivot_table = df.pivot_table(
            index=["pre", "post"], values="conditional", aggfunc=list
        ).reset_index()
        # get the average activation of the conditionals across timepoints
        conditional_acts = torch.stack(
            [
                torch.mean(model.activations[torch.tensor(cond)], dim=0)
                for cond in pivot_table.conditional.values
            ]
        )
        # then multiply with the pre and post activations, and the coefficeint
        if offset == 0:
            weight_change = (
                torch.mean(
                    # the activation of the conditional neurons relates
                    # positively to the plasticity
                    model.activations[pivot_table.pre, :]
                    * model.activations[pivot_table.post, :]
                    * conditional_acts,
                    dim=1,
                )
                * coefficient
            )
        else:
            weight_change = (
                torch.mean(
                    # the activation of the conditional neurons relates
                    # positively to the plasticity
                    model.activations[pivot_table.pre, -offset:]
                    * model.activations[pivot_table.post, -offset:]
                    * conditional_acts[:, :offset],
                    dim=1,
                )
                * coefficient
            )

    else:
        weight_change = (
            torch.mean(
                model.activations[df.pre, :] * model.activations[df.post, :],
                dim=1,
            )
            * coefficient
        )
    # shape of weight_change is the same as the number of rows in the df

    # take out the weights to change
    # NOTE: all_weights have pre in the columns, post in the rows
    weights_subset = model.all_weights[df.post, df.pre]
    # change the weights: keep the signs separate, strengthen (ltp) / weaken
    # (ltd) the connection by using the absolute values
    if mode == "ltp":
        # strengthen connectivity
        weights_subset_abs = torch.abs(weights_subset) + weight_change.t()
    elif mode == "ltd":
        # weaken connectivity
        weights_subset_abs = torch.abs(weights_subset) - weight_change.t()
    else:
        raise ValueError("The mode should be either 'ltp' or 'ltd'.")

    # add the sign back
    weights_subset = torch.sign(weights_subset) * torch.clamp(weights_subset_abs, 0, 1)
    # finally modify in the big weights matrix.
    model.all_weights[df.post, df.pre] = weights_subset

    if normalise:
        # make sure all the rows modified (df.post) have their absolute values
        # sum up to 1
        model.all_weights[df.post, :] = (
            model.all_weights[df.post, :]
            / model.all_weights[df.post, :].abs().sum(dim=1)[:, None]
        )


def count_keys_per_value(d):
    """
    Count the number of keys per value in a dictionary.

    Args:
        d (dict): The input dictionary.

    Returns:
        dict: A dictionary where each key is a value from the input dictionary,
        and each value is the number of keys that map to that value.
    """
    value_counts = defaultdict(int)
    for value in d.values():
        value_counts[value] += 1
    return dict(value_counts)


def sc_connectivity_summary(df, inidx_map=None, outidx_map=None):
    """
    Single cell connectivity summary. For each group based on inidx_map,
    select the neuron with the strongest input to each postsynaptic group, and
    give value by the total weight from that presynaptic group for each average
    post-synaptic neuron. The output could be fed into `get_ngl_link()`. The
    idea of this function came from Dr Alexandra Fragniere.

    Args:
        df (pd.DataFrame): A DataFrame with pre in the row indices, post in
            the column names, and weights as values.
        inidx_map (dict, optional): A mapping from the presynaptic indices to
            group identifiers. If None, the presynaptic indices themselves are
            used as group identifiers. Defaults to None.
        outidx_map (dict, optional): A mapping from the postsynaptic indices
            to group identifiers. If None, the postsynaptic indices are used
            as group identifiers. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame, with values in `outidx_map` as column names.
            Row indices are a subset of the original row indices: the top input
            in each group. The values in the dataframe are the sum of the
            weights from each `inidx_map` group to an average member of the
            `outidx_map` group.

    """

    if inidx_map is None:
        inidx_map = {idx: str(idx) for idx in df.index}
    if outidx_map is None:
        outidx_map = {idx: str(idx) for idx in df.columns}

    # calculate the average first for the post_grp
    dft = df.T
    dft.loc[:, ["group"]] = dft.index.map(outidx_map)
    df = dft.groupby("group").mean().T

    df.loc[:, ["pre_grp"]] = df.index.map(inidx_map)
    # get values in the output of the function
    values = (
        df.groupby("pre_grp")
        .sum()
        .reset_index()
        .melt(id_vars=["pre_grp"], var_name="post", value_name="value")
    )
    # get indices (top input in each group) in the output of the function
    indices = (
        df.groupby("pre_grp")
        .idxmax()
        .reset_index()
        .melt(id_vars=["pre_grp"], var_name="post", value_name="index_value")
    )
    # put together, and make wide
    merged_df = pd.merge(indices, values, on=["pre_grp", "post"])
    merged_df = merged_df.pivot(index="index_value", columns="post", values="value")
    merged_df.fillna(0, inplace=True)

    return merged_df


def check_consecutive_layers(df):
    """
    Check if the layers in the DataFrame are consecutive.

    Args:
        df (pd.DataFrame): A DataFrame containing the column 'layer' (integer).

    Returns:
        bool: True if the layers are consecutive, False otherwise
    """

    all_layers = sorted(df["layer"].unique())
    consecutive_layers = all(
        all_layers[i] + 1 == all_layers[i + 1] for i in range(len(all_layers) - 1)
    )
    return consecutive_layers


def group_edge_by(edgelist, group_dict):
    """
    Group the edges in an edgelist by a dictionary.

    Args:
        edgelist (pd.DataFrame): A DataFrame with columns 'pre', 'post' and
            'weight' representing the edges.
        by (dict): A dictionary mapping the values in the 'pre' and 'post'
            columns of `edgelist` to the groups.

    Returns:
        pd.DataFrame: A DataFrame with columns 'pre' and 'post' representing
            the edges, and 'group_pre' and 'group_post' representing the
            groups.
    """
    edgelist.loc[:, ["group_pre"]] = edgelist.pre.map(group_dict)
    edgelist.loc[:, ["group_post"]] = edgelist.post.map(group_dict)
    # group by pre, sum; group by post, average
    # e.g. if group is cell type: the input proportion from type A to an
    # average neuron in type B
    edgelist = (
        edgelist.groupby(["group_pre", "group_post", "post"]).weight.sum().reset_index()
    )  # sum across pre
    edgelist = (
        edgelist.groupby(["group_pre", "group_post"]).weight.mean().reset_index()
    )  # average across post
    edgelist.columns = ["pre", "post", "weight"]

    return edgelist


def display_df(df, cmap="Blues"):
    """
    Thin wrapper around the `display` function to display a DataFrame with a
        background gradient.

    Args:
        df (pd.DataFrame): The DataFrame to display.
        cmap (str, optional): The name of the colormap to use for the
            background gradient. Defaults to 'Blues'.

    Returns:
        None: This function displays the DataFrame using the `display` function
    """
    result_dp = df.style.background_gradient(
        cmap=cmap, vmin=df.min().min(), vmax=df.max().max()
    )
    display(result_dp)


def compare_connectivity(
    m1,
    m2,
    inidx1: arrayable,
    outidx1: arrayable,
    inidx2: arrayable,
    outidx2: arrayable,
    g1_pre: dict | None = None,
    g1_post: dict | None = None,
    g2_pre: dict | None = None,
    g2_post: dict | None = None,
    suffices: List[str] = ["_l", "_2"],
    display: bool = True,
    threshold: float = 0,
    sort_within: str = "column",
    sort_by: str | None = None,
    threshold_axis: str = "row",
    merge: str = "outer",
    suffix_in: str = "column",
):
    """
    Compare the connectivity between two matrices.

    Args:
        m1 (scipy.sparse matrix or numpy.ndarray): The first connectivity matrix.
        m2 (scipy.sparse matrix or numpy.ndarray): The second connectivity matrix.
        inidx1 (int, float, list, set, numpy.ndarray, or pandas.Series): The indices of
            the presynaptic neurons in the first matrix.
        outidx1 (int, float, list, set, numpy.ndarray, or pandas.Series): The indices of
            the postsynaptic neurons in the first matrix.
        inidx2 (int, float, list, set, numpy.ndarray, or pandas.Series): The indices of
            the presynaptic neurons in the second matrix.
        outidx2 (int, float, list, set, numpy.ndarray, or pandas.Series): The indices of
            the postsynaptic neurons in the second matrix.
        g1_pre (dict, optional): A dictionary mapping the presynaptic indices to groups
            in the first matrix. Defaults to None.
        g1_post (dict, optional): A dictionary mapping the postsynaptic indices to
            groups in the first matrix. Defaults to None.
        g2_pre (dict, optional): A dictionary mapping the presynaptic indices to groups
            in the second matrix. Defaults to None.
        g2_post (dict, optional): A dictionary mapping the postsynaptic indices to
            groups in the second matrix. Defaults to None.
        suffices (list, optional): A list of suffixes to append to the column names of
            the two matrices. Defaults to ['_l', '_2'].
        display (bool, optional): Whether to display the resulting DataFrame in colour
            gradient. Defaults to True.
        threshold (float, optional): The threshold below which to remove values.
            Defaults to 0.
        sort_within (str, optional): Whether to sort the DataFrame with 'column' (across
            rows) or 'row' (across columns). Defaults to 'column'.
        sort_by (str, optional): The column to sort by. Defaults to None.
        threshold_axis (str, optional): The axis to apply the threshold to. Defaults to
        'row' (removing entire rows if no value exceeds display_threshold).
        merge (str, optional): The type of merge to perform on the two DataFrames.
            When suffix_in is 'row' (separating input by suffices), the merge is
            performed on the columns of the two DataFrames. When suffix_in is 'column'
            (separating target by suffices), the merge is performed on the rows of the
            two DataFrames. Possible values are 'inner', 'outer', 'left', and 'right'.
            Defaults to 'outer'.
        suffix_in (str, optional): Whether to put the suffixes on the rows or columns.
            Possible values are 'row' and 'column'. Defaults to 'column'.

    Returns:
        pd.DataFrame: A DataFrame containing the connectivity values from the
            two matrices, with columns suffixed by the values in `suffices`.

    """

    from .compress_paths import result_summary

    df1 = result_summary(
        m1,
        inidx1,
        outidx1,
        g1_pre,
        g1_post,
        display_output=False,
        display_threshold=threshold,
        threshold_axis=threshold_axis,
    )
    df2 = result_summary(
        m2,
        inidx2,
        outidx2,
        g2_pre,
        g2_post,
        display_output=False,
        display_threshold=threshold,
        threshold_axis=threshold_axis,
    )

    if suffix_in not in ["row", "column"]:
        raise ValueError("suffix_in should be either 'row' or 'column'.")

    if suffix_in == "row":
        # transpose the dataframes to have the same orientation
        df1 = df1.T
        df2 = df2.T

    df1.columns = [col + suffices[0] for col in df1.columns]
    df2.columns = [col + suffices[1] for col in df2.columns]

    # Join the dataframes on their index (row names), keeping all rows
    df_merged = df1.merge(df2, left_index=True, right_index=True, how=merge)
    df_merged = df_merged.fillna(0)
    # order the columns alphabetically
    # Sort columns alphabetically
    df_merged = df_merged.reindex(sorted(df_merged.columns), axis=1)

    if suffix_in == "row":
        df_merged = df_merged.T

    if sort_within == "column":
        if sort_by is None:
            sort_by = df_merged.columns[0]
        df_merged = df_merged.sort_values(sort_by, ascending=False)
    elif sort_within == "row":
        if sort_by is None:
            sort_by = df_merged.index[0]
        df_merged = df_merged.sort_values(sort_by, ascending=False, axis=1)
    else:
        raise ValueError("sort_within should be either 'column' or 'row'.")

    if display:
        display_df(df_merged)

    return df_merged


def make_grid_inputs(
    v1: arrayable,  # see above
    v2: arrayable,
    num_layers: int,
    grid_size: int = 10,
    timepoints: int | list = 0,
    device=None,  # type?
    cap: bool = True,
    cap_value: float = 1.0,
) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
    """
    Make a batch of input using combinations of v1 and v2 at different
    strengths (0 to 1).

    Args:
        v1: The first input vector.
        v2: The second input vector. Its length should match that of `v1`.
        num_layers (int): The number of layers in the model.
        grid_size (int, optional): The number of points in the grid. Defaults
            to 10.
        timepoints (int|list, optional): The timepoints at which combinations
            of v1 and v2 are used. Defaults to 0 (the first timepoint).
        device (str, optional): The device to create the inputs on. If None,
            if GPU is available, the inputs are created on the GPU. Otherwise
            CPU.
        cap (bool, optional): Whether to cap the values at `cap_value`. This is
            to prevent two inputs adding up to make a neuron activation > 1.
            Defaults to True.
        cap_value (float, optional): The value to cap the inputs at. Defaults
            to 1.0.

    Returns:
        torch.Tensor: A tensor of shape (grid_size**2, len(v1), num_layers).
        list: A list of grid coordinates.
    """
    # first convert to np array
    v1 = to_nparray(v1, unique=False)
    v2 = to_nparray(v2, unique=False)

    # error if v1 and v2 have more than 1 dimension
    if v1.ndim > 1 or v2.ndim > 1:
        raise ValueError("v1 and v2 should be 1D arrays.")

    # check if length of v1 and v2 are the same
    if len(v1) != len(v2):
        raise ValueError("The length of v1 and v2 should be the same.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sc1 = np.linspace(0, 1, grid_size)
    sc2 = np.linspace(0, 1, grid_size)
    sc1_grid, sc2_grid = np.meshgrid(sc1, sc2)

    inputs = np.zeros((grid_size**2, len(v1), num_layers), dtype=np.float32)
    for i, (s1, s2) in enumerate(zip(sc1_grid.ravel(), sc2_grid.ravel())):
        inputs[i, :, timepoints] = v1 * s1 + v2 * s2

    # cap the values
    if cap:
        inputs[inputs > cap_value] = cap_value

    # Convert to torch tensor
    inputs_tensor = torch.from_numpy(inputs).to(device)

    # Create grid coordinates for reference
    grid_coords = list(zip(sc1_grid.ravel(), sc2_grid.ravel()))

    return inputs_tensor, grid_coords


def output_grid_data(
    grid_outputs: torch.Tensor | npt.NDArray,
    grid_coords: List[Tuple[float, float]],
    selected_index: arrayable,
) -> List[npt.NDArray]:
    """
    Extract the output grid data for selected indices.

    Args:
        grid_outputs (torch.Tensor|np.ndarray): The output grid tensor of
            shape (grid_size**2, num_neurons, num_layers).
        grid_coords (list): A list of grid coordinates. Best from function
            `make_grid_inputs()`.
        selected_index (arrayable): The index or indices to extract.

    Returns:
        list: A list of length `num_layers` of mean values with shape
            (grid_size, grid_size) for the selected indices.
    """

    selected_index = to_nparray(selected_index)
    heatmaps = []
    # if grid_outputs is tensor, convert to numpy
    if isinstance(grid_outputs, torch.Tensor):
        grid_outputs = grid_outputs.cpu().numpy()

    for i in range(grid_outputs.shape[2]):
        # Extract the selected index values
        values = grid_outputs[:, selected_index, i].mean(axis=1)

        # Reshape values to match the grid
        grid_size = int(np.sqrt(len(grid_coords)))
        values_grid = values.reshape((grid_size, grid_size))
        heatmaps.append(values_grid)

    return heatmaps


def plot_output_grid(
    grid_outputs: torch.Tensor | npt.NDArray,
    grid_coords: List[Tuple[float, float]],
    selected_index: arrayable,
    *,  # see note
    figsize: tuple = (10, 8),
    cmap: str = "viridis",
    xlab: str = "v1 Activation",
    ylab: str = "v2 Activation",
    title: str | None = None,
    show_values: bool = False,
    fmt: str = ".2f",
):
    """
    Plot the output grid for selected indices.

    Args:
        grid_outputs (torch.Tensor|np.ndarray): The output grid tensor of shape
            (grid_size**2, num_neurons, num_layers).
        grid_coords (list): A list of grid coordinates. Best from function
            `make_grid_inputs()`.
        selected_index (int|list): The index or indices to plot.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 8).
        cmap (str, optional): The colormap to use. Defaults to 'viridis'.
        xlab (str, optional): The x-axis label. Defaults to 'v1 Activation'.
        ylab (str, optional): The y-axis label. Defaults to 'v2 Activation'.
        title (str, optional): The title of the plot. Defaults to None.
        show_values (bool, optional): Whether to display the values on the heatmap.
            Defaults to False.
        fmt (str, optional): The format of the values. Defaults to '.2f'.

    Returns:
        None: This function displays the plot.
    """
    selected_index = to_nparray(selected_index)

    # if grid_outputs is tensor, convert to numpy
    if isinstance(grid_outputs, torch.Tensor):
        grid_outputs = grid_outputs.detach().cpu().numpy()

    if title is None:
        title = f"Output Grid for {selected_index}"

    def plot_heatmap(index, xlab, ylab, title):
        # # Create the plot
        plt.figure(figsize=figsize)
        # use seaborn heatmap
        if show_values:
            sns.heatmap(heatmaps[index - 1], cmap=cmap, annot=True, fmt=fmt)
        else:
            sns.heatmap(heatmaps[index - 1], cmap=cmap, fmt=fmt)
        plt.gca().invert_yaxis()

        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.show()

    heatmaps = output_grid_data(grid_outputs, grid_coords, selected_index)

    slider = widgets.IntSlider(
        value=1,
        min=1,
        max=len(heatmaps),
        step=1,
        description="Timepoint",
        continuous_update=True,
    )

    # Link the slider to the plotting function
    display(
        widgets.interactive(
            plot_heatmap, index=slider, xlab=xlab, ylab=ylab, title=title
        )
    )


def pytorch_sparse_to_scipy(sparse_tensor, scipy_format="csr"):
    """
    Convert a PyTorch sparse tensor to a SciPy sparse matrix.

    Args:
        sparse_tensor (torch.sparse.Tensor): PyTorch sparse tensor in COO format
        scipy_format (str): Desired format for the SciPy sparse matrix. Must be one of
            'csr', 'csc', 'coo'.

    Returns:
        scipy.sparse.csr_matrix: SciPy CSR sparse matrix
    """
    assert sparse_tensor.is_sparse, "Input tensor must be sparse"

    # Extract indices and values
    indices = sparse_tensor._indices().cpu().numpy()
    values = sparse_tensor._values().cpu().numpy()
    shape = sparse_tensor.shape

    # Create scipy COO matrix
    coo = coo_matrix((values, (indices[0], indices[1])), shape=shape)

    # Convert to desired format
    if scipy_format == "csr":
        return coo.tocsr()
    elif scipy_format == "csc":
        return coo.tocsc()
    elif scipy_format == "coo":
        return coo
    else:
        raise ValueError("Invalid scipy_format. Must be one of 'csr', 'csc', 'coo'.")


def scipy_sparse_to_pytorch(
    scipy_sparse,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Convert a SciPy sparse matrix to a PyTorch sparse tensor.

    Args:
        scipy_sparse (scipy.sparse.spmatrix): SciPy sparse matrix
        device (torch.device): Device to create the PyTorch sparse tensor on. Defaults
            to GPU if available, otherwise CPU.

    Returns:
        torch.sparse.Tensor: PyTorch sparse tensor
    """
    # Convert to COO format if not already
    coo = scipy_sparse.tocoo()

    # Create indices and values
    indices = torch.LongTensor(np.vstack((coo.row, coo.col))).to(device)
    values = torch.FloatTensor(coo.data).to(device)
    shape = torch.Size(coo.shape)

    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def plot_activity_by_column(
    activity,
    idx_to_group: dict,
    idx_to_column: dict,
    selected_group: arrayable,
    plot_type: str = "line",
    model_input: Optional[Union[np.ndarray, torch.Tensor]] = None,
    sensory_indices: Optional[arrayable] = None,
    figsize: tuple = (800, 600),
    global_min: Optional[float] = None,
    global_max: Optional[float] = None,
):
    """
    Take output from `activity_by_column()` and plot the activity per neuron group, per
    time step, per column. The x axis is the normalised column, normalised within each
    neuron group.

    Args:
        activity (torch.Tensor | numpy.ndarray): The activity of the model. Shape should
            be (num_neurons, num_timepoints).
        idx_to_group (dict): A dictionary mapping indices from the model to the groups
            of interest (e.g. cell type). max(idx_to_group.keys()) should be equal to
            number of units in the model.
        idx_to_column (dict): A dictionary mapping indices from the model to the columns
            of interest (e.g. column in the central complex).
        selected_group (arrayable): The groups to select from the activity. This should
            be a list of groups that are present in `idx_to_group.values()`.
        plot_type (str, optional): The type of plot to create. Can be either 'scatter'
            or 'line'. Defaults to 'line'.
        model_input (numpy.ndarray | torch.Tensor, optional): The input to the model.
            Shape should be (num_neurons, num_timepoints). If provided, the first
            timepoint of model_input is also included in the output dataframe
            (time_step = 0). Defaults to None.
        sensory_indices (arrayable, optional): The indices of sensory neurons.
            If provided, it should be a list of indices that are present in
            `idx_to_group`. If provided, it must also be provided with `model_input`.
            Defaults to None.
        figsize (tuple, optional): The size of the figure in pixels. Defaults to (800,
            600).
        global_min (float, optional): The minimum value for the y-axis. If None, the
            minimum value is set to the smaller of 0, and the minimum activation value
            across all groups and time steps. Defaults to None.
        global_max (float, optional): The maximum value for the y-axis. If None, the
            maximum value is set to the bigger of 1, and the maximum activation value
            across all groups and time steps. Defaults to None.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object.
    """
    from .activation_maximisation import activity_by_column

    column_acts = activity_by_column(
        activity,
        idx_to_group,
        idx_to_column,
        selected_group,
        model_input=model_input,
        sensory_indices=sensory_indices,
    )

    # Get unique values
    unique_groups = column_acts["cell_group"].unique()
    unique_times = sorted(column_acts["time_step"].unique())

    # Create color mapping
    colors = plotly.colors.qualitative.Plotly[: len(unique_groups)]
    color_map = {group: colors[i] for i, group in enumerate(unique_groups)}

    # Create figure
    fig = go.Figure()

    # Add traces for each time step and group combination
    for time_step in unique_times:
        time_data = column_acts[column_acts["time_step"] == time_step]

        for group in unique_groups:
            group_data = time_data[time_data["cell_group"] == group]

            if plot_type == "scatter":
                fig.add_trace(
                    go.Scatter(
                        x=group_data["normalised_column"],
                        y=group_data["activation"],
                        mode="markers",
                        name=group,
                        legendgroup=group,  # Group legend entries
                        showlegend=True,
                        visible=bool(
                            time_step == unique_times[0]
                        ),  # Convert to Python bool
                        marker=dict(color=color_map[group], size=8),
                    )
                )
            else:  # line
                fig.add_trace(
                    go.Scatter(
                        x=group_data["normalised_column"],
                        y=group_data["activation"],
                        mode="lines+markers",
                        name=group,
                        legendgroup=group,
                        showlegend=True,
                        visible=bool(
                            time_step == unique_times[0]
                        ),  # Convert to Python bool
                        line=dict(color=color_map[group]),
                        marker=dict(size=6),
                    )
                )

    # Create slider steps
    steps = []
    for i, time_step in enumerate(unique_times):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": f"Neural Activity - Time Step: {time_step}"},
            ],
            label=str(time_step),
        )

        # Make traces visible for current time step
        start_idx = i * len(unique_groups)
        end_idx = start_idx + len(unique_groups)
        for j in range(start_idx, end_idx):
            step["args"][0]["visible"][j] = True

        steps.append(step)

    # Add slider
    sliders = [
        dict(
            active=0, currentvalue={"prefix": "Time Step: "}, pad={"t": 50}, steps=steps
        )
    ]

    fig.update_layout(
        sliders=sliders,
        title=f"Neural Activity - Time Step: {unique_times[0]}",
        xaxis_title="Normalised Column",
        yaxis_title="Activation Value",
        width=figsize[0],
        height=figsize[1],
    )
    if global_min is None:
        global_min = min(0, column_acts["activation"].min())
    if global_max is None:
        global_max = max(1, column_acts["activation"].max())
    fig.update_yaxes(range=[global_min, global_max])

    return fig
