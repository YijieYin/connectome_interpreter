# Standard library imports
import itertools
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple, Optional
from collections import defaultdict

# Third-party package imports
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import issparse, spmatrix
from tqdm import tqdm

from .utils import (
    arrayable,
    check_consecutive_layers,
    count_keys_per_value,
    to_nparray,
)


def find_path_once(
    inprop,
    steps_cpu,
    inidx: arrayable,
    outidx: arrayable,
    target_layer_number,
    top_n=-1,
    threshold=0,
):
    """
    Finds the path once between input and output, of distance
    target_layer_number, returning indices of neurons in the previous layer
    that connect the input with the output. This works by taking the top_n
    direct upstream partners of the outidx neurons, and intersect those with
    neurons 'effectively' connected (through steps_cpu) to the inidx neurons.

    Args:
      inprop (scipy.sparse.csc_matrix): The connectivity matrix in
        Compressed Sparse Column format.
      steps_cpu (list): A list of compressed connectivity matrices: one matrix
        for each compressed path length.
      inidx (int, float, list, set, numpy.ndarray, or pandas.Series): The
        input neuron index/indices.
      outidx (int, float, list, set, numpy.ndarray, or pandas.Series): The
        output neuron index/indices.
      target_layer_number (int): The target layer number to examine. Must be
        >= 1. When target_layer_number = 1, we are looking at the direct
        synaptic connectivity.
      top_n (int, optional): The number of top connections to consider based
        on direct connectivity from inprop_csc. If top_n = -1, all connections
        are considered.
      threshold (float, optional): The threshold of the direct connectivity
        from inidx to an average outidx.

    Returns:
      np.ndarray: An array of neuron indices in the previous layer that have
        significant connectivity, connecting between the `inidx` and `outidx`.
    """

    inprop_csc = inprop.copy()
    inprop_csc.data = np.abs(inprop.data)

    # make sure they are integers
    inidx = [int(i) for i in inidx]
    outidx = [int(i) for i in outidx]

    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    if target_layer_number == 1:
        # if the target layer is 1, we are looking at the direct synaptic
        # connectivity
        # so we just need to find the indices of the non-zero values in the
        # inprop_csc matrix that correspond to the outidx, and intersect
        # those with the inidx we are interested in.
        colidx = inidx
    else:
        # first get the neurons that effectively connect to inidx, at layer
        # number target_layer_number - 1.
        # for example, in the ORN->PN->KC case, target_layer_number = 2 for KCs.
        # top_n_row_indices are indices of the PNs that connect to the KCs. So
        # in this case we should use direct connectivity from PNs to get the
        # ORNs.
        # so when target_layer_number == 2, we should use steps_cpu[0] (direct
        # connectivity), that is, target_layer_number-2:
        # subtract 1 for getting top_n_row_indices below which is going one
        # step upstream; and another for 0-based indexing in steps_cpu.
        # the next line gets the targets that receive non-zero compressed
        # input from inidx
        # when target_layer_number >= 2.
        # .nonzero() returns row, column of the nonzero values.
        colidx = steps_cpu[target_layer_number - 2][inidx, :].nonzero()[1]

    # then go back one step from the outidx, based on the direct connectivity
    # matrix (inprop)
    us = inprop_csc[:, outidx].nonzero()
    # intersect the non-zero upstream neurons with those effectively connected
    # across layers (colidx)
    # these are still row indices of inprop
    intersect = np.intersect1d(us[0], colidx)

    # direct connectivity between target_layer_number-1 and target_layer_number
    submatrix = inprop_csc[intersect, :][:, outidx]
    # in case outidx is more than one index, calculate the average of each row
    # in the submatrix
    # same length as len(intersect)
    row_averages = np.array(submatrix.mean(axis=1)).flatten()

    # thresholding
    # these are indices of row_averages, which can also be used to index
    # row_averages
    thresholded_indices = np.where(row_averages >= threshold)[0]
    thresholded_row_averages = row_averages[thresholded_indices]
    # thresholded_intersect are a subset of of intersect, which are indices of
    # inprop_csc
    # intersect has the same length as row_averages
    thresholded_intersect = intersect[thresholded_indices]

    # Find the indices (of thresholded_row_averages) of the top n averages
    if top_n == -1:
        top_n_indices = np.argsort(thresholded_row_averages)
    else:
        top_n_indices = np.argsort(thresholded_row_averages)[-top_n:]
    # Get the original row indices corresponding to these top n averages
    top_n_row_indices = thresholded_intersect[top_n_indices]

    # make the edgelist
    submatrix = inprop_csc[top_n_row_indices, :][:, outidx]

    # Convert submatrix to COO format to efficiently find non-zero elements
    coo_submatrix = submatrix.tocoo()

    # Create DataFrame directly from COO format data
    df = pd.DataFrame(
        {
            # Map back to original indices
            "pre": top_n_row_indices[coo_submatrix.row],
            # Map back to original output indices
            "post": outidx[coo_submatrix.col],
            "weight": coo_submatrix.data,
        }
    )

    return df


def find_paths_of_length(
    edgelist: Union[spmatrix, pd.DataFrame],
    inidx: arrayable,
    outidx: arrayable,
    target_layer_number: int,
):
    """
    Finds the path of length target_layer_number between inidx and outidx, returning the
    edgelist in a DataFrame, including the pre and post indices, the layer (direct
    connections from inidx: layer = 1), and the weight of the direct connection between
    pre and post.

    Args:
      edgelist (Union[spmatrix, pd.DataFrame]): The edgelist of the entire graph. If a
        DataFrame, it must contain columns "pre", "post", and "weight". If a sparse
        matrix, the pre needs to be in the rows.
      inidx (int, float, list, set, numpy.ndarray, or pandas.Series): The source indices.
      outidx (int, float, list, set, numpy.ndarray, or pandas.Series): The target
        indices.
      target_layer_number (int): The target layer number to examine. Must be >= 1. When
        target_layer_number = 1, we are looking at the direct synaptic connectivity.

    Returns:
      pd.DataFrame: A DataFrame containing the path data, including the pre-synaptic and
        post-synaptic neuron indices, the layer (direct connections from inidx: layer =
        1), and the weight (input proportion of the postsynaptic neuron) of the direct
        connection between pre and post. If no path is found, returns None.
    """

    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    assert len(inidx) > 0, "inidx must not be empty"
    assert len(outidx) > 0, "outidx must not be empty"

    if isinstance(edgelist, pd.DataFrame):
        # if edgelist is a DataFrame, convert it to a sparse matrix
        if not all(col in edgelist.columns for col in ["pre", "post", "weight"]):
            raise ValueError(
                "edgelist DataFrame must contain 'pre', 'post', and 'weight' columns."
            )
    elif isinstance(edgelist, spmatrix):
        edgelist = edgelist.tocoo()
        edgelist = pd.DataFrame(
            data={
                "pre": edgelist.row,
                "post": edgelist.col,
                "weight": edgelist.data,
            }
        )
    else:
        raise TypeError("edgelist must be a pandas DataFrame or a scipy sparse matrix.")

    assert target_layer_number >= 1, "target_layer_number must be >= 1"
    if target_layer_number == 1:
        # if target_layer_number is 1, we just return the direct connections
        el = edgelist[
            (edgelist["pre"].isin(inidx)) & (edgelist["post"].isin(outidx))
        ].copy()
        if el.empty:
            return None
        el.loc[:, ["layer"]] = 1
        return el

    # first find the middle layer:
    if target_layer_number % 2 == 0:
        # when target_layer_number == 2, the middle layer is 1
        middle_layer = target_layer_number // 2
    else:
        # when target_layer_number == 3, the middle layer is 2
        middle_layer = (target_layer_number + 1) // 2

    # list of empty lists, of length target_layer_number+1
    layer_indices = [[] for _ in range(target_layer_number + 1)]
    # first one is inidx
    layer_indices[0] = inidx
    # last one is outidx
    layer_indices[-1] = outidx

    # first go from inidx to middle layer
    for layer in range(1, middle_layer + 1):
        layer_indices[layer] = list(
            edgelist[edgelist["pre"].isin(layer_indices[layer - 1])]["post"].unique()
        )
        if len(layer_indices[layer]) == 0:
            print(f"No neurons found in layer {layer}. Returning None.")
            return None
    # then go from outidx to middle layer
    # stopping at the layer after middle_layer
    # when target_layer_number is 2 or 3, this is skipped
    # when target_layer_number == 4:
    for layer in range(target_layer_number - 1, middle_layer, -1):  # layer is 3
        layer_indices[layer] = list(
            edgelist[edgelist["post"].isin(layer_indices[layer + 1])]["pre"].unique()
        )
        if len(layer_indices[layer]) == 0:
            print(f"No neurons found in layer {layer}. Returning None.")
            return None

    # link middle layer with the one after
    el = edgelist[
        (edgelist["pre"].isin(layer_indices[middle_layer]))
        & (edgelist["post"].isin(layer_indices[middle_layer + 1]))
    ]
    layer_indices[middle_layer] = el["pre"].unique()
    layer_indices[middle_layer + 1] = el["post"].unique()
    if (
        len(layer_indices[middle_layer]) == 0
        or len(layer_indices[middle_layer + 1]) == 0
    ):
        print(
            f"No neurons found in layer {middle_layer} or {middle_layer + 1}. Returning None."
        )
        return None

    # now go from middle layer to the first layer
    for layer in range(middle_layer - 1, -1, -1):
        el = edgelist[
            edgelist["pre"].isin(layer_indices[layer])
            & edgelist["post"].isin(layer_indices[layer + 1])
        ]
        layer_indices[layer] = el["pre"].unique()
        if len(layer_indices[layer]) == 0:
            print(f"No neurons found in layer {layer}. Returning None.")
            return None
    # now go from middle layer to the last layer
    for layer in range(middle_layer + 1, target_layer_number):
        el = edgelist[
            edgelist["pre"].isin(layer_indices[layer - 1])
            & edgelist["post"].isin(layer_indices[layer])
        ]
        layer_indices[layer] = el["post"].unique()
        if len(layer_indices[layer]) == 0:
            print(f"No neurons found in layer {layer}. Returning None.")
            return None

    # add the weights
    path = []
    for layer in range(target_layer_number):
        el = edgelist[
            (edgelist["pre"].isin(layer_indices[layer]))
            & (edgelist["post"].isin(layer_indices[layer + 1]))
        ]
        el.loc[:, ["layer"]] = layer + 1
        path.append(el)
    return pd.concat(path, ignore_index=True)


def enumerate_paths(
    edgelist: pd.DataFrame,
    start_layer: int = 1,
    end_layer: Optional[int] = None,
) -> List[List[Tuple[Union[str, int], Union[str, int], float]]]:
    """
    Finds all paths that begin with an edge in start_layer and end with an edge in
    end_layer, assuming valid paths proceed layer-by-layer without skipping.

    Args:
      edgelist (pd.DataFrame): The edgelist of the entire graph. Must contain columns:
        "layer", "pre", "post", and "weight". Each row is a directed, weighted edge
        from "pre" to "post" at a given layer.
      start_layer (int): The layer from which all paths must begin. Must be <= end_layer.
      end_layer (int): The layer at which all paths must terminate. If None, defaults to
        the maximum layer in the edgelist.

    Returns:
      List[List[Tuple[Union[str, int], Union[str, int], float]]]: A list of valid paths.
        Each path is a list of (pre, post, weight) tuples, ordered from start to end.
    """
    if end_layer is None:
        end_layer = edgelist["layer"].max()
    if start_layer > end_layer:
        raise ValueError("start_layer must be less than or equal to end_layer.")

    # Build adjacency: adj[layer][pre] → list of (post, weight)
    # pre and post can be either strings or integers
    adj: Dict[int, Dict[Union[str, int], List[Tuple[Union[str, int], float]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for _, row in edgelist.iterrows():
        adj[row["layer"]][row["pre"]].append((row["post"], row["weight"]))

    # list of paths
    # path: list of edges (pre, post, weight)
    paths: List[List[Tuple[Union[str, int], Union[str, int], float]]] = []

    def dfs(
        node: Union[str, int],
        layer: int,
        path: List[Tuple[Union[str, int], Union[str, int], float]],
    ):
        if layer == end_layer:
            paths.append(path)
            return
        for post, wt in adj.get(layer + 1, {}).get(node, []):
            dfs(post, layer + 1, path + [(node, post, wt)])

    for pre, edges in adj.get(start_layer, {}).items():
        for post, wt in edges:
            dfs(post, start_layer, [(pre, post, wt)])

    return paths


def find_path_iteratively(
    inprop_csc: spmatrix,
    steps_cpu: list,
    inidx: arrayable,
    outidx: arrayable,
    target_layer_number: int,
    top_n: int = -1,
    threshold: float = 0,
):
    """
    Iteratively finds the path from the specified output (outidx) back to the
    input (inidx) across multiple layers, using the `find_path_once` function
    to traverse each layer.

    Args:
      inprop_csc (scipy.sparse matrix): The direct connectivity matrix in
        Compressed Sparse Column format.
      steps_cpu (list): A list of compressed connectivity matrices: one
        matrix for each compressed path length.
      inidx (int, float, list, set, numpy.ndarray, or pandas.Series): The
        input neuron indices.
      outidx (int, float, list, set, numpy.ndarray, or pandas.Series): The
        output neuron indices to start the reverse path finding.
      target_layer_number (int): The number of layers to traverse backwards
        from the outidx. If target_layer_number = 1, we are looking at the
        direct synaptic connectivity.
      top_n (int, optional): The number of top connections to consider at each
        layer based on direct connectivity from inprop_csc. If top_n = -1, all
        connections are considered.
      threshold (float, optional): The threshold for the average of the direct
        connectivity from inidx to outidx.

    Returns:
        pd.DataFrame: A DataFrame containing the path data, including the
            pre-synaptic and post-synaptic neuron indices, the layer (direct
            connections from inidx: layer = 1), and the weight (input
            proportion of the postsynaptic neuron) of the direct connection
            between pre and post.
    """

    print(
        'Have you tried "find_paths_of_length()" instead? About the same speed, no need to pre-load `steps`, and more accurate!'
    )

    inprop_csc.data = np.abs(inprop_csc.data)
    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    if len(inidx) == 0 or len(outidx) == 0:
        # raise error
        raise ValueError("The input or output indices are empty!")

    if issparse(inprop_csc):
        # if stepsn is coo, turn into csc
        if inprop_csc.format == "coo":
            inprop_csc = inprop_csc.tocsc()

    # path_indices = []  # This will store the path data as a list of arrays
    dfs = []  # This will store the path data as a list of DataFrames

    current_outidx = outidx
    # from target_layer_number to 1, go back one step at a time
    for layer in range(target_layer_number, 0, -1):
        # Find the indices in the current layer that connect to the next layer
        df = find_path_once(
            inprop_csc,
            steps_cpu,
            inidx,
            current_outidx,
            layer,
            top_n,
            threshold,
        )

        # If no indices are found, break the loop as no path can be formed
        if len(df) == 0:
            print(f"Cannot trace back to the input in {target_layer_number} steps.")
            if (top_n > -1) | (threshold > 0):
                print("Try lowering the threshold or increasing top_n.")
            return

        df["layer"] = layer

        dfs.append(df)

        # Update the outidx for the next iteration to move backwards through
        # layers
        current_outidx = set(df["pre"])

    return pd.concat(dfs)


def create_layered_positions(
    df: pd.DataFrame, priority_indices=None, sort_dict: dict | None = None
) -> dict:
    """
    Creates a dictionary of positions for each neuron in the paths, so that
    the paths can be visualized in a layered manner. It assumes that `df`
    contains the columns 'layer', 'pre_layer', 'post_layer' (or 'layer', 'pre',
    'post'). If a neuron exists in multiple layers, it is plotted multiple
    times.

    Args:
        df (pd.DataFrame): The DataFrame containing the path data, including the layer
            number, pre-synaptic index, and post-synaptic index.
        priority_indices (list, set, pd.Series, numpy.ndarray optional): A list of
            neuron indices that should be plotted on top of each layer. Defaults to None.
        sort_dict (dict, optional): A dictionary of neuron indices as keys and their
            sorting order as values (bigger value is higher in the plot). Defaults to
            None.
    Returns:
        dict: A dictionary of positions for each neuron in the paths, with the
        keys as the neuron indices and the values as the (x, y) coordinates.
    """

    # check if layer numbers in 'layer' column are consecutive. If anyone is
    # absent, raise an error
    if not check_consecutive_layers(df):
        raise ValueError("The layer numbers in 'layer' column are not consecutive. ")

    # if post_layer and pre_layer are not present, create them
    if "post_layer" not in df.columns:
        df["post_layer"] = df["post"].astype(str) + "_" + df["layer"].astype(str)
    if "pre_layer" not in df.columns:
        df["pre_layer"] = df["pre"].astype(str) + "_" + (df["layer"] - 1).astype(str)

    if priority_indices is not None:
        priority_indices = set(priority_indices)

    number_of_layers = df["layer"].nunique()
    layer_width = 1.0 / (number_of_layers + 1)
    positions = {}

    name_to_idx = dict(zip(df.pre_layer, df["pre"]))
    name_to_idx.update(dict(zip(df.post_layer, df["post"])))

    global_to_local_layer_number = {
        l: i for i, l in enumerate(sorted(df["layer"].unique()))
    }
    df.loc[:, ["local_layer"]] = df["layer"].map(global_to_local_layer_number)

    for layer in range(0, number_of_layers + 1):
        if layer != number_of_layers:
            layer_name = list(
                set(df.pre_layer[df.local_layer == layer]).union(
                    set(df.post_layer[df.local_layer == (layer - 1)])
                )
            )
        else:
            layer_name = list(set(df.post_layer[df.local_layer == layer - 1]))

        if sort_dict is not None:
            layer_name = sorted(
                layer_name, key=lambda x: sort_dict.get(x, float("-inf"))
            )

        if priority_indices is not None:
            layer_name_priority = [
                item for item in layer_name if name_to_idx[item] in priority_indices
            ]
            layer_name_not = [
                item for item in layer_name if name_to_idx[item] not in priority_indices
            ]

            layer_name = layer_name_not + layer_name_priority
        for index, neuron in enumerate(layer_name, start=1):
            positions[neuron] = (
                layer * layer_width,
                # the later in the list, the higher
                index * 1.0 / (len(layer_name) + 1),
            )

    return positions


def remove_excess_neurons(
    df: pd.DataFrame,
    keep=None,
    target_indices=None,
    keep_targets_in_middle: bool = False,
) -> pd.DataFrame:
    """After filtering, some neurons are no longer on the paths between the input and
    output neurons. This function removes those neurons from the paths.

    Args:
        df (pd.Dataframe): a filtered dataframe with similar structure as the dataframe
            returned by `find_paths_of_length()`.
        keep (list, set, pd.Series, numpy.ndarray, str, optional): A list of neuron
            indices that should be kept in the paths, even if they don't connect between
            input and target in the last layer. Defaults to None.
        target_indices (list, set, pd.Series, numpy.ndarray, str, optional): A list of
            target neuron indices that should be kept in the last layer. Defaults to
            None, in which case all neurons in the last layer in `df` would be kept.
        keep_targets_in_middle (bool, optional): If True, the target_indices are kept in
            the middle layers as well, even if they don't connect between input and
            target in the last layer. Defaults to False.

    Returns:
        pd.Dataframe: a dataframe with similar structure as the result of
        `find_paths_of_length()`, with the excess neurons removed. If no path is found,
        returns None.
    """

    if df.shape[0] == 0:
        # raise error
        raise ValueError(
            "No connections found in the input of `remove_excess_neurons()`. "
        )

    max_layer_num = df["layer"].max()
    if max_layer_num == 1:
        return df

    # if target_indices are provided, first use this to filter the last layer ----
    if target_indices is not None:
        # if it's a string or int, convert to list
        if isinstance(target_indices, str) or (type(target_indices) == int):
            target_indices = [target_indices]
        target_indices = set(target_indices)
        # check if datatype is the same, between target_indices and post
        if not all([type(i) == type(df["post"].iloc[0]) for i in target_indices]):
            raise ValueError(
                f"The datatype in `target_indices` should be the same as the elements in `post` column of the DataFrame. Elements in `post` is {type(df.post.iloc[0])}."
            )

        if not target_indices.issubset(df[df["layer"] == df["layer"].max()]["post"]):
            raise ValueError(
                "The target indices are not in the post-synaptic neurons of the last layer. Here are the indices of the last layer: ",
                ", ".join(df[df["layer"] == df["layer"].max()]["post"].unique()),
                ". Your target_indices should be a subset.",
            )

        df = df[(df["layer"] != df["layer"].max()) | df["post"].isin(target_indices)]

    # check if all layer numbers are consecutive ----
    if not check_consecutive_layers(df):
        print(
            "Warning: The layer numbers are not consecutive. Will only use the"
            " consecutive layers from the last one."
        )
        selected = []
        # get the consecutive layers from the last one
        for l in range(df["layer"].max(), 0, -1):
            if l in df["layer"].unique():
                selected.append(l)
            else:
                break
        df = df[df["layer"].isin(selected)]

        global_to_local_layer_number = {
            l: i for i, l in enumerate(sorted(df["layer"].unique()))
        }
        df.loc[:, ["local_layer"]] = df["layer"].map(global_to_local_layer_number)

    elif df["layer"].min() != 1:
        # if the layer numbers do not start from 1
        print(
            "Warning: The layer numbers do not start from 1. Will only use the "
            "consecutive layers from the last one."
        )
        global_to_local_layer_number = {
            l: i for i, l in enumerate(sorted(df["layer"].unique()))
        }
        df.loc[:, ["local_layer"]] = df["layer"].map(global_to_local_layer_number)

    else:
        df.loc[:, ["local_layer"]] = df["layer"]

    # while there is any layer whose post is not the same as the next layer's pre ----
    while any(
        [
            set(df[df.local_layer == i]["post"])
            != set(df[df.local_layer == (i + 1)]["pre"])
            for i in range(1, df.local_layer.max())
        ]
    ):
        if keep is not None:
            if isinstance(keep, str):
                keep = [keep]
            keep = set(keep)

            if all(
                [
                    set(df[df.local_layer == i]["post"]).union(keep)
                    == set(df[df.local_layer == (i + 1)]["pre"]).union(keep)
                    for i in range(1, df.local_layer.max())
                ]
            ):
                break
        else:
            keep = set()

        if keep_targets_in_middle:
            if target_indices is not None:
                keep = keep.union(target_indices)

        # start adding each layer to df_layers_update ---
        df_layers_update = []
        # if there are only two layers
        if df.local_layer.max() == 2:
            df_layer = df[df.local_layer == 2]
            df_prev_layer = df[df.local_layer == 1]

            # filter by pre in the second layer
            df_layers_update.append(
                df_layer[df_layer["pre"].isin(set(df_prev_layer["post"]).union(keep))]
            )
            # filter by post in the first layer
            df_layers_update.append(
                df_prev_layer[
                    df_prev_layer["post"].isin(set(df_layer["pre"]).union(keep))
                ]
            )
            df = pd.concat(df_layers_update)

        else:
            for i in range(2, df.local_layer.max()):
                df_layer = df[df.local_layer == i]
                df_next_layer = df[df.local_layer == i + 1]
                df_prev_layer = df[df.local_layer == i - 1]

                # pre that should be in the current layer: an intersection of
                # previous layer's post; and current layer's pre
                df_pre = set.intersection(
                    set(df_prev_layer["post"]) & set(df_layer["pre"])
                )
                # post that should be in the current layer: an intersection of
                # current layer's post; and next layer's pre
                df_post = set.intersection(
                    set(df_layer["post"]), set(df_next_layer["pre"])
                )

                if i == 2:
                    # add edges in the first layer
                    df_prev_layer = df_prev_layer[
                        df_prev_layer["post"].isin(df_pre.union(keep))
                    ]
                    df_layers_update.append(df_prev_layer)

                df_layer = df_layer[
                    df_layer["pre"].isin(df_pre.union(keep))
                    & df_layer["post"].isin(df_post.union(keep))
                ]

                if df_layer.shape[0] == 0:
                    print(
                        "No path found. Try lowering the threshold for the edges to be included in the path."
                    )
                    return
                df_layers_update.append(df_layer)

                if i == (df.local_layer.max() - 1):
                    # add edges in the last layer
                    if target_indices is None:
                        df_next_layer = df_next_layer[
                            df_next_layer["pre"].isin(df_post.union(keep))
                        ]
                    else:
                        df_next_layer = df_next_layer[
                            df_next_layer["pre"].isin(
                                df_post.union(keep).union(target_indices)
                            )
                        ]
                    df_layers_update.append(df_next_layer)

            df = pd.concat(df_layers_update)

        # at this point, if no edges left: return None
        if df.shape[0] == 0:
            print("No path found. Try relaxing the criteria for edge inclusion.")
            return

    df = df.loc[:, df.columns != "local_layer"]

    # in case we removed all the connections in the last layer
    if df["layer"].max() != max_layer_num:
        print(
            "No path found. Try lowering the threshold for the edges to be included in the path."
        )
        return
    return df


def remove_excess_neurons_batched(
    df: pd.DataFrame,
    keep=None,
    target_indices=None,
    keep_targets_in_middle: bool = False,
) -> pd.DataFrame:
    """Does the same thing as `remove_excess_neurons()`, but for batched input
    (i.e. assumes column `batch` in `df`).

    Args:
        df (pd.DataFrame): a filtered dataframe with similar structure as the
            dataframe returned by `find_path_iteratively()`. Must contain a
            column `batch`.
        keep (list, set, pd.Series, numpy.ndarray, str, optional): A list of
            neuron indices that should be kept in the paths, even if they don't
            connect between input and target in the last layer. Defaults to
            None.
        target_indices (list, set, pd.Series, numpy.ndarray, str, optional): A
            list of target neuron indices that should be kept in the last
            layer. Defaults to None, in which case all neurons in the last
            layer in `df` would be kept.
        keep_targets_in_middle (bool, optional): If True, the target_indices
            are kept in the middle layers as well, even if they don't connect
            between input and target in the last layer. Defaults to False.

    Returns:
        pd.DataFrame: The filtered DataFrame containing the path data,
            including the layer number, pre-synaptic index, post-synaptic
            index, and weight
    """
    # first check if 'batch' is in the columns
    if "batch" not in df.columns:
        raise ValueError("The column `batch` is not in the DataFrame.")

    dfs = []
    for b in tqdm(df.batch.unique()):
        df_batch = df[df.batch == b]
        df_batch = remove_excess_neurons(
            df_batch, keep, target_indices, keep_targets_in_middle
        )
        dfs.append(df_batch)
    return pd.concat(dfs)


def filter_paths(
    df: pd.DataFrame,
    threshold: float = 0,
    necessary_intermediate: Dict[int, arrayable] | None = None,
) -> pd.DataFrame:
    """Filters the paths based on the weight threshold and the necessary
    intermediate neurons. The weight threshold refers to the direct
    connectivity between connected neurons in the path. It is recommended to
    not put too may neurons in necessary_intermediate, as it may be too
    stringent and remove all paths.

    Args:
        df (pd.DataFrame): The DataFrame containing the path data, including
            the layer number, pre-synaptic index, post-synaptic index, and
            weight.
        threshold (float, optional): The threshold for the weight of the
            direct connection between pre and post. Defaults to 0.
        necessary_intermediate (dict, optional): A dictionary of necessary
            intermediate neurons, where the keys are the layer numbers
            (starting neurons: 1; directly downstream: 2) and the values are
            the neuron indices (can be int, float, list, set, numpy.ndarray,
            or pandas.Series). Defaults to None.

    Returns:
        pd.DataFrame: The filtered DataFrame containing the path data,
            including the layer number, pre-synaptic index, post-synaptic
            index, and weight.
    """
    if df.shape[0] == 0:
        # raise error
        raise ValueError("The input DataFrame for filter_paths() is empty! ")

    if threshold > 0:
        max_layer_num = df["layer"].max()
        df = df[df.weight > threshold]
        if df.shape[0] == 0:
            print("No edges left after thresholding. Try lowering the threshold.")
            return
        elif df["layer"].max() != max_layer_num:
            print(
                "No edges left in the last layer after thresholding."
                "Try lowering the threshold."
            )
            return

        df = remove_excess_neurons(df)

    if necessary_intermediate is not None:
        for layer, indices in necessary_intermediate.items():
            if type(indices) != list:
                indices = list(to_nparray(indices))

            if layer < 1:
                # error: layer has to be an integer >=1
                raise ValueError("Layer has to be an integer >=1")

            if layer < (df["layer"].max() + 1):
                df = df[df["pre"].isin(indices) | (df["layer"] != layer)]
            elif layer == (df["layer"].max() + 1):
                # filter for targets
                df = df[df["post"].isin(indices) | ((df["layer"] + 1) != layer)]
            else:
                # error: layer number too big
                raise ValueError("Layer number too big")

        df = remove_excess_neurons(df)
    return df


def group_paths(
    paths: pd.DataFrame,
    pre_group: Optional[dict] = None,
    post_group: Optional[dict] = None,
    intermediate_group: dict | None = None,
    avg_within_connected: bool = False,
    outprop: bool = False,
    combining_method: str = "mean",
) -> pd.DataFrame:
    """
    Group the paths by user-specified variable (e.g. cell type, cell class etc.).
    If outprop=False, weights are summed across presynaptic neurons of the same group 
    and combined across all postsynaptic neurons of the same group using combining_method
    (even if some postsynaptic neurons are not in `paths`).
    If outprop=True, weights are summed across postsynaptic neurons of the same group
    and combined across all presynaptic neurons of the same group using combining_method
    (even if some presynaptic neurons are not in `paths`).

    Args:
        paths (pd.DataFrame): The DataFrame containing the path data, looking
            like the output from `find_path_iteratively()`.
        pre_group (dict): A dictionary that maps pre-synaptic neuron indices
            to their respective group.
        post_group (dict): A dictionary that maps post-synaptic neuron indices
            to their respective group.
        intermediate_group (dict, optional): A dictionary that maps
            intermediate neuron indices to their respective group. Defaults to
            None. If None, it will be set to pre_group.
        avg_within_connected (bool, optional): If True, the average weight is
            calculated within the connected neurons of the same group. If
            False, the average weight is calculated across all neurons of the
            same group. Defaults to False.
        outprop (bool, optional): If True, get the summed output proportion (across
            recipient single cells in the same cell type) for each average sender. If
            False (default), get the summed input proportion across all senders for
            each average recipient.
        combining_method (str, optional): Method to combine inputs (outprop=False)
            or outputs (outprop=True). Can be 'sum', 'mean', or 'median'. Defaults to
            'mean'.
    Returns:
        pd.DataFrame: The grouped DataFrame containing the path data,
            including the layer number, pre-synaptic index, post-synaptic
            index, and weight.
    """
    assert combining_method in ["mean", "sum", "median"], (
        "The combining_method should be either 'mean', 'sum' or 'median'. "
        f"Currently it is {combining_method}."
    )

    all_nodes = set(paths["pre"]).union(set(paths["post"]))
    if pre_group is None:
        pre_group = {node: node for node in all_nodes}
    if post_group is None:
        post_group = {node: node for node in all_nodes}

    if intermediate_group is None:
        intermediate_group = pre_group

    # make values in pre_group, post_group, and intermediate_group strings
    pre_group = {k: str(v) for k, v in pre_group.items()}
    post_group = {k: str(v) for k, v in post_group.items()}
    intermediate_group = {k: str(v) for k, v in intermediate_group.items()}

    # add cell type information
    if "layer" in paths.columns:
        # first use intermediate_group, then modify specifically for pre at the
        # first layer, and post at the last layer
        paths["pre_type"] = paths["pre"].map(intermediate_group).astype(str)
        paths["post_type"] = paths["post"].map(intermediate_group).astype(str)

        paths.loc[paths["layer"] == paths["layer"].min(), "pre_type"] = (
            paths.loc[paths["layer"] == paths["layer"].min(), "pre"]
            .map(pre_group)
            .astype(str)
        )
        paths.loc[paths["layer"] == paths["layer"].max(), "post_type"] = (
            paths.loc[paths["layer"] == paths["layer"].max(), "post"]
            .map(post_group)
            .astype(str)
        )
        group_columns = ["layer", "pre_type", "post_type"]
    else:
        paths["pre_type"] = paths["pre"].map(pre_group).astype(str)
        paths["post_type"] = paths["post"].map(post_group).astype(str)
        group_columns = ["pre_type", "post_type"]

    if not outprop:
        # in this case, calculating the summed input proportion across all senders for
        # each average recipient

        if not avg_within_connected:
            # sometimes only one neuron in a type is connected to another type, so
            # only this connection is in paths
            # but to calculate the average weight between two types, we should
            # take into account all the neurons of the post-synaptic type
            # so let's count the number of neurons in each post-synaptic type
            nneuron_per_type = count_keys_per_value(intermediate_group)
            nneuron_per_type.update(count_keys_per_value(post_group))
        else:
            # count number of unique neurons in each type (for each layer)
            # group by post_type, and layer if it exists
            group_columns_sub = list(set(group_columns) - set(["pre_type"]))
            nneuron_per_type = (
                paths.groupby(group_columns_sub)["post"]
                .nunique()
                .reset_index(name="nneuron_post")
            )

        if combining_method == "median":
            paths_weights = (
                paths.groupby(group_columns)["weight"].median().reset_index()
            )
        elif combining_method == "sum":
            # sum across presynaptic neurons of the same type
            paths_weights = paths.groupby(group_columns).weight.sum().reset_index()
        elif combining_method == "mean":
            # sum across presynaptic neurons of the same type
            paths_weights = paths.groupby(group_columns).weight.sum().reset_index()
            # divide by number of postsynaptic neurons of the same type
            if isinstance(nneuron_per_type, pd.DataFrame):
                paths_weights = paths_weights.merge(
                    nneuron_per_type, on=group_columns_sub
                )
            elif isinstance(nneuron_per_type, dict):
                paths_weights["nneuron_post"] = paths_weights.post_type.map(
                    nneuron_per_type
                )
            paths_weights["weight"] = paths_weights.weight / paths_weights.nneuron_post
            paths_weights.drop(columns="nneuron_post", inplace=True)

        if "pre_activation" in paths.columns:
            paths = (
                paths.groupby(group_columns)
                .agg(
                    pre_activation=("pre_activation", "mean"),
                    post_activation=("post_activation", "mean"),
                )
                .reset_index()
            )
            # then merge the weights
            paths = pd.merge(paths, paths_weights, on=group_columns)
        else:
            paths = paths_weights.copy()
        paths.rename(columns={"pre_type": "pre", "post_type": "post"}, inplace=True)

        return paths

    else:  # outprop
        # in this case, calculating the summed output proportion (across recipient
        # single cells in the same cell type) for each average sender
        if not avg_within_connected:
            # sometimes only one neuron in a type is connected to another type, so
            # only this connection is in paths
            # but to calculate the average weight between two types, we should
            # take into account all the neurons of the post-synaptic type
            # so let's count the number of neurons in each post-synaptic type
            nneuron_per_type = count_keys_per_value(intermediate_group)
            nneuron_per_type.update(count_keys_per_value(pre_group))
        else:
            # count number of unique neurons in each type
            group_columns_sub = list(set(group_columns) - set(["post_type"]))
            nneuron_per_type = (
                paths.groupby(group_columns_sub)["pre"]
                .nunique()
                .reset_index(name="nneuron_pre")
            )

        # weights
        if combining_method == "median":
            paths_weights = (
                paths.groupby(group_columns)["weight"].median().reset_index()
            )
        elif combining_method == "sum":
            # sum across presynaptic neurons of the same type
            paths_weights = paths.groupby(group_columns).weight.sum().reset_index()
        elif combining_method == "mean":
            # sum across presynaptic neurons of the same type
            paths_weights = paths.groupby(group_columns).weight.sum().reset_index()
            # divide by number of presynaptic neurons of the same type
            if isinstance(nneuron_per_type, pd.DataFrame):
                paths_weights = paths_weights.merge(
                    nneuron_per_type, on=group_columns_sub
                )
            elif isinstance(nneuron_per_type, dict):
                paths_weights["nneuron_pre"] = paths_weights.pre_type.map(
                    nneuron_per_type
                )
            paths_weights["weight"] = paths_weights.weight / paths_weights.nneuron_pre
            paths_weights.drop(columns="nneuron_pre", inplace=True)

        if "pre_activation" in paths.columns:
            paths = (
                paths.groupby(group_columns)
                .agg(
                    pre_activation=("pre_activation", "mean"),
                    post_activation=("post_activation", "mean"),
                )
                .reset_index()
            )
            # then merge the weights
            paths = pd.merge(paths, paths_weights, on=group_columns)
        else:
            paths = paths_weights.copy()
        paths.rename(columns={"pre_type": "pre", "post_type": "post"}, inplace=True)

        return paths


def compare_layered_paths(
    paths: List[pd.DataFrame],
    priority_indices=None,
    neuron_to_sign: dict | None = None,
    sign_color_map: dict = {1: "red", -1: "blue"},
    el_colours: List[str] = ["rosybrown", "burlywood"],
    legend_labels: List[str] = ["Path 1", "Path 2"],
    weight_decimals: int = 2,
    figsize: tuple = (10, 8),
    label_pos: List[float] = [0.7, 0.7],
):
    """
    Compare two layered paths by overlaying them and annotating the weights. The paths
    should be in the format of the output from `find_path_iteratively()`. The width of
    the edges is based on the weight in the first path, when the connection is present
    in both paths.

    Args:
        paths (List[pd.DataFrame]): A list of two DataFrames containing the
            path data, including columns 'layer', 'pre', 'post', and 'weight'.
        priority_indices (list, set, pd.Series, numpy.ndarray, optional): A
            list of neuron indices that should be plotted on top of each layer.
            Defaults to None.
        neuron_to_sign (dict, optional): A dictionary that maps neuron indices
            to their signs. Defaults to None.
        sign_color_map (dict, optional): A dictionary that maps neuron signs
            to their respective colors. Defaults to {1: 'red', -1: 'blue'}.
        el_colours (List[str], optional): A list of two colors for the edge
            labels of the two paths. Defaults to ['rosybrown', 'burlywood'].
        legend_labels (List[str], optional): A list of two labels for the
            legend. Defaults to ['Path 1', 'Path 2'].
        weight_decimals (int, optional): The number of decimal places to round
            the edge weights to. Defaults to 2.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 8).

    Returns:
        None: The function plots the graph.
    """
    # add layer columns if necessary
    new_paths = []
    for path in paths:
        # Create a 'post_layer' column to use as unique identifiers
        if "post_layer" not in path.columns:
            path["post_layer"] = (
                path["post"].astype(str) + "_" + path["layer"].astype(str)
            )
        if "pre_layer" not in path.columns:
            path["pre_layer"] = (
                path["pre"].astype(str) + "_" + (path.layer - 1).astype(str)
            )
        new_paths.append(path)

    composite_paths = pd.concat(new_paths)
    # get unique edges
    composite_paths.drop_duplicates(["pre_layer", "post_layer"], inplace=True)
    composite_G = nx.from_pandas_edgelist(
        composite_paths,
        "pre_layer",
        "post_layer",
        ["weight"],
        create_using=nx.DiGraph(),
    )

    # Labels for nodes
    labels = dict(zip(composite_paths.post_layer, composite_paths.post))
    labels.update(dict(zip(composite_paths.pre_layer, composite_paths.pre)))

    # Determine the width of the edges
    weights = [composite_G[u][v]["weight"] for u, v in composite_G.edges()]
    weight_min = min(weights)
    weight_max = max(weights)
    if weight_min == weight_max:
        widths = [1.0] * len(weights)
    else:
        widths = [1 + 9 * (w - weight_min) / (weight_max - weight_min) for w in weights]

    # Generate positions
    positions = create_layered_positions(composite_paths, priority_indices)

    # Edge colors based on neuron signs
    default_color = "lightgrey"  # Default edge color

    # specify edge colour based on pre-neuron sign, if available
    edge_colors = []
    for u, _ in composite_G.edges():
        pre_neuron = labels[u]
        if neuron_to_sign and pre_neuron in neuron_to_sign:
            edge_colors.append(
                sign_color_map.get(neuron_to_sign[pre_neuron], default_color)
            )
        else:
            edge_colors.append(default_color)

    # Plot the graph
    _, ax = plt.subplots(figsize=figsize)

    nx.draw(
        composite_G,
        pos=positions,
        labels=labels,
        with_labels=True,
        node_size=100,
        node_color="lightblue",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
        font_size=8,
        width=widths,
        ax=ax,
        edge_color=edge_colors,
    )

    G1 = nx.from_pandas_edgelist(
        paths[0],
        "pre_layer",
        "post_layer",
        ["weight"],
        create_using=nx.DiGraph(),
    )
    G2 = nx.from_pandas_edgelist(
        paths[1],
        "pre_layer",
        "post_layer",
        ["weight"],
        create_using=nx.DiGraph(),
    )
    el1 = {
        (u, v): f'{data["weight"]:.{weight_decimals}f}'
        for u, v, data in G1.edges(data=True)
    }
    el2 = {
        (u, v): f'{data["weight"]:.{weight_decimals}f}'
        for u, v, data in G2.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        G1,
        pos=positions,
        edge_labels=el1,
        horizontalalignment="left",
        verticalalignment="top",
        font_color=el_colours[0],
        ax=ax,
        label_pos=label_pos[0],
    )
    nx.draw_networkx_edge_labels(
        G2,
        pos=positions,
        edge_labels=el2,
        horizontalalignment="right",
        verticalalignment="bottom",
        font_color=el_colours[1],
        ax=ax,
        label_pos=label_pos[1],
    )

    ax.set_ylim(0, 1)
    # add lengend using the el_colours
    ax.legend(
        handles=[
            mpatches.Patch(color=el_colours[0], label=legend_labels[0]),
            mpatches.Patch(color=el_colours[1], label=legend_labels[1]),
        ],
        loc="upper right",
    )
    plt.show()


@dataclass
class XORCircuit:
    # Input nodes
    input1: list
    input2: list
    # Middle layer nodes
    exciter1: str  # Takes input from input1 only
    exciter2: str  # Takes input from input2 only
    inhibitor: str  # Takes input from both inputs
    # Output node
    output: list


def find_xor(paths: pd.DataFrame) -> List[XORCircuit]:
    """
    Find XOR-like circuits in a 3-layer network, based on [Wang et al. 2024]
    (https://www.biorxiv.org/content/10.1101/2024.09.24.614724v2). Note: this
    function currently ignores middle excitatory neruons that receive both
    inputs.

    Args:
        paths: DataFrame with columns ['pre', 'post', 'sign', 'layer']
        pre: source node
        post: target node
        sign: 1 (excitatory) or -1 (inhibitory)
        layer: 1 (input->middle) or 2 (middle->output)

    Returns:
    List of XORCircuit objects, each representing a found XOR motif
    """
    # checking input ----
    if set(paths["layer"]) != {1, 2}:
        raise ValueError(
            "The input DataFrame should have exactly 2 unique layers, 1 and 2"
        )

    # check column names of paths
    if not {"pre", "post", "sign", "layer"}.issubset(paths.columns):
        raise ValueError(
            "The input DataFrame should have columns ['pre', 'post', 'sign', "
            "'layer']"
        )

    # check values of signs: if it's a subset of {1, -1}
    if not set(paths["sign"]) <= {1, -1}:
        raise ValueError(
            f"The input DataFrame should have values 1 (excitatory) or -1 (inhibitory) in the 'sign' column, but got {set(paths['sign'])}"
        )

    # define variables ----
    circuits: list[XORCircuit] = []

    exciters = paths["pre"][(paths["layer"] == 2) & (paths["sign"] == 1)].unique()
    inhibitors = paths["pre"][(paths["layer"] == 2) & (paths["sign"] == -1)].unique()

    l1 = paths[paths["layer"] == 1]
    l2 = paths[paths["layer"] == 2]

    exciter_us_dict: dict[int | str, set[int | str]] = {
        exc: set(l1["pre"][l1["post"] == exc]) for exc in exciters
    }
    inhibitor_us_dict = {inh: set(l1["pre"][l1["post"] == inh]) for inh in inhibitors}

    exciter_ds_dict = {exc: set(l2["post"][l2["pre"] == exc]) for exc in exciters}
    inhibitor_ds_dict = {inh: set(l2["post"][l2["pre"] == inh]) for inh in inhibitors}

    # main algorithm ----
    for e1, e2 in itertools.combinations(exciters, 2):
        common = exciter_us_dict[e1] & exciter_us_dict[e2]
        onlye1 = exciter_us_dict[e1] - common
        onlye2 = exciter_us_dict[e2] - common

        for i in inhibitors:
            e1_i_intersect = onlye1 & inhibitor_us_dict[i]
            e2_i_intersect = onlye2 & inhibitor_us_dict[i]
            if (len(e1_i_intersect) == 0) or (len(e2_i_intersect) == 0):
                continue
            targets = exciter_ds_dict[e1] & exciter_ds_dict[e2] & inhibitor_ds_dict[i]
            if not targets:
                continue
            circuits.append(
                XORCircuit(
                    input1=list(e1_i_intersect),
                    input2=list(e2_i_intersect),
                    exciter1=e1,
                    exciter2=e2,
                    inhibitor=i,
                    output=list(targets),
                )
            )

    return circuits


def path_for_ngl(path):
    """
    Convert a path DataFrame to one that can be used to visualize the path in
    neuroglancer with `get_ngl_link(df_format = 'long')`. Neurons are coloured
    by their (indirect) connectivity (calculated using
    `effective_conn_from_paths()`) to an average neuron in the last layer. `pre`
    and `post` columns must contain neuron ids. This function can be used for
    visualizing signal propagation in a pathway.

    Args:
        path (pd.DataFrame): The DataFrame containing the path data, including
            the layer number, pre-synaptic index, post-synaptic index, and
            weight.

    Returns:
        pd.DataFrame: A DataFrame with columns 'neuron_id', 'layer', and
            'activation' (which is (indirect) connectivity in this case),
            suitable for Neuroglancer visualization.
    """

    out = []
    from .compress_paths import effective_conn_from_paths

    for alayer in path.layer.unique():
        path_sub = path[path.layer >= alayer]
        effconn = effective_conn_from_paths(path_sub)
        # get mean across columns, for each row
        effconn = effconn.mean(axis=1).to_frame()
        effconn.loc[:, ["layer"]] = alayer
        out.append(effconn)

    out = pd.concat(out, axis=0).reset_index()
    out.columns = ["neuron_id", "activation", "layer"]

    out = pd.concat(
        [
            out,
            (
                pd.DataFrame(
                    {
                        "neuron_id": path_sub.post,
                        "activation": 1,
                        "layer": alayer + 1,
                    }
                )
            ),
        ]
    )

    return out


def connected_components(
    paths: pd.DataFrame,
    threshold: float = 0,
) -> list:
    """
    Find connected components in a directed graph represented by a DataFrame of paths.
    The DataFrame should contain columns 'pre', 'post', 'layer', and 'weight'. The
    function filters the paths based on a weight threshold and then constructs a
    directed graph using NetworkX. It identifies weakly connected components in the
    graph and returns a list of DataFrames of paths, each representing a connected
    component.

    Args:
        paths (pd.DataFrame): The DataFrame containing the path data, including
            the layer number, pre-synaptic index, post-synaptic index, and
            weight.
        threshold (float, optional): The threshold for the weight of the
            direct connection between pre and post. Defaults to 0.

    Returns:
        list: A list of DataFrames, each representing a connected component
            in the directed graph.
    """
    paths = filter_paths(paths, threshold)
    paths_unique_edges = paths.drop_duplicates(subset=["pre", "post"])
    # Create a graph from the DataFrame
    G = nx.from_pandas_edgelist(
        paths_unique_edges,
        source="pre",
        target="post",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )

    # Find connected components
    weak_components = list(nx.weakly_connected_components(G))

    components = []
    for i, component in enumerate(weak_components):
        path = paths[paths.pre.isin(component) & paths.post.isin(component)]
        path.loc[:, ["component_idx"]] = i
        path = remove_excess_neurons(path)
        components.append(path)
    return components


def el_within_n_steps(
    inprop: spmatrix,
    inidx: arrayable,
    outidx: arrayable,
    n: int,
    threshold: float = 0,
    pre_group: dict = None,
    post_group: dict = None,
    return_raw_el: bool = False,
    combining_method: str = "mean",
    avg_within_connected: bool = False,
    all_connections_between_groups: bool = False,
):
    """
    Find paths within a specified number of steps in a directed graph, starting from
    input indices and ending at output indices. The unique edges are returned. Filtering
    by `threshold` happens after grouping if `pre_group` and `post_group` are provided.

    `avg_within_connected` calculates the weight based on the connected neurons, instead
    of all neurons in any group involved. This might be useful when obtaining paths from
    one single neuron in a cell type, while there are many other neurons in the same
    type. This might be especially useful for optic lobe connectivity analysis that's
    spatially local.

    Two neuron groups might be connected to different extents in different layers
    (when a pair of cell types are connected with different individual neurons in
    different layers.). In that case the highest weight is returned by default, but see
    `all_connections_between_groups` argument.

    Args:
        inprop (spmatrix): The connectivity matrix, with presynaptic in the rows.
        inidx (arrayable): The input neuron indices to start the paths from.
        outidx (arrayable): The output neuron indices to end the paths at.
        n (int): The maximum number of hops. n=1 for direct connections.
        threshold (float, optional): The threshold for the weight of the direct
            connection between pre and post. If `pre_group` or `post_group` are
            provided, filtering happens after grouping. Defaults to 0.
        pre_group (dict, optional): A dictionary mapping pre neuron indices to
            their respective groups. Defaults to None.
        post_group (dict, optional): A dictionary mapping post neuron indices to
            their respective groups. Defaults to None.
        return_raw_el (bool, optional): If True, returns the raw edges before
            grouping. Defaults to False.
        combining_method (str, optional): Method to combine inputs (outprop=False)
            or outputs (outprop=True). Can be 'sum', 'mean', or 'median'. Defaults to
            'mean'.
        avg_within_connected (bool, optional): If True, the weight is calculated within
            the *connected* neurons of the same group. If False, the weight is
            calculated across *all* neurons of the same group. Defaults to False.
        all_connections_between_groups (bool, optional): If True, use all connections
            between groups inidx and outidx are in, even if inidx doesn't cover all
            neurons in that group. For example, if inidx is *one* L1 neuron, and
            pre_group maps indices to cell type, outidx is *one* Tm3 neuron, then the
            function will return L1->Tm3 connections for *all* L1 and Tm3 neurons.
            Defaults to False.
    Returns:
        pd.DataFrame: A DataFrame containing the edges of the paths found, including
        columns 'pre', 'post', and 'weight'. If `return_raw_el` is True, returns a
        tuple of two DataFrames: the first is the grouped edges, and the second is the
        raw edges before grouping. If not paths are found, returns None.
    """

    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    all_paths = []
    raw_el = []
    for i in tqdm(range(n)):
        paths = find_paths_of_length(inprop, inidx, outidx, i + 1)
        if paths is None or paths.shape[0] == 0:
            continue
        if return_raw_el:
            raw_el.append(paths)
        if pre_group is not None and post_group is not None:
            paths = group_paths(
                paths,
                pre_group,
                post_group,
                avg_within_connected=avg_within_connected,
                combining_method=combining_method,
            )
        paths = filter_paths(paths, threshold)
        all_paths.append(paths)
    if len(all_paths) == 0:
        return None
    all_paths = pd.concat(all_paths, axis=0)
    el = all_paths.groupby(["pre", "post"])["weight"].max().reset_index()

    if all_connections_between_groups:
        from .compress_paths import result_summary

        new_inidx = [idx for idx, grp in pre_group.items() if grp in el.pre]
        new_outidx = [idx for idx, grp in post_group.items() if grp in el.post]
        mat = result_summary(
            inprop,
            new_inidx,
            new_outidx,
            pre_group,
            post_group,
            display_threshold=0,
            display_output=False,
        )
        # make longer
        mat_long = mat.melt(ignore_index=False).reset_index()
        mat_long.columns = ["pre", "post", "weight"]
        el = el[["pre", "post"]].merge(mat_long, on=["pre", "post"], how="left")

    if return_raw_el:
        raw_el = pd.concat(raw_el, axis=0)
        # pre, post are single neurons, so the rows should be real duplicates
        raw_el = raw_el.drop_duplicates(subset=["pre", "post", "weight"])
        return el, raw_el
    else:
        return el
