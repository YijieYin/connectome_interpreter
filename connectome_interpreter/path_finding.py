from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from tqdm import tqdm
from numbers import Real
from typing import Collection, Dict, List
import itertools

from .utils import to_nparray, count_keys_per_value, check_consecutive_layers

# types that can be made into a numeric numpy array
arrayable = Real | Collection[Real]


def find_path_once(inprop_csc, steps_cpu, inidx: arrayable, outidx: arrayable, target_layer_number, top_n=-1, threshold=0):
    """
    Finds the path once between input and output, of distance target_layer_number, returning indices
    of neurons in the previous layer that connect the input with the output. This works by taking the top_n direct upstream partners of the outidx neurons, and intersect those with neurons 'effectively' connected (through steps_cpu) to the inidx neurons.

    Args:
      inprop_csc (scipy.sparse.csc_matrix): The connectivity matrix in Compressed Sparse Column format.
      steps_cpu (list): A list of compressed connectivity matrices: one matrix for each compressed path length.
      inidx (int, float, list, set, numpy.ndarray, or pandas.Series): The input neuron index/indices.
      outidx (int, float, list, set, numpy.ndarray, or pandas.Series): The output neuron index/indices.
      target_layer_number (int): The target layer number to examine. Must be >= 1. When target_layer_number = 1, we are looking at the direct synaptic connectivity.
      top_n (int, optional): The number of top connections to consider based on direct connectivity from inprop_csc. If top_n = -1, all connections are considered.
      threshold (float, optional): The threshold of the direct connectivity from inidx to an average outidx.

    Returns:
      np.ndarray: An array of neuron indices in the previous layer that have significant connectivity, connecting between the `inidx` and `outidx`.
    """

    # make sure they are integers
    inidx = [int(i) for i in inidx]
    outidx = [int(i) for i in outidx]

    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    if target_layer_number == 1:
        # if the target layer is 1, we are looking at the direct synaptic connectivity
        # so we just need to find the indices of the non-zero values in the inprop_csc matrix
        # that correspond to the outidx, and intersect those with the inidx we are interested in.
        colidx = inidx
    else:
        # first get the neurons that effectively connect to inidx, at layer number target_layer_number - 1.
        # for example, in the ORN->PN->KC case, target_layer_number = 2 for the KCs.
        # top_n_row_indices are indices of the PNs that connect to the KCs. So in this case we should use direct connectivity from PNs to get the ORNs.
        # so when target_layer_number == 2, we should use steps_cpu[0] (direct connectivity), that is, target_layer_number-2:
        # subtract 1 for getting top_n_row_indices below which is going one step upstream; and another for 0-based indexing in steps_cpu.
        # the next line gets the targets that receive non-zero compressed input from inidx
        # when target_layer_number >= 2.
        # .nonzero() returns row, column of the nonzero values.
        colidx = steps_cpu[target_layer_number-2][inidx, :].nonzero()[1]

    # then go back one step from the outidx, based on the direct connectivity matrix (inprop)
    us = inprop_csc[:, outidx].nonzero()
    # intersect the non-zero upstream neurons with those effectively connected across layers (colidx)
    # these are still row indices of inprop
    intersect = np.intersect1d(us[0], colidx)

    # direct connectivity between target_layer_number-1 and target_layer_number
    submatrix = inprop_csc[intersect, :][:, outidx]
    # in case outidx is more than one index, calculate the average of each row in the submatrix
    # same length as len(intersect)
    row_averages = np.array(submatrix.mean(axis=1)).flatten()

    # thresholding
    # these are indices of row_averages, which can also be used to index row_averages
    thresholded_indices = np.where(row_averages >= threshold)[0]
    thresholded_row_averages = row_averages[thresholded_indices]
    # thresholded_intersect are a subset of of intersect, which are indices of inprop_csc
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
    df = pd.DataFrame({
        # Map back to original indices
        'pre': top_n_row_indices[coo_submatrix.row],
        # Map back to original output indices
        'post': outidx[coo_submatrix.col],
        'weight': coo_submatrix.data
    })

    return df


def find_path_iteratively(inprop_csc, steps_cpu, inidx: arrayable, outidx: arrayable, target_layer_number, top_n=-1, threshold=0):
    """
    Iteratively finds the path from the specified output (outidx) back to the input (inidx) across
    multiple layers, using the `find_path_once` function to traverse each layer.

    Args:
      inprop_csc (scipy.sparse matrix): The direct connectivity matrix in Compressed Sparse Column format.
      steps_cpu (np.ndarray): A list of compressed connectivity matrices: one matrix for each compressed path length.
      inidx (int, float, list, set, numpy.ndarray, or pandas.Series): The input neuron indices.
      outidx (int, float, list, set, numpy.ndarray, or pandas.Series): The output neuron indices to start the reverse path finding.
      target_layer_number (int): The number of layers to traverse backwards from the outidx. If target_layer_number = 1, we are looking at the direct synaptic connectivity.
      top_n (int, optional): The number of top connections to consider at each layer based on direct connectivity from inprop_csc. If top_n = -1, all connections are considered.
      threshold (float, optional): The threshold for the average of the direct connectivity from inidx to outidx.

    Returns:
        pd.DataFrame: A DataFrame containing the path data, including the pre-synaptic and post-synaptic neuron indices, the layer (direct connections from inidx: layer = 1), and the weight (input proportion of the postsynaptic neuron) of the direct connection between pre and post.
    """

    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    if issparse(inprop_csc):
        # if stepsn is coo, turn into csc
        if inprop_csc.format == 'coo':
            inprop_csc = inprop_csc.tocsc()

    # path_indices = []  # This will store the path data as a list of arrays
    dfs = []  # This will store the path data as a list of DataFrames

    current_outidx = outidx
    # from target_layer_number to 1, go back one step at a time
    for layer in range(target_layer_number, 0, -1):
        # Find the indices in the current layer that connect to the next layer
        df = find_path_once(
            inprop_csc, steps_cpu, inidx, current_outidx, layer, top_n, threshold)

        # If no indices are found, break the loop as no path can be formed
        if len(df) == 0:
            print(
                'Cannot trace back to the input :(. Try providing a bigger top_n value, or a lower threshold?')
            break

        df['layer'] = layer

        dfs.append(df)

        # Update the outidx for the next iteration to move backwards through layers
        current_outidx = set(df['pre'])

    return pd.concat(dfs)


def create_layered_positions(df: pd.DataFrame, priority_indices=None, sort_dict: dict = None) -> dict:
    """
    Creates a dictionary of positions for each neuron in the paths, so that the paths can be visualized in a layered manner. It assumes that `df` contains the columns 'layer', 'pre_layer', 'post_layer' (or 'layer', 'pre', 'post'). If a neuron exists in multiple layers, it is plotted multiple times.
    Args:
        df (pd.DataFrame): The DataFrame containing the path data, including the layer number, pre-synaptic index, and post-synaptic index.
        priority_indices (list, set, pd.Series, numpy.ndarray optional): A list of neuron indices that should be plotted on top of each layer. Defaults to None.
        sort_dict (dict, optional): A dictionary of neuron indices as keys and their sorting order as values. Defaults to None.
    Returns:
        dict: A dictionary of positions for each neuron in the paths, with the keys as the neuron indices and the values as the (x, y) coordinates.
    """

    # check if layer numbers in 'layer' column are consecutive. If anyone is absent, raise an error
    if not check_consecutive_layers(df):
        raise ValueError(
            "The layer numbers in 'layer' column are not consecutive. ")

    # if post_layer and pre_layer are not present, create them
    if 'post_layer' not in df.columns:
        df['post_layer'] = df['post'].astype(
            str) + '_' + df['layer'].astype(str)
    if 'pre_layer' not in df.columns:
        df['pre_layer'] = df['pre'].astype(
            str) + '_' + (df['layer']-1).astype(str)

    if priority_indices is not None:
        priority_indices = set(priority_indices)

    number_of_layers = df['layer'].nunique()
    layer_width = 1.0 / (number_of_layers + 1)
    positions = {}

    name_to_idx = dict(zip(df.pre_layer, df['pre']))
    name_to_idx.update(dict(zip(df.post_layer, df['post'])))

    global_to_local_layer_number = {l: i for i,
                                    l in enumerate(sorted(df['layer'].unique()))}
    df.loc[:, ['local_layer']] = df['layer'].map(global_to_local_layer_number)

    for layer in range(0, number_of_layers + 1):
        if layer != number_of_layers:
            layer_name = list(set(df.pre_layer[df.local_layer == layer]).union(
                set(df.post_layer[df.local_layer == (layer-1)])))
        else:
            layer_name = list(set(df.post_layer[df.local_layer == layer-1]))

        if sort_dict is not None:
            layer_name = sorted(layer_name, key=lambda x: sort_dict[x])

        if priority_indices is not None:
            layer_name_priority = [
                item for item in layer_name if name_to_idx[item] in priority_indices]
            layer_name_not = [
                item for item in layer_name if name_to_idx[item] not in priority_indices]

            layer_name = layer_name_not + layer_name_priority
        for index, neuron in enumerate(layer_name, start=1):
            positions[neuron] = (layer * layer_width,
                                 index * 1.0 / (len(layer_name) + 1))

    return positions


def remove_excess_neurons(df: pd.DataFrame, keep=None, target_indices=None, keep_targets_in_middle: bool = False) -> pd.DataFrame:
    """After filtering, some neurons are no longer on the paths between the input and output neurons. This function removes those neurons from the paths.

    Args:
        df (pd.Dataframe): a filtered dataframe with similar structure as the dataframe returned by `find_path_iteratively()`.
        keep (list, set, pd.Series, numpy.ndarray, str, optional): A list of neuron indices that should be kept in the paths, even if they don't connect between input and target in the last layer. Defaults to None.
        target_indices (list, set, pd.Series, numpy.ndarray, str, optional): A list of target neuron indices that should be kept in the last layer. Defaults to None, in which case all neurons in the last layer in `df` would be kept.
        keep_targets_in_middle (bool, optional): If True, the target_indices are kept in the middle layers as well, even if they don't connect between input and target in the last layer. Defaults to False.

    Returns:
        pd.Dataframe: a dataframe with similar structure as the result of `find_path_iteratively()`, with the excess neurons removed.
    """

    if df.shape[0] == 0:
        # raise error
        raise ValueError(
            'There are no connections (`df` is empty)! Threshold too high?')

    max_layer_num = df['layer'].max()
    if max_layer_num == 1:
        return df

    # if target_indices are provided, first use this to filter the last layer ----
    if target_indices is not None:
        # if it's a string or int, convert to list
        if (type(target_indices) == str) or (type(target_indices) == int):
            target_indices = [target_indices]
        target_indices = set(target_indices)
        # check if datatype is the same, between target_indices and post
        if not all([type(i) == type(df['post'].iloc[0]) for i in target_indices]):
            raise ValueError(
                f'The datatype in `target_indices` should be the same as the elements in `post` column of the DataFrame. Elements in `post` is {type(df.post.iloc[0])}.')

        if not target_indices.issubset(df[df['layer'] == df['layer'].max()]['post']):
            raise ValueError('The target indices are not in the post-synaptic neurons of the last layer. Here are the indices of the last layer: ',
                             ', '.join(df[df['layer'] == df['layer'].max()]['post'].unique()), '. Your target_indices should be a subset.')

        df = df[(df['layer'] != df['layer'].max())
                | df['post'].isin(target_indices)]

    # check if all layer numbers are consecutive ----
    if not check_consecutive_layers(df):
        print('Warning: The layer numbers are not consecutive. Will only use the consecutive layers from the last one.')
        selected = []
        # get the consecutive layers from the last one
        for l in range(df['layer'].max(), 0, -1):
            if l in df['layer'].unique():
                selected.append(l)
            else:
                break
        df = df[df['layer'].isin(selected)]

        global_to_local_layer_number = {l: i for i,
                                        l in enumerate(sorted(df['layer'].unique()))}
        df.loc[:, ['local_layer']] = df['layer'].map(
            global_to_local_layer_number)

    elif df['layer'].min() != 1:
        # if the layer numbers do not start from 1
        print('Warning: The layer numbers do not start from 1. Will only use the consecutive layers from the last one.')
        global_to_local_layer_number = {l: i for i,
                                        l in enumerate(sorted(df['layer'].unique()))}
        df.loc[:, ['local_layer']] = df['layer'].map(
            global_to_local_layer_number)

    else:
        df.loc[:, ['local_layer']] = df['layer']

    # while there is any layer whose post is not the same as the next layer's pre ----
    while any([set(df[df.local_layer == i]['post']) != set(df[df.local_layer == (i+1)]['pre']) for i in range(1, df.local_layer.max())]):
        if keep is not None:
            if type(keep) == str:
                keep = [keep]
            keep = set(keep)

            if all([set(df[df.local_layer == i]['post']).union(keep) == set(df[df.local_layer == (i+1)]['pre']).union(keep) for i in range(1, df.local_layer.max())]):
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
                df_layer[df_layer['pre'].isin(set(df_prev_layer['post']).union(keep))])
            # filter by post in the first layer
            df_layers_update.append(
                df_prev_layer[df_prev_layer['post'].isin(set(df_layer['pre']).union(keep))])
            df = pd.concat(df_layers_update)

        else:
            for i in range(2, df.local_layer.max()):
                df_layer = df[df.local_layer == i]
                df_next_layer = df[df.local_layer == i+1]
                df_prev_layer = df[df.local_layer == i-1]

                # pre that should be in the current layer: an intersection of previous layer's post; and current layer's pre
                df_pre = set.intersection(
                    set(df_prev_layer['post']) & set(df_layer['pre']))
                # post that should be in the current layer: an intersection of current layer's post; and next layer's pre
                df_post = set.intersection(
                    set(df_layer['post']), set(df_next_layer['pre']))

                if i == 2:
                    # add edges in the first layer
                    df_prev_layer = df_prev_layer[df_prev_layer['post'].isin(
                        df_pre.union(keep))]
                    df_layers_update.append(df_prev_layer)

                df_layer = df_layer[df_layer['pre'].isin(
                    df_pre.union(keep)) & df_layer['post'].isin(df_post.union(keep))]

                if df_layer.shape[0] == 0:
                    raise ValueError(
                        'No path found. Try lowering the threshold for the edges to be included in the path.')
                df_layers_update.append(df_layer)

                if i == (df.local_layer.max()-1):
                    # add edges in the last layer
                    if target_indices is None:
                        df_next_layer = df_next_layer[df_next_layer['pre'].isin(
                            df_post.union(keep))]
                    else:
                        df_next_layer = df_next_layer[df_next_layer['pre'].isin(
                            df_post.union(keep).union(target_indices))]
                    df_layers_update.append(df_next_layer)

            df = pd.concat(df_layers_update)

    df.drop(columns='local_layer', inplace=True)

    # in case we removed all the connections in the last layer
    if df['layer'].max() != max_layer_num:
        raise ValueError(
            'No path found. Try lowering the threshold for the edges to be included in the path. Currently no connections are found in the last layer.')
    return df


def remove_excess_neurons_batched(df: pd.DataFrame, keep=None, target_indices=None, keep_targets_in_middle: bool = False) -> pd.DataFrame:
    """Does the same thing as `remove_excess_neurons()`, but for batched input (i.e. assumes column `batch` in `df`). 

    Args:
        df (pd.DataFrame): a filtered dataframe with similar structure as the dataframe returned by `find_path_iteratively()`. Must contain a column `batch`.
        keep (list, set, pd.Series, numpy.ndarray, str, optional): A list of neuron indices that should be kept in the paths, even if they don't connect between input and target in the last layer. Defaults to None.
        target_indices (list, set, pd.Series, numpy.ndarray, str, optional): A list of target neuron indices that should be kept in the last layer. Defaults to None, in which case all neurons in the last layer in `df` would be kept.
        keep_targets_in_middle (bool, optional): If True, the target_indices are kept in the middle layers as well, even if they don't connect between input and target in the last layer. Defaults to False.

    Returns:
        pd.DataFrame: The filtered DataFrame containing the path data, including the layer number, pre-synaptic index, post-synaptic index, and weight
    """
    # first check if 'batch' is in the columns
    if 'batch' not in df.columns:
        raise ValueError('The column `batch` is not in the DataFrame.')

    dfs = []
    for b in tqdm(df.batch.unique()):
        df_batch = df[df.batch == b]
        df_batch = remove_excess_neurons(
            df_batch, keep, target_indices, keep_targets_in_middle)
        dfs.append(df_batch)
    return pd.concat(dfs)


def filter_paths(df: pd.DataFrame, threshold: float = 0, necessary_intermediate: Dict[int, arrayable] = None) -> pd.DataFrame:
    """Filters the paths based on the weight threshold and the necessary intermediate neurons. The weight threshold refers to the direct connectivity between connected neurons in the path. It is recommended to not put too may neurons in necessary_intermediate, as it may be too stringent and remove all paths.

    Args:
        df (pd.DataFrame): The DataFrame containing the path data, including the layer number, pre-synaptic index, post-synaptic index, and weight.
        threshold (float, optional): The threshold for the weight of the direct connection between pre and post. Defaults to 0.
        necessary_intermediate (dict, optional): A dictionary of necessary intermediate neurons, where the keys are the layer numbers (starting neurons: 1; directly downstream: 2) and the values are the neuron indices (can be int, float, list, set, numpy.ndarray, or pandas.Series). Defaults to None.

    Returns:
        pd.DataFrame: The filtered DataFrame containing the path data, including the layer number, pre-synaptic index, post-synaptic index, and weight.
    """
    if threshold > 0:
        df = df[df.weight > threshold]

        df = remove_excess_neurons(df)

    if necessary_intermediate is not None:
        for layer, indices in necessary_intermediate.items():
            if type(indices) != list:
                indices = list(to_nparray(indices))

            if layer < 1:
                # error: layer has to be an integer >=1
                raise ValueError("Layer has to be an integer >=1")

            if layer < (df['layer'].max()+1):
                df = df[df['pre'].isin(indices) | (df['layer'] != layer)]
            elif layer == (df['layer'].max()+1):
                # filter for targets
                df = df[df['post'].isin(indices) | (df['layer'] != layer)]
            else:
                # error: layer number too big
                raise ValueError("Layer number too big")

        df = remove_excess_neurons(df)
    return df


def group_paths(paths: pd.DataFrame, pre_group: dict, post_group: dict, intermediate_group: dict = None, avg_within_connected: bool = False) -> pd.DataFrame:
    """
    Group the paths by user-specified variable (e.g. cell type, cell class etc.). Weights are summed across presynaptic neurons of the same group and averaged across all postsynaptic neurons of the same group (even if some postsynaptic neurons are not in `paths`).

    Args:
        paths (pd.DataFrame): The DataFrame containing the path data, looking like the output from `find_path_iteratively()`.
        pre_group (dict): A dictionary that maps pre-synaptic neuron indices to their respective group.
        post_group (dict): A dictionary that maps post-synaptic neuron indices to their respective group.
        intermediate_group (dict, optional): A dictionary that maps intermediate neuron indices to their respective group. Defaults to None. If None, it will be set to pre_group.

    Returns:
        pd.DataFrame: The grouped DataFrame containing the path data, including the layer number, pre-synaptic index, post-synaptic index, and weight.
    """

    if intermediate_group is None:
        intermediate_group = pre_group

    # add cell type information
    # first use intermediate_group, then modify specifically for pre at the first layer, and post at the last layer
    paths['pre_type'] = paths['pre'].map(intermediate_group).astype(str)
    paths['post_type'] = paths['post'].map(intermediate_group).astype(str)

    paths.loc[paths['layer'] == paths['layer'].min(), 'pre_type'] = paths.loc[paths['layer'] ==
                                                                              paths['layer'].min(), 'pre'].map(pre_group).astype(str)
    paths.loc[paths['layer'] == paths['layer'].max(), 'post_type'] = paths.loc[paths['layer'] ==
                                                                               paths['layer'].max(), 'post'].map(post_group).astype(str)

    if not avg_within_connected:
        # sometimes only one neuron in a type is connected to another type, so only this connection is in paths
        # but to calculate the average weight between two types, we should take into account all the neurons of the post-synaptic type
        # so let's count the number of neurons in each post-synaptic type
        nneuron_per_type = count_keys_per_value(intermediate_group)
        nneuron_per_type.update(count_keys_per_value(post_group))
    else:
        # count number of unique neurons in each type
        nneuron_per_type = paths.groupby(
            'post_type')['post'].nunique().to_dict()

    # sum across presynaptic neurons of the same type
    paths = paths.groupby(
        ['layer', 'pre_type', 'post_type']).weight.sum().reset_index()
    # divide by number of postsynaptic neurons of the same type
    paths['nneuron_post'] = paths.post_type.map(nneuron_per_type)
    paths['weight'] = paths.weight / paths.nneuron_post
    paths.rename(columns={'pre_type': 'pre',
                 'post_type': 'post'}, inplace=True)
    paths.drop(columns='nneuron_post', inplace=True)

    return paths


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
    Find XOR-like circuits in a 3-layer network, based on [Wang et al. 2024](https://www.biorxiv.org/content/10.1101/2024.09.24.614724v2). Note: this function currently ignores middle excitatory neruons that receive both inputs.

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
    if set(paths['layer']) != {1, 2}:
        raise ValueError(
            "The input DataFrame should have exactly 2 unique layers, 1 and 2")

    # check column names of paths
    if not {'pre', 'post', 'sign', 'layer'}.issubset(paths.columns):
        raise ValueError(
            "The input DataFrame should have columns ['pre', 'post', 'sign', 'layer']")

    # check values of signs: if it's a subset of {1, -1}
    if not set(paths['sign']) <= {1, -1}:
        raise ValueError(
            f"The input DataFrame should have values 1 (excitatory) or -1 (inhibitory) in the 'sign' column, but got {set(paths['sign'])}")

    # define variables ----
    circuits: list[XORCircuit] = []

    exciters = paths['pre'][(paths['layer'] == 2) &
                            (paths['sign'] == 1)].unique()
    inhibitors = paths['pre'][(paths['layer'] == 2)
                              & (paths['sign'] == -1)].unique()

    l1 = paths[paths['layer'] == 1]
    l2 = paths[paths['layer'] == 2]

    exciter_us_dict: dict[int | str, set[int | str]] = {
        exc: set(l1['pre'][l1['post'] == exc]) for exc in exciters}
    inhibitor_us_dict = {inh: set(l1['pre'][l1['post'] == inh])
                         for inh in inhibitors}

    exciter_ds_dict = {exc: set(l2['post'][l2['pre'] == exc])
                       for exc in exciters}
    inhibitor_ds_dict = {inh: set(l2['post'][l2['pre'] == inh])
                         for inh in inhibitors}

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
                    output=list(targets)
                )
            )

    return circuits
