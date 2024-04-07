import numpy as np
import pandas as pd

from .utils import to_nparray


def find_path_once(inprop_csc, steps_cpu, inidx, outidx, target_layer_number, top_n=-1, threshold=0):
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

    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)
    # make sure they are integers as well
    inidx = [int(i) for i in inidx]
    outidx = [int(i) for i in outidx]

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

    # Initialize lists to store data for DataFrame
    pre_list, post_list, weight_list = [], [], []

    # Iterate through each possible combination and store only non-zero weights
    for pre in top_n_row_indices:
        for post in outidx:
            weight = inprop_csc[pre, post]
            if weight > 0:  # Only consider non-zero weights
                pre_list.append(pre)
                post_list.append(post)
                weight_list.append(weight)

    # Construct DataFrame from lists
    df = pd.DataFrame({
        'pre': pre_list,
        'post': post_list,
        'weight': weight_list
    })

    return df


def find_path_iteratively(inprop_csc, steps_cpu, inidx, outidx, target_layer_number, top_n=-1, threshold=0):
    """
    Iteratively finds the path from the specified output (outidx) back to the input (inidx) across
    multiple layers, using the `find_path_once` function to traverse each layer.

    Args:
      inprop_csc (scipy.sparse.csc_matrix): The direct connectivity matrix in Compressed Sparse Column format.
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


def create_layered_positions(df):
    """
    Creates a dictionary of positions for each neuron in the paths, so that the paths can be visualized in a layered manner. It assumes that `df` contains the columns 'layer', 'pre_layer', 'post_layer' (or 'layer', 'pre', 'post'). If a neuron exists in multiple layers, it is plotted multiple times.
    Args:
        df (pd.DataFrame): The DataFrame containing the path data, including the layer number, pre-synaptic index, and post-synaptic index.
    Returns:
        dict: A dictionary of positions for each neuron in the paths, with the keys as the neuron indices and the values as the (x, y) coordinates.
    """

    # if post_layer and pre_layer are not present, create them
    if 'post_layer' not in df.columns:
        df['post_layer'] = df['post'].astype(
            str) + '_' + df['layer'].astype(str)
    if 'pre_layer' not in df.columns:
        df['pre_layer'] = df['pre'].astype(
            str) + '_' + (df.layer-1).astype(str)

    number_of_layers = df.layer.nunique()
    layer_width = 1.0 / (number_of_layers + 1)
    positions = {}

    pre_neurons = df[df['layer'] == 1]['pre_layer'].unique()
    vertical_spacing = 1.0 / (len(pre_neurons) + 1)
    for index, neuron in enumerate(pre_neurons, start=1):
        positions[neuron] = (0, index * vertical_spacing)

    for layer in range(1, number_of_layers + 1):
        posts_in_layer = df[df['layer'] == layer]['post_layer'].unique()
        vertical_spacing = 1.0 / (len(posts_in_layer) + 1)
        for index, post in enumerate(posts_in_layer, start=1):
            positions[post] = (layer * layer_width, index * vertical_spacing)
    return positions


def remove_excess_neurons(df):
    """After filtering, some neurons are no longer on the paths between the input and output neurons. This function removes those neurons from the paths.

    Args:
        df (pd.Dataframe): a filtered dataframe with similar structure as the dataframe returned by `find_path_iteratively()`.

    Returns:
        pd.Dataframe: a dataframe with similar structure as the result of `find_path_iteratively()`, with the excess neurons removed.
    """
    if df.layer.max() == 1:
        return df

    while any([set(df[df.layer == i].post) != set(df[df.layer == (i+1)].pre) for i in range(1, df.layer.max())]):
        df_layers_update = []

        if df.layer.max() == 2:
            df_layer = df[df.layer == 2]
            df_prev_layer = df[df.layer == 1]

            df_layers_update.append(
                df_layer[df_layer.pre.isin(df_prev_layer.post)])
            df_layers_update.append(
                df_prev_layer[df_prev_layer.post.isin(df_layer.pre)])
            df = pd.concat(df_layers_update)

        else:
            for i in range(2, df.layer.max()):
                df_layer = df[df.layer == i]
                df_next_layer = df[df.layer == i+1]
                df_prev_layer = df[df.layer == i-1]

                df_pre = set.intersection(
                    set(df_prev_layer.post) & set(df_layer.pre))
                df_post = set.intersection(
                    set(df_layer.post), set(df_next_layer.pre))

                if i == 2:
                    df_prev_layer = df_prev_layer[df_prev_layer.post.isin(
                        df_pre)]
                    df_layers_update.append(df_prev_layer)

                df_layer = df_layer[df_layer.pre.isin(
                    df_pre) & df_layer.post.isin(df_post)]
                df_layers_update.append(df_layer)

                if i == (df.layer.max()-1):
                    df_next_layer = df_next_layer[df_next_layer.pre.isin(
                        df_post)]
                    df_layers_update.append(df_next_layer)

            df = pd.concat(df_layers_update)
    return df


def filter_paths(df, threshold=0, necessary_intermediate=None):
    """Filters the paths based on the weight threshold and the necessary intermediate neurons. The weight threshold refers to the direct connectivity between connected neurons in the path. It is recommended to not put too may neurons in necessary_intermediate, as it may be too stringent and remove all paths.

    Args:
        df (pd.DataFrame): The DataFrame containing the path data, including the layer number, pre-synaptic index, post-synaptic index, and weight.
        threshold (float, optional): The threshold for the weight of the direct connection between pre and post. Defaults to 0.
        necessary_intermediate (dict, optional): A dictionary of necessary intermediate neurons, where the keys are the layer numbers and the values are the neuron indices. Defaults to None.

    Returns:
        pd.DataFrame: The filtered DataFrame containing the path data, including the layer number, pre-synaptic index, post-synaptic index, and weight.
    """
    if threshold > 0:
        df = df[df.weight > threshold]

        df = remove_excess_neurons(df)

    if necessary_intermediate is not None:
        for layer, indices in necessary_intermediate.items():
            if type(indices) != list:
                indices = [indices]
            df = df[df.post.isin(indices) | (df.layer != layer)]

        df = remove_excess_neurons(df)
    return df
