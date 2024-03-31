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
      threshold (float, optional): The threshold for the average of the direct connectivity from inidx to outidx.

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

    submatrix = inprop_csc[intersect, :][:, outidx]
    # in case outidx is more than one index, calculate the average of each row in the submatrix
    # same length as len(intersect)
    row_averages = np.array(submatrix.mean(axis=1)).flatten()

    # thresholding
    # these are indices of row_averages, which can also be used to index row_averages
    thresholded_indices = np.where(row_averages >= threshold)[0]
    thresholded_row_averages = row_averages[thresholded_indices]
    # thresholded_intersect are a subset of of intersect, which are indices of inprop_csc
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
