import numpy as np
import pandas as pd

from .utils import to_nparray


def find_path_once(inprop_csc, steps_cpu, inidx, outidx, target_layer_number, top_n, threshold = 0):
    """
    Finds the path once between input and output, of distance target_layer_number, returning indices
    of neurons in the previous layer that connect the input with the output. This works by taking the top_n direct upstream partners of the outidx neurons, and intersect those with neurons 'effectively' connected (through steps_cpu) to the inidx neurons.

    Args: 
      inprop_csc (scipy.sparse.csc_matrix): The connectivity matrix in Compressed Sparse Column format.
      steps_cpu (list): A list of compressed connectivity matrices: one matrix for each compressed path length. 
      inidx (int or np.ndarray): The input neuron index/indices.
      outidx (int or np.ndarray): The output neuron index/indices.
      target_layer_number (int): The target layer number to examine. Must be >= 1. When target_layer_number = 1, we are looking at the direct synaptic connectivity.
      top_n (int): The number of top connections to consider based on direct connectivity from inprop_csc. if top_n = -1, all connections are considered.
      threshold (float, optional): The threshold for the average of the direct connectivity from inidx to outidx.

    Returns:
      np.ndarray: An array of neuron indices in the previous layer that have significant connectivity, connecting between the `inidx` and `outidx`.
    """

    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    if target_layer_number == 1:
        # if the target layer is 1, we are looking at the direct synaptic connectivity
        # so we just need to find the indices of the non-zero values in the inprop_csc matrix
        # that correspond to the outidx, and intersect those with the inidx we are interested in.
        colidx = inidx
    else: 
        # first get the neurons that effectively connect to inidx, at layer number target_layer_number - 1. 
        # for example, in the ORN->PN->KC case, target_layer_number = 2 for the KCs. top_n_row_indices are indices of the PNs that connect to the KCs. So in this case we should use direct connectivity to get the ORNs. 
        # so when target_layer_number == 2, we should use steps_cpu[0] (direct connectivity), that is, target_layer_number-2. 
        # subtract 1 for getting top_n_row_indices below which is going one step upstream; and another for 0-based indexing. 
        # the next line gets the targets that receive non-zero compressed input from inidx
        # when target_layer_number >= 2. 
        colidx = steps_cpu[target_layer_number-2][inidx, :].nonzero()[1]

    # then go back one step from the outidx, based on the direct connectivity matrix (inprop)
    # .nonzero() returns row, column of the nonzero values
    us = inprop_csc[:, outidx].nonzero()
    # intersect the non-zero upstream neurons with those effectively connected across layers (colidx)
    # these are still row indices of inprop 
    intersect = np.intersect1d(us[0], colidx)

    submatrix = inprop_csc[intersect, :][:, outidx]
    # in case outidx is more than one index, calculate the average of each row in the submatrix
    row_averages = np.array(submatrix.mean(axis=1)).flatten()

    # thresholding 
    # these are indices of row_averages
    thresholded_indices = np.where(row_averages >= threshold)[0]
    thresholded_row_averages = row_averages[thresholded_indices]
    # these are indices of intersect, which are indices of inprop_csc
    thresholded_intersect = intersect[thresholded_indices]

    # Find the indices (of thresholded_row_averages) of the top n averages
    if top_n == -1:
        top_n_indices = np.argsort(thresholded_row_averages)
    else:
        top_n_indices = np.argsort(thresholded_row_averages)[-top_n:]
    # Get the original row indices corresponding to these top n averages
    top_n_row_indices = thresholded_intersect[top_n_indices]

    return top_n_row_indices


def find_path_iteratively(inprop_csc, steps_cpu, inidx, outidx, target_layer_number, top_n, threshold = 0):
    """
    Iteratively finds the path from the specified output (outidx) back to the input (inidx) across
    multiple layers, using the `find_path_once` function to traverse each layer. 

    Args:
      inprop_csc (scipy.sparse.csc_matrix): The direct connectivity matrix in Compressed Sparse Column format.
      steps_cpu (np.ndarray): A list of compressed connectivity matrices: one matrix for each compressed path length. 
      inidx (int or np.ndarray): The input neuron indices.
      outidx (int or np.ndarray): The output neuron indices to start the reverse path finding.
      target_layer_number (int): The number of layers to traverse backwards from the outidx. If target_layer_number = 1, we are looking at the direct synaptic connectivity.
      top_n (int): The number of top connections to consider at each layer based on direct connectivity from inprop_csc.
      threshold (float, optional): The threshold for the average of the direct connectivity from inidx to outidx.

    Returns:
      list of np.ndarray: A list where each element is an array of neuron indices at each layer
      that constitute a (multi-layered) path from the input index to the output index (inclusive). Input indices only include those that have significant input to the output indices.
    """

    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    path_indices = []  # This will store the path data as a list of arrays

    current_outidx = outidx
    # from target_layer_number to 1, go back one step at a time
    for layer in range(target_layer_number, 0, -1):
        # Find the indices in the current layer that connect to the next layer
        current_layer_indices = find_path_once(
            inprop_csc, steps_cpu, inidx, current_outidx, layer, top_n, threshold)

        # If no indices are found, break the loop as no path can be formed
        if len(current_layer_indices) == 0:
            print(
                'Cannot trace back to the input :(. Try providing a bigger top_n value?')
            break

        # Store the current layer's indices
        # Prepend to maintain order from input to output
        path_indices.insert(0, current_layer_indices)

        # Update the outidx for the next iteration to move backwards through layers
        current_outidx = current_layer_indices

    path_indices.append(outidx)

    return path_indices


def get_ids(paths, idx_to_group, root_to_group):
    """Get the root ids at each step of the path between previously specified input and output. 

    Args:
        paths (list): A list of np.ndarray, each array contains the indices of neurons traversed for that path length. 
        idx_to_group (dict): A dictionary that maps the indices to group. 
        root_to_group (dict): A dictionary that maps the root ids to group. `idx_to_group` and `root_to_group` are used to map the indices to the neuron ids. 
    """

    types = [[idx_to_group[key] for key in array] for array in paths]

    # get the ids
    for i, collection in enumerate(types):
        if i == 0:
            print('From: ')
        else:
            print(str(i) + ' steps away from input')
        print(collection)
        ids = [str(key)
               for key, v in root_to_group.items() if v in collection]
        print(','.join(set(ids)))
