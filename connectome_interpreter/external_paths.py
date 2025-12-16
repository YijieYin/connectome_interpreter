# Standard library imports
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from typing import Dict, Union, Optional

# Third-party package imports
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm

from .utils import to_nparray, arrayable
from .path_finding import group_paths, filter_paths, find_paths_of_length
from .compress_paths import effective_conn_from_paths


def compute_flow_hitting_time(
    conn_df: Union[pd.DataFrame, spmatrix],
    flow_seed_idx: np.ndarray[int],
    flow_steps: int,
    flow_thre: float,
):
    """
    Compute hitting time for all cells in conn_df.
    Hitting time is the average number of hops required to reach a cell from a set of seed cells.
    The main algorithm is implemented in the 'navis' library (https://github.com/navis-org/navis).

    Args:
        conn_df (pd.DataFrame): DataFrame containing the connections with columns 'pre', 'post', and 'weight'.
        flow_seed_idx (np.ndarray): Array of seed cell indices.
        flow_steps (int): Number of steps for flow calculation.
        flow_thre (float): Threshold for activation in flow calculation.

    Returns:
        pd.DataFrame: DataFrame with columns 'idx' and 'hitting_time',
        where 'idx' is the cell index and 'hitting_time' is the computed hitting time.
    """

    try:
        from navis.models import BayesianTraversalModel, linear_activation_p
    except ImportError as e:
        raise ImportError(
            "The 'navis' library is required for computing information flow."
        ) from e

    # choose threshold for flow activation and define model
    def my_act(x):
        return linear_activation_p(x, min_w=0, max_w=flow_thre)

    if isinstance(conn_df, spmatrix):
        # convert sparse matrix to DataFrame
        coo = conn_df.tocoo()
        conn_df = pd.DataFrame({"pre": coo.row, "post": coo.col, "weight": coo.data})
    elif isinstance(conn_df, pd.DataFrame):
        # make sure the column names are correct
        if not all(col in conn_df.columns for col in ["pre", "post", "weight"]):
            raise ValueError(
                "Input DataFrame must contain columns 'pre', 'post', and 'weight'."
            )
    else:
        raise TypeError(
            "Input must be a sparse matrix or a DataFrame with columns 'pre', 'post', and 'weight'."
        )

    print(
        f"Computing hitting time for {len(set(conn_df.pre)|set(conn_df.post))} cells... May take a while."
    )
    edges = conn_df[["pre", "post", "weight"]]
    edges.columns = ["source", "target", "weight"]
    model = BayesianTraversalModel(
        edges, flow_seed_idx, max_steps=flow_steps + 1, traversal_func=my_act
    )

    res = model.run()

    # compute hitting times from cmf (cumulative mass function)
    cmf_arr = np.stack(res["cmf"].values)
    hitting_prob = np.zeros(cmf_arr.shape)
    hitting_prob[:, 1:] = np.diff(cmf_arr, axis=1)
    hitting_time = np.dot(hitting_prob, np.arange(0, flow_steps + 1))

    # cmf of unreached and seed cell types is 0
    # change hitting time of unreached cell types to flow_steps
    idx_unreached = np.where(
        (hitting_time < 0.1) & (~np.isin(res["node"], flow_seed_idx))
    )[0]
    if len(idx_unreached) > 0:
        hitting_time[idx_unreached] = flow_steps

    flow_df = pd.DataFrame({"idx": res["node"].values, "hitting_time": hitting_time})

    return flow_df


def find_instance_flow(
    inprop: Union[spmatrix, pd.DataFrame],
    idx_to_group: dict,
    flow_seed_groups: list[str] = [
        "L1",
        "L2",
        "L3",
        "R7p",
        "R8p",
        "R7y",
        "R8y",
        "R7d",
        "R8d",
        "HBeyelet",
    ],
    file_path: str = None,
    save_flow: Optional[bool] = True,
    save_prefix: Optional[str] = "flow_",
    flow_steps: int = 20,
    flow_thre: float = 0.1,
) -> pd.DataFrame:
    """
    Get the hitting time for all cell groups. The hitting time is computed using the
    information flow algorithm (navis:https://github.com/navis-org/navis) for each
    neuron, and then taking the median across neurons of each cell group.

    Args:
        inprop (Union[spmatrix, pd.DataFrame]): Input sparse matrix or DataFrame
            representing connections. If a DataFrame, it should have columns 'pre',
            'post', and 'weight'.
        idx_to_group (dict): Dictionary mapping cell indices to their respective cell
            groups.
        flow_seed_groups (list): List of cell groups to be used as seeds for flow
            calculation.
        file_path (str): Path to the directory containing the hitting time data.
        save_flow (bool): Whether to save the computed hitting time to a CSV file.
            Defaults to True.
        save_prefix (str): Prefix for the saved file names.
        flow_seed_groups (list): List of cell group to be used as seeds for flow
            calculation.
        flow_steps (int): Number of steps for flow calculation.
        flow_thre (float): Threshold for flow calculation.

    Returns:
        pd.DataFrame: DataFrame with columns 'cell_group' and 'hitting_time', where
        'cell_group' is the name of the cell group and 'hitting_time' is the median
        hitting time for that group.
    """

    if file_path is None:
        # Check if running in Google Colab
        if "COLAB_GPU" in os.environ:
            # Running in Colab
            file_path = "/content/"
        else:
            # Running locally
            file_path = "./"

    # load hitting time or compute
    flow_file_name = os.path.join(
        file_path, f"{save_prefix}{flow_steps}step_{flow_thre}thre_hit.csv"
    )
    if os.path.exists(flow_file_name):
        flow_df = pd.read_csv(flow_file_name)
        if "cell_group" not in flow_df.columns:
            flow_df["cell_group"] = flow_df["idx"].map(idx_to_group)
    else:
        # if file_path doesn't exist, create it
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # get indices whose value is in flow_seed_groups
        flow_seed_idx = np.array(
            [
                idx
                for idx, cell_type in idx_to_group.items()
                if cell_type in flow_seed_groups
            ]
        )
        if len(flow_seed_idx) == 0:
            raise ValueError(
                f"No flow seed groups found in the provided idx_to_group mapping. "
                f"Please check the flow_seed_groups: {flow_seed_groups}."
            )
        flow_df = compute_flow_hitting_time(
            inprop, flow_seed_idx, flow_steps, flow_thre
        )
        # add cell group to flow_df
        # note: this only includes indices in flow_seed_idx, which could in theory be a
        # subset of cells in a cell type
        flow_df["cell_group"] = flow_df["idx"].map(idx_to_group)
        if save_flow:
            print(f"Saving hitting time to {flow_file_name}")
            flow_df.to_csv(flow_file_name, index=False)

    flow_type_df = (
        flow_df.groupby("cell_group")["hitting_time"]
        .median()
        .to_frame()
        .reset_index()
        .sort_values("hitting_time")
    )
    flow_type_file_name = os.path.join(
        file_path, f"{save_prefix}{flow_steps}step_{flow_thre}thre_hit_per_group.csv"
    )
    if not os.path.exists(flow_type_file_name):
        if save_flow:
            print(f"Saving flow group hitting time to {flow_type_file_name}")
            flow_type_df.to_csv(flow_type_file_name, index=False)

    return flow_type_df


def layered_el(
    inprop: spmatrix,
    inidx: arrayable,
    outidx: arrayable,
    n: int,
    idx_to_group: dict,
    thre_cumsum: float | None=None,
    thre_step_min: float = 0.0,
    combining_method: str = "mean",
    flow_steps: int = 20,
    flow_thre: float = 0.1,
    flow: pd.DataFrame | None = None,
):
    """
    Similar to `el_within_n_steps` but using filter_paths_to_cumsum and layers based
    on the information flow hitting time (navis: https://github.com/navis-org/navis).
    If thre_cumsum is None, then paths are filtered based on direct weight thre_step_min;
    otherwise, paths are filtered such that the cumulative effective weight reaches thre_cumsum.

    Args:
        inprop (spmatrix): Input sparse matrix representing connections.
        inidx (np.ndarray): Array of input indices.
        outidx (np.ndarray): Array of output indices.
        n (int): The maximum number of hops. n=1 for direct connections.
        idx_to_group (dict): Dictionary mapping cell indices to their respective cell
            groups.
        thre_cumsum (float): The cumulative effective weight threshold to reach for filtered paths.
            Defaults to None, in which case paths are only filtered based on thre_step_min.
        thre_step_min (float, optional): The minimum threshold for the weight of the
            direct connection between pre and post. Defaults to 0.0.
        combining_method (str, optional): Method to combine inputs (outprop=False) or
            outputs (outprop=True). Can be 'sum', 'mean', or 'median'. Defaults to 'mean'.
        threshold (float): The threshold for the weight of the direct connection between
            pre and post, after grouping by `idx_to_group`. Defaults to 0.
        flow_steps (int): Number of steps for flow calculation. Defaults to 20.
        flow_thre (float): Threshold for flow calculation. Defaults to 0.1.
        flow (pd.DataFrame, optional): DataFrame containing the flow hitting time.
            If provided, it should have columns 'cell_group' and 'hitting_time'.
            If None, the flow hitting time is computed from `inprop` and `idx_to_group`.
    Returns:
        pd.DataFrame: DataFrame containing the grouped edge list with flow layers.
    """

    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    # get edge list, both grouped by idx_to_group, and with raw indices
    all_paths = []
    raw_el = []
    for i in tqdm(range(n)):
        paths = find_paths_of_length(inprop, inidx, outidx, i + 1)
        if paths is None or paths.shape[0] == 0:
            continue
        raw_el.append(paths)
        paths = group_paths(
            paths,
            idx_to_group,
            idx_to_group,
            combining_method=combining_method,
        )
        if paths is None or paths.shape[0] == 0:
            continue
        all_paths.append(paths)
    if len(all_paths) == 0:
        return None, 0, 0

    if thre_cumsum is None:
        w_all = 0
        w_filter = 0
        for i in range(len(all_paths)):
            thre_step = thre_step_min
            w_all_i = effective_conn_from_paths(all_paths[i])
            w_all += w_all_i.sum().sum()
            all_paths[i] = filter_paths(all_paths[i], thre_step_min)
            if all_paths[i] is None or all_paths[i].shape[0] == 0:
                continue
            w_filter_i = effective_conn_from_paths(all_paths[i])
            w_filter += w_filter_i.sum().sum()
    else:
        all_paths, w_filter, w_all, thre_step = filter_all_paths_to_cumsum(all_paths, thre_cumsum, thre_step_min)

    all_paths = pd.concat(all_paths, axis=0)
    grouped = all_paths.groupby(["pre", "post"])["weight"].max().reset_index()
    raw_el = pd.concat(raw_el, axis=0)
    # pre, post are single neurons, so the rows should be real duplicates
    raw_el = raw_el.drop_duplicates(subset=["pre", "post", "weight"])

    # information flow
    if flow is None:
        rawel["pre_type"] = rawel.pre.map(idx_to_group)
        rawel["post_type"] = rawel.post.map(idx_to_group)
        # only keep the neurons in the types that pass the threshold
        rawel = rawel[rawel.pre_type.isin(grouped.pre) & rawel.post_type.isin(grouped.post)]

        # compute flow hitting time
        flow = find_instance_flow(
            rawel,
            idx_to_group,
            set([grp for idx, grp in idx_to_group.items() if idx in inidx]),
            save_flow=False,
            flow_steps=flow_steps,
            flow_thre=flow_thre,
        )
    else:
        # use provided flow hitting time
        if not isinstance(flow, pd.DataFrame):
            raise TypeError("Flow must be a pandas DataFrame.")
        if "cell_group" not in flow.columns or "hitting_time" not in flow.columns:
            raise ValueError(
                "Flow DataFrame must contain 'cell_group' and 'hitting_time' columns."
            )
    type_layer = dict(zip(flow.cell_group, flow.hitting_time))

    grouped["pre_layer"] = grouped.pre.map(type_layer)
    grouped["post_layer"] = grouped.post.map(type_layer)
    #neurons that were not reached are assigned the layer flow_steps
    grouped.fillna(flow_steps, inplace=True)

    return grouped, w_filter, w_all, thre_step


def find_shortest_paths(
    paths: pd.DataFrame, start_nodes: list[str], end_nodes: list[str]
) -> list[list[str]]:
    """
    Find the shortest paths between groups in start_nodes and end_nodes
    in a paths dataframe (paths is the output of find_path_iteratively).

    Args:
        paths (pd.DataFrame): DataFrame containing the path data, including
            columns 'weight', 'pre', and 'post'.
        start_nodes (list): List of 'pre' groups.
        end_nodes (list): List of 'post' groups.

    Returns:
        list: A list of shortest paths, where each path is a list of groups
            that connect the start and end nodes (ordered from start to end).
    """

    paths_unique = paths[["weight", "pre", "post"]].drop_duplicates()
    nodes_unique = np.unique(
        np.concatenate([paths_unique.pre.unique(), paths_unique.post.unique()])
    )
    pre = np.searchsorted(nodes_unique, paths_unique.pre.values)
    post = np.searchsorted(nodes_unique, paths_unique.post.values)
    graph_matrix = csr_matrix(
        (1 / paths_unique["weight"].values, (pre, post)),
        shape=(len(nodes_unique), len(nodes_unique)),
    )

    _, predecessors = shortest_path(
        csgraph=graph_matrix, directed=True, return_predecessors=True
    )

    def reconstruct_single_path(predecessors, start, end, node_names):
        """
        Helper function that reconstructs the path from start to end
        using the predecessors array.
        """
        idx_start = np.where(nodes_unique == start)[0]
        idx_end = np.where(nodes_unique == end)[0]
        if len(idx_start) == 0 or len(idx_end) == 0:
            return None
        path = [node_names[idx_end[0]]]
        i = idx_end[0]
        while i != idx_start[0]:
            i = predecessors[idx_start[0], i]
            if i == -9999:  # not reached
                return None
            path.append(node_names[i])
        return path[::-1]  # reverse

    shortest_paths = []
    for start_node in start_nodes:
        for end_node in end_nodes:
            if start_node != end_node:
                path = reconstruct_single_path(
                    predecessors, start_node, end_node, nodes_unique
                )
                if path is not None:
                    shortest_paths.append(path)

    return shortest_paths


def effective_conn_per_path_from_paths(
        paths_df: pd.DataFrame,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the total effective weight of each path of fixed length. 
    Beware that this might be slow for long paths.

    Args:
        paths_df (pd.DataFrame): DataFrame containing the path data with columns
            'layer', 'pre', 'post', and 'weight'.

    Returns:
        total_effective_weight (float): Total effective weight of all paths
            ending in target group.
        all_path_weights (np.array): Array of weights for each individual path.
        min_path_weights (np.array): Array of minimum edge weights for each individual path.
    """
    if paths_df.empty:
        return 0.0, [], []
    
    # Convert to set for fast membership tests
    max_layer = int(paths_df['layer'].max())
    target_set = set(paths_df[paths_df['layer']==max_layer]['post'].unique())

    # Ensure columns are the right dtype for merging speed
    paths_df = paths_df[['layer', 'pre', 'post', 'weight']].copy()
    paths_df['layer'] = paths_df['layer'].astype(int)

    # Prepare a mapping from layer -> DataFrame of edges
    layer_edges = {l: df for l, df in paths_df.groupby('layer')}

    # Start from input layer (layer 1 presynaptic neurons)
    current_paths = layer_edges[1][['pre', 'post', 'weight']].copy()
    current_paths.rename(columns={'pre': 'path_start', 'post': 'path_end'}, inplace=True)
    current_paths['path_weight'] = current_paths['weight']
    current_paths['min_edge_weight'] = current_paths['weight'] 

    # Iterate layer by layer (forward chaining)
    for layer in range(2, max_layer + 1):
        if layer not in layer_edges:
            break
        
        next_edges = layer_edges[layer][['pre', 'post', 'weight']]
        # Join current path ends to next layer presynaptic neurons
        merged = current_paths.merge(
            next_edges, left_on='path_end', right_on='pre', how='inner', suffixes=('', '_next')
        )

        if merged.empty:
            break

        # Update path info
        merged['path_end'] = merged['post']
        merged['path_weight'] = merged['path_weight'] * merged['weight_next']
        merged['min_edge_weight'] = merged[['min_edge_weight', 'weight_next']].min(axis=1)

        current_paths = merged[['path_start', 'path_end', 'path_weight', 'min_edge_weight']].copy()
        current_paths['weight'] = current_paths['path_weight']

    # Only keep paths that end in target neurons
    final_paths = current_paths[current_paths['path_end'].isin(target_set)]
    if final_paths.empty:
        return 0.0, [], []

    # Compute relevant effective weights
    min_path_weights = final_paths['min_edge_weight'].to_numpy()
    all_path_weights = final_paths['path_weight'].to_numpy()
    total_effective_weight = all_path_weights.sum()

    return total_effective_weight, all_path_weights, min_path_weights


def filter_all_paths_to_cumsum(
    all_paths: list[pd.DataFrame],
    thre_cumsum: float,
    thre_step_min: float = 0.0,
    necessary_intermediate: Dict[int, arrayable] | None = None,
):
    """
    Filters paths such that intermediate neurons are specified in necessary_intermediate and the
    cumulative effective weight minimally reaches thre_cumsum of the total effective weight
    and the minimum edge weight across the selected paths is either the minimum along those
    remaining paths or thre_step_min.

    Args:
        paths (list[pd.DataFrame]): The list of DataFrames containing the path data, 
            where each DataFrame is like the output from `find_paths_of_length()`.
        thre_cumsum (float): The cumulative effective weight threshold to reach for filtered paths.
        thre_step_min (float, optional): The minimum threshold for the weight of the
            direct connection between pre and post. Defaults to 0.0.
        necessary_intermediate (dict, optional): A dictionary of necessary
            intermediate neurons, where the keys are the layer numbers
            (starting neurons: 1; directly downstream: 2) and the values are
            the neuron indices (can be int, float, list, set, numpy.ndarray,
            or pandas.Series). Defaults to None.

    Returns:
        paths (list[pd.DataFrame]): List of DataFrames containing the paths that meet the criteria.
        w_filter (float): The total effective weight of the filtered paths.
        w_all (float): The total effective weight of all paths before filtering.
        thre_step (float): The minimum edge weight threshold used to filter paths which is
            minimally thre_step_min.
    """
    # compute total effective weight across all paths
    w_all = 0.0
    w_prop = []
    w_min = []
    for i in range(len(all_paths)):
        w_all_i, w_prod_i, w_min_i = effective_conn_per_path_from_paths(all_paths[i])
        w_all += w_all_i
        w_prop.append(w_prod_i)
        w_min.append(w_min_i)
    w_prod = np.concatenate(w_prop, axis=0)
    w_min = np.concatenate(w_min, axis=0)

    # find minimum edge weight across strongest paths that make up thre_cumsum of total weight
    # since in filter_paths, threshold > thre_step, take 99.9%, with a minimum of thre_step_min
    idx_sort = np.argsort(-w_prod)
    idx_thre = np.where(np.cumsum(w_prod[idx_sort]/w_all) > thre_cumsum)[0][0]+1
    thre_step = max(0.999 * np.min(w_min[idx_sort[:idx_thre]]), thre_step_min)

    # filter paths and compute total effective weight across those paths
    w_filter = 0.0
    for i in range(len(all_paths)):
        all_paths[i] = filter_paths(all_paths[i], thre_step, necessary_intermediate)
        w_filter_i, _, _ = effective_conn_per_path_from_paths(all_paths[i])
        w_filter += w_filter_i

    return all_paths, w_filter, w_all, thre_step
