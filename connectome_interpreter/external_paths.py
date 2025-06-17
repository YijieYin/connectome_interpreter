# Standard library imports
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from typing import Optional, Union

# Third-party package imports
from scipy.sparse import csc_matrix, csr_matrix, spmatrix
from scipy.sparse.csgraph import shortest_path
import networkx as nx

from .utils import to_nparray, el_within_n_steps, arrayable


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
        # map idx to cell group
        flow_df["cell_group"] = flow_df["idx"].map(idx_to_group)
    else:
        print(
            f"Computing hitting time for {len(idx_to_group)} cells... May take a while."
        )
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


def trim_inprop_by_flow(
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
    flow_diff_min: float = 0.5,
    flow_diff_max: float = 20,
) -> csc_matrix:
    """
    Trim connections based on hitting time assigned by information flow algorithm
    (navis: https://github.com/navis-org/navis). The hitting time is the mean
    number of hops required to reach a neuron from a neuron in flow_seed_groups.
    If the hitting time of the post neuron is larger than that of the pre neuron
    (i.e., the difference is between flow_diff_min and flow_diff_max), then the
    connection is interpreted as a feedforward connection and kept. For similar
    hitting times, the connection is interpreted as a lateral connection.
    If the hitting time of the pre neuron is larger than that of the post neuron,
    the connection is interpreted as a feedback connection.
    Lateral and feedback connections are removed.

    Args:
        inprop (csc_matrix): Input sparse matrix representing connections.
        idx_to_group (dict): Dictionary mapping cell indices to their respective cell groups.
        flow_seed_groups (list): List of cell groups to be used as seeds for flow calculation.
        file_path (str): Path to the directory containing the hitting time data.
        save_flow (bool): Whether to save the computed hitting time to a CSV file.
        flow_steps (int): Number of steps for flow calculation.
        flow_thre (float): Threshold for flow calculation.
        flow_diff_min (float): Minimum difference in hitting time for connection retention.
        flow_diff_max (float): Maximum difference in hitting time for connection retention.

    Returns:
        csc_matrix: sparse matrix for which pairs of connections have hitting time
        within specified range.
    """

    if file_path is None:
        # Check if running in Google Colab
        if "COLAB_GPU" in os.environ:
            # Running in Colab
            file_path = "/content/"
        else:
            # Running locally
            file_path = ""

    flow_type_df = find_instance_flow(
        inprop,
        idx_to_group,
        flow_seed_groups,
        file_path,
        save_flow,
        save_prefix,
        flow_steps,
        flow_thre,
    )  # has columns 'cell_group' and 'hitting_time'
    # note: this includes all neurons of e.g. a cell type
    meta = pd.DataFrame(idx_to_group.items(), columns=["idx", "cell_group"])
    meta = meta.merge(flow_type_df, how="inner", on="cell_group")

    if isinstance(inprop, pd.DataFrame):
        conn_df = inprop[["pre", "post", "weight"]]
    elif isinstance(inprop, spmatrix):
        # convert sparse matrix to DataFrame
        coo = inprop.tocoo()
        conn_df = pd.DataFrame({"pre": coo.row, "post": coo.col, "weight": coo.data})

    # add hitting time to conn_df
    conn_flow_df = (
        conn_df.merge(
            meta[["idx", "hitting_time"]],
            how="inner",
            left_on="post",
            right_on="idx",
        )
        .drop(columns=["idx"])
        .rename(columns={"hitting_time": "hitting_time_post"})
    )
    conn_flow_df = (
        conn_flow_df.merge(
            meta[["idx", "hitting_time"]],
            how="inner",
            left_on="pre",
            right_on="idx",
        )
        .drop(columns=["idx"])
        .rename(columns={"hitting_time": "hitting_time_pre"})
    )

    # remove connections for which difference in hitting time is too small or too large
    conn_flow_df = conn_flow_df[
        (
            conn_flow_df["hitting_time_post"] - conn_flow_df["hitting_time_pre"]
            >= flow_diff_min
        )
        & (
            conn_flow_df["hitting_time_post"] - conn_flow_df["hitting_time_pre"]
            <= flow_diff_max
        )
    ]
    inprop_flow = csc_matrix(
        (
            conn_flow_df["weight"],
            (conn_flow_df["pre"], conn_flow_df["post"]),
        ),
        shape=inprop.shape,
    )

    return inprop_flow


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


def plot_flow_layered_paths(
    paths: pd.DataFrame,
    figsize: tuple = (10, 8),
    weight_decimals: int = 2,
    neuron_to_sign: dict | None = None,
    sign_color_map: dict = {1: "red", -1: "blue"},
    neuron_to_color: dict | None = None,
    edge_text: bool = True,
    node_text: bool = True,
    highlight_nodes: list[str] = [],
    interactive: bool = True,
    save_plot: bool = False,
    file_name: str = "layered_paths",
    label_pos: float = 0.7,
    default_neuron_color: str = "lightblue",
    default_edge_color: str = "lightgrey",
    node_size: int = 500,
) -> None:
    """
    Plots a directed graph of layered paths based on flow layers.
    Similar to the `plot_layered_paths` function, but the x-axis is defined by the
    flow layers of the nodes.

    Args:
        paths (pandas.DataFrame): A dataframe containing the columns 'pre', 'post',
            'weight', 'pre_layer', and 'post_layer'. Each row represents an edge in the
            graph. The 'pre' and 'post' columns refer to the source and target nodes,
            respectively, and 'weight' indicates the edge weight. 'pre_layer' and
            'post_layer' are the flow layers of the corresponding nodes.
        figsize (tuple, optional): A tuple indicating the size of the matplotlib figure.
            Defaults to (10, 8).
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
        The positions of the nodes are determined by a custom positioning function
            (`find_flow_positions`).
        This function requires the networkx library for graph operations and matplotlib
            for plotting. For interactive plots, it requires the pyvis library (where
            the node label has to be underneath the node).
    """

    def find_flow_positions(
        vertices: np.array, layers: np.array, height: float, width: float
    ) -> dict:
        """
        Find positions for the nodes depending on the layer flow.
        """
        layer_bins = np.arange(-0.5 / 2, np.ceil(max(layers)) + 0.5 / 2, 1 / 2)
        layer_labels = layer_bins[1:] + layer_bins[0]
        layer_binned = pd.cut(layers, bins=layer_bins, labels=layer_labels)

        layer_w = width / max(layers)
        xy_pos = np.zeros((len(vertices), 2))
        for layer_b in np.unique(layer_binned):
            layer = layers[layer_binned == layer_b]
            layer_vertices = vertices[layer_binned == layer_b]
            if len(layer_vertices) > 0:
                layer_h = height / len(layer_vertices)
                for index, v in enumerate(layer_vertices):
                    idx = np.where(vertices == v)[0][0]
                    xy_pos[idx, 1] = index * layer_h + 0.05
                    xy_pos[idx, 0] = layer[index] * layer_w - width / 2

        positions = dict(zip(vertices, xy_pos))

        return positions

    if paths.shape[0] == 0:
        raise ValueError("The provided DataFrame is empty.")

    path_df = paths.copy()

    # Rescale weights to be between 1 and 10
    weight_min = path_df.weight.min()
    weight_max = path_df.weight.max()
    if weight_max == weight_min:
        path_df["weight"] = 1
    else:
        path_df["weight"] = 1 + 9 * (path_df["weight"].values - weight_min) / (
            weight_max - weight_min
        )

    # Create the graph
    G = nx.from_pandas_edgelist(
        path_df,
        source="pre",
        target="post",
        edge_attr=True,
        create_using=nx.DiGraph(),
    )

    # Labels for nodes
    labels = {v: v for v in G.nodes()}

    # Generate positions
    layers = np.zeros(len(labels))
    for index, v in enumerate(labels):
        if v in path_df.pre.values:
            layers[index] = path_df[path_df.pre == v].pre_layer.values[0]
        elif v in path_df.post.values:
            layers[index] = path_df[path_df.post == v].post_layer.values[0]
    positions = find_flow_positions(np.array(list(labels.values())), layers, 1, 1)

    # Default color for nodes if not provided otherwise
    if neuron_to_color is None:
        neuron_to_color = {label: default_neuron_color for label in labels.values()}
    else:
        # Ensure neuron_to_color has all labels, even if not in the dictionary
        nodediff = set(labels.values()) - set(neuron_to_color.keys())
        if len(nodediff) > 0:
            neuron_to_color.update(dict.fromkeys(nodediff, default_neuron_color))

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

        # create pyvis graph
        print(f"Store interactive graph as {file_name}" + ".pdf")
        net2 = Network(
            directed=True, layout=False, notebook=True, cdn_resources="in_line"
        )
        net2.height = figsize[1] * 100
        net2.width = figsize[0] * 100
        net2.from_nx(G)

        node_colors_dict = dict(zip(G.nodes(), node_colors))
        edge_colors_dict = dict(zip(G.edges(), edge_colors))
        for v in net2.nodes:
            v["x"], v["y"] = positions.get(v["id"])
            v["x"] = v["x"] * net2.width
            v["y"] = v["y"] * net2.height
            if node_text:
                v["label"] = labels[v["id"]]
            else:
                v["label"] = ""
            v["color"] = node_colors_dict[v["id"]]
            v["size"] = node_size / 50
            v["font"] = {"size": 24}
            if labels[v["id"]] in highlight_nodes:
                v["font"] = {"face": "arial black"}
            else:
                v["font"] = {"face": "arial"}

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
        net2.set_options(
            """
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
        """
        )
        net2.show(str(file_name) + ".html", notebook=False)
        print(f"Interactive graph saved as {file_name}.html")

    else:

        fig, ax = plt.subplots(figsize=(9, 12))
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

        if node_text:
            # label highlighted and normal nodes separately
            bold_nodes = [n for n in G.nodes() if labels[n] in highlight_nodes]
            normal_nodes = [n for n in G.nodes() if labels[n] not in highlight_nodes]

            nx.draw_networkx_labels(
                G,
                pos=positions,
                labels={n: labels[n] for n in normal_nodes},
                font_family="arial",
                font_weight="normal",
                font_size=14,
                ax=ax,
            )
            nx.draw_networkx_labels(
                G,
                pos=positions,
                labels={n: labels[n] for n in bold_nodes},
                font_family="arial",
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


def layered_el(
    inprop: spmatrix,
    steps: list,
    inidx: arrayable,
    outidx: arrayable,
    n: int,
    idx_to_group: dict,
    threshold: float = 0,
    flow_steps: int = 20,
    flow_thre: float = 0.1,
):
    """
    First finds paths within `n` steps, given the `threshold` (applied to direct
    connections), grouped by `idx_to_group`, and adds flow layers to the edge list based
    on the information flow hitting time (navis: https://github.com/navis-org/navis).

    Args:
        inprop (spmatrix): Input sparse matrix representing connections.
        steps (list): A list of connectivity matrices, e.g. the result from
            `compress_paths()`.
        inidx (np.ndarray): Array of input indices.
        outidx (np.ndarray): Array of output indices.
        n (int): The maximum number of hops. n=1 for direct connections.
        idx_to_group (dict): Dictionary mapping cell indices to their respective cell
            groups.
        threshold (float): The threshold for the weight of the direct
            connection between pre and post, after grouping by `idx_to_group`. Defaults
            to 0.
    Returns:
        pd.DataFrame: DataFrame containing the grouped edge list with flow layers.
    """

    inidx = to_nparray(inidx)
    outidx = to_nparray(outidx)

    # get edge list, both grouped by idx_to_group, and with raw indices
    grouped, rawel = el_within_n_steps(
        inprop,
        steps,
        inidx,
        outidx,
        n,
        threshold,
        idx_to_group,
        idx_to_group,
        return_raw_el=True,
    )
    rawel["pre_type"] = rawel.pre.map(idx_to_group)
    rawel["post_type"] = rawel.post.map(idx_to_group)
    # only keep the neurons in the types that pass the threshold
    rawel = rawel[rawel.pre_type.isin(grouped.pre) & rawel.post_type.isin(grouped.post)]

    # information flow
    flow = find_instance_flow(
        rawel,
        idx_to_group,
        set([grp for idx, grp in idx_to_group.items() if idx in inidx]),
        save_flow=False,
        flow_steps=flow_steps,
        flow_thre=flow_thre,
    )
    type_layer = dict(zip(flow.cell_group, flow.hitting_time))

    grouped["pre_layer"] = grouped.pre.map(type_layer)
    grouped["post_layer"] = grouped.post.map(type_layer)
    grouped.fillna(0, inplace=True)
    return grouped
