# Standard library imports
import numpy as np
import pickle as pkl
import pandas as pd
import os

# Third-party package imports
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import shortest_path


def compute_flow_hitting_time(
        conn_df: pd.DataFrame, 
        flow_seed_idx: np.ndarray[int], 
        flow_steps: int, 
        flow_thre: float
):
    """
    Compute hitting time for all cells in conn_df.
    Hitting time is the average number of hops required to reach a cell from a set of seed cells.
    The main algorithm is implemented in the 'navis' library.
    Args:
        conn_df (pd.DataFrame): DataFrame containing the connections with columns 'idx_pre', 'idx_post', and 'rel_in_weight'.
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
        raise ImportError("The 'navis' library is required for computing information flow.") from e

    # choose threshold for flow activation and define model
    def my_act(x):
            return linear_activation_p(x, min_w=0, max_w=flow_thre)
    edges = conn_df[['idx_pre', 'idx_post', 'rel_in_weight']]
    edges.columns = ['source', 'target', 'weight']
    model = BayesianTraversalModel(edges, flow_seed_idx, max_steps=flow_steps+1, traversal_func=my_act)

    res = model.run()

    # compute hitting times from cmf (cumulative mass function)
    cmf_arr = np.stack(res['cmf'].values)
    hitting_prob = np.zeros(cmf_arr.shape)
    hitting_prob[:,1:] = np.diff(cmf_arr, axis=1)
    hitting_time = np.dot(hitting_prob, np.arange(0, flow_steps+1))

    # cmf of unreached and seed cell types is 0
    # change hitting time of unreached cell types to flow_steps
    idx_unreached = np.where( (hitting_time < 0.1) &\
                              (~np.isin(res['node'], flow_seed_idx)) )[0]
    if len(idx_unreached)>0:
        hitting_time[idx_unreached] = flow_steps

    flow_df = pd.DataFrame({'idx': res['node'].values, 'hitting_time': hitting_time})

    return flow_df


def trim_inprop_by_flow(
        inprop: csc_matrix, 
        meta: pd.DataFrame, 
        file_path: str = None, 
        save_prefix: str = 'flow_', 
        flow_seed_types: list[str] = ['L1','L2','L3','R7p','R8p','R7y','R8y','R7d','R8d','HBeyelet'], 
        flow_steps: int = 20, 
        flow_thre: float = 0.1, 
        flow_diff_min: float = 0.5, 
        flow_diff_max: float = 20
):
    """
    Trim connections based on hitting time assigned by information flow algorithm (navis).

    Args:
        inprop (csc_matrix): Input sparse matrix representing connections.
        meta (pd.DataFrame): DataFrame containing metadata with 'bodyId' and 'cell_type'.
        file_path (str): Path to the directory containing the hitting time data.
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

    # convert inprop to dataframe
    coo = inprop.tocoo()
    conn_df = pd.DataFrame({'idx_pre': coo.row, 'idx_post': coo.col, 'rel_in_weight': coo.data})

    # load hitting time or compute
    flow_file_name = os.path.join(file_path, f'{save_prefix}{flow_steps}step_{flow_thre}thre_hit.p')
    if os.path.exists(flow_file_name):
        flow_df = pd.read_csv(flow_file_name)
    else:
        flow_seed_idx = meta[np.isin(meta['cell_type'],flow_seed_types)]['idx'].values
        flow_df = compute_flow_hitting_time(conn_df, flow_seed_idx, flow_steps, flow_thre)
        flow_df = flow_df.merge(meta, how='inner', on='idx')
        flow_df.to_csv(flow_file_name, index=False)

    # add hitting time to conn_df
    conn_flow_df = conn_df.merge(flow_df[['idx', 'hitting_time']], how='inner', left_on='idx_post', right_on='idx').drop(\
                            columns=['idx']).rename(columns={'hitting_time': 'hitting_time_post'})
    conn_flow_df = conn_flow_df.merge(flow_df[['idx', 'hitting_time']], how='inner', left_on='idx_pre', right_on='idx').drop(\
                            columns=['idx']).rename(columns={'hitting_time': 'hitting_time_pre'})

    # remove connections for which difference in hitting time is too small or too large
    conn_flow_df = conn_flow_df[(conn_flow_df['hitting_time_post']-conn_flow_df['hitting_time_pre'] > flow_diff_min) &\
                             (conn_flow_df['hitting_time_post']-conn_flow_df['hitting_time_pre'] <= flow_diff_max)]
    inprop_flow = csc_matrix((conn_flow_df['rel_in_weight'], (conn_flow_df['idx_pre'], conn_flow_df['idx_post'])), \
                             shape=inprop.shape)

    return inprop_flow