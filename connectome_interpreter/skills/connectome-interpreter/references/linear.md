# Linear methods: effective connectivity and path-finding

Conventions: `inprop` is scipy sparse, pre in rows, post in columns. `inidx`/`outidx` are arrays of metadata `idx`. Group maps are dicts `idx -> group` (usually `idx_to_type` or `idx_to_type_side`). Default aggregation (`combining_method="mean"`): inputs are **summed** across presynaptic cells of a group and **averaged** across postsynaptic cells of a group, i.e. "total input from type A to an average B cell". Path length 1 = direct connection.

## Direct connectivity

```python
coin.connectivity_summary(inprop, inidx, outidx, inidx_map=idx_to_type_side,
                          outidx_map=idx_to_type_side, display_threshold=0.01)
```
Returns a DataFrame (pre groups in rows, post groups in columns); `display_output=False` in scripts. Also accepts an edgelist (`pre_id`, `post_id`, `weight`), a precomputed step matrix, or `outprop=True`.

## Effective connectivity for a specific question (no precompute)

**Which path lengths matter** (unsigned):
```python
coin.conn_by_path_length(inprop, inidx, outidx, n=5, inidx_map=idx_to_type_side)  # plotly line plot
coin.conn_by_path_length_heatmap(inprop, inidx, outidx, n=5, outidx_map=idx_to_type_side)
coin.conn_by_path_length_data(...)                                              # the data
```
Give exactly one of `inidx_map` / `outidx_map`.

**Signed** (excitation and inhibition per path length):
```python
exc, inh = coin.signed_conn_by_path_length_data(  # lists, one DataFrame per path length
       inprop, inidx, outidx, n=5,
        group_to_sign=type_side_to_sign, inidx_map=idx_to_type_side,
        outidx_map=idx_to_type_side, intermediate_map=idx_to_type_side)
# type_side_to_sign = dict(zip(meta.type_side, meta.sign))  - keys are group names, not idx
```
Net effect = excitation - inhibition. To summarise a target's RFc over many known sources (e.g. appetitive vs aversive MBONs -> DNs), compute net excitation per source group and path length, then sum within a functional category.

**From a path subgraph** (after `find_paths_of_length`, see below), exact for those paths:
```python
coin.effective_conn_from_paths(paths)                       # unsigned; wide DataFrame first-layer pre x last-layer post
exc, inh = coin.signed_effective_conn_from_paths(paths, idx_to_nt=idx_to_sign)  # or a 'sign' column in paths
coin.effconn_without_loops(paths, pre_group=..., post_group=...)  # excluding paths revisiting a neuron
```
For grouped paths (`group_paths` output), pass the group-level sign mapping. Use `wide=False` when both ends have thousands of distinct neurons.

## Whole-network effective connectivity (precompute)

```python
steps = coin.compress_paths(inprop, step_number=5, chunkSize=2000, device=device,
                            output_threshold=1e-4, save_to_disk=True, save_path="precomputed/",
                            save_prefix="fafb_step_", return_results=False)
steps = coin.read_precomputed("fafb_step_", file_path="precomputed/")   # list, steps[0] = direct
exc_steps, inh_steps = coin.compress_paths_signed(inprop, idx_to_sign, target_layer_number=5, chunkSize=2000)
```
- `steps[i]` is the all-to-all effective connectivity for path length i+1 (pre rows, post columns). Query with `connectivity_summary(steps[i], inidx, outidx, ...)` or `coin.result_summary(...)`, or `contribution_by_path_lengths(steps, inidx, outidx, ...)`. Sum steps (`add_first_n_matrices`) for "within n hops".
- `threshold` drops small values *during* multiplication (changes the result); `output_threshold` only sparsifies what is stored.
- `root=True` returns n-th roots (per-hop equivalent strength).
- Whole brain, 5 hops: minutes on a GPU, longer on CPU; chunking keeps memory bounded (see SKILL.md compute section). On HPC run as a batch job that saves `.npz` per step.

## Path-finding

Bidirectional BFS returns the full edge-induced subcircuit of a fixed length between source and target sets (not every simple path). Fast on a laptop, no threshold required.

```python
paths = coin.find_paths_of_length(inprop, inidx, outidx, target_layer_number=3)
# DataFrame: pre, post, layer (1 = edges leaving sources), weight (direct inprop). None if no path.
paths = coin.group_paths(paths, idx_to_type_side, idx_to_type_side, intermediate_group=idx_to_type_side)
paths = coin.filter_paths(paths, threshold=0.01)                 # drop weak edges, then prune dead ends
paths = coin.filter_paths(paths, necessary_intermediate={2: ["LPLC1_left"]})  # must pass through these at layer 2
# layer numbering for necessary_intermediate: sources = 1, directly downstream = 2, ...; values are whatever
# is in pre/post (group names after group_paths, idx before)
coin.plot_paths(paths, neuron_to_sign=type_side_to_sign, interactive=False)
```
- Multiple lengths: loop `target_layer_number` over 1..n; or `coin.el_within_n_steps(inprop, inidx, outidx, n, threshold=..., pre_group=..., post_group=...)` for all edges on paths within n hops.
- Strongest pathways only: `kept, w_kept, w_all = coin.filter_all_paths_to_cumsum(list_of_paths, thre_cumsum=0.5)` raises the edge threshold as far as possible while keeping >= `thre_cumsum` of the total effective weight (report `w_kept / w_all`); `coin.layered_el(inprop, inidx, outidx, n, idx_to_group, thre_cumsum=...)` also assigns continuous information-flow layers (needs `navis`) for combining path lengths on one axis.
- Explicit path lists: `coin.enumerate_paths(paths, loop_mode="allow"|"exclude")`, `coin.count_paths(...)`. Can explode combinatorially at long lengths; prefer the subgraph.
- After manually removing rows, call `coin.remove_excess_neurons(paths)` to prune dangling edges.
- `coin.find_xor(paths)` finds XOR-like motifs; `coin.find_shortest_paths(paths, start, end)`.
- `plot_paths` options: `sort_by_weight=True`, `priority_indices`, `highlight_nodes`, `neuron_to_color`, `interactive=True` (pyvis, notebook), `save_plot=True, file_name=...`, `show=False` for scripts. Edge labels are direct weights (input proportion of the post). Red/blue = excitatory/inhibitory when `neuron_to_sign` is given.

## Continuous layers (information flow)

`coin.compute_flow_hitting_time(inprop_or_edgelist, flow_seed_idx, flow_steps=20, flow_thre=0.1)` assigns each neuron a mean number of hops from seed neurons (Schlegel et al. 2021, via navis). Non-integer layers (e.g. 2.5 = equally likely reached in 2 or 3 hops) let pathways of different lengths share one axis, and classify connections as feedforward / lateral / feedback. A neuron's layer approximates the path length at which effective connectivity is most robust to thresholding.

## Neuroglancer and visual receptive fields

- `coin.get_ngl_link(df, ...)`: wide DataFrame with neuron IDs (root_id/bodyid) as index and columns as layers/timepoints/targets; colour by value. `source=` for non-FAFB datasets (default FAFB). Needs `nglscenes`. Show a pathway by layer: `coin.get_ngl_link(coin.path_for_ngl(paths_with_neuron_ids), df_format="long")` (map idx -> root_id in `pre`/`post` first); `coin.activated_path_for_ngl` for model activity.
- Visual RF of a target through a channel: find paths of length n from columnar inputs (e.g. L1-L3, R7, R8) to the target, optionally requiring a specific visual projection type as the last intermediate (`filter_paths(necessary_intermediate=...)`); `effective_conn_from_paths` per source neuron; map each source root_id to its column (`load_dataset("Matsliah2024")` for FAFB: `root_id -> coords`); sum per column; `coin.hex_heatmap(series, dataset="fafb_right")` (or `"mcns_right"`) and/or `coin.plot_mollweide_projection(series)`.
- `coin.display_df(df)` styled tables; `coin.compare_connectivity(...)` compares two connectomes/hemispheres.
