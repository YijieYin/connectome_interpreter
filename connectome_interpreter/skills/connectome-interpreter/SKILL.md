---
name: connectome-interpreter
description: Analyse Drosophila connectomes (FAFB/FlyWire, maleCNS, BANC, incl. axon-dendrite split) with the connectome_interpreter Python package - effective (polysynaptic, signed) connectivity, path-finding between neuron groups, receptive fields of a neuron in terms of functionally characterised cell types / odours / visual columns, a differentiable rate model, activation maximisation, saliency, and visualisation (wiring diagrams, hex eye maps, neuroglancer). Use whenever someone asks how neurons/cell types are (indirectly) connected, what drives or is driven by a neuron, what a neuron's receptive field or function might be from the connectome, or uses `import connectome_interpreter` / `coin`.
---

# Connectome Interpreter

`connectome_interpreter` (import as `coin`) turns a weighted, signed wiring diagram into testable hypotheses about circuit function. Docs: https://connectome-interpreter.readthedocs.io. Source + Colab tutorials: https://github.com/YijieYin/connectome_interpreter. Preprint: https://www.biorxiv.org/content/10.1101/2025.09.29.679410v2.

## The idea: link the known to the unknown

Neuroscience interprets uncharacterised circuits through characterised ones. The package formalises this as the **contextual receptive field (RFc)**: a neuron's receptive field expressed not only in peripheral sensory neurons, but in *any* functionally characterised neurons anywhere in the brain (e.g. "excited by appetitive MBONs, inhibited by aversive ones", "driven by looming-sensitive LPLC1 on the contralateral side", "tuned to these odours").

Anchors for "known":
- **Known function** - community table of cell types with experimentally tested functions, each entry with references (https://tinyurl.com/known-neuron-function). A literature lookup, not a stipulation of function.
- **Stimulus tuning** - bundled odour-response datasets (`coin.load_dataset`), mapped onto downstream neurons.
- **Sensory space** - optic-lobe columnar coordinates, rendered as hex eye maps / Mollweide projections.

Report findings as hypotheses with the evidence chain (which known neurons, path length, sign, strength), and cite the source papers from the known-function table.

## Workflow

1. **Load** a connectome (sparse `inprop` matrix, pre in rows, post in columns, input-normalised) + metadata. See [references/data.md](references/data.md).
2. **RFc via effective connectivity** - how strongly (and with what sign) do known groups influence the target across 1..n hops? Decompose by path length to see *which* hop counts matter. See [references/linear.md](references/linear.md).
3. **Path-finding** - for the hop counts that matter, retrieve the actual subcircuit (intermediate neurons, edges), group by cell type, filter, plot. Also [references/linear.md](references/linear.md).
4. **Non-linear model** (when needed) - inject stimuli into `MultilayeredNetwork`; read out activity, active pathways; saliency for which inputs/intermediates mattered; activation maximisation for which input best drives a target; `train_model` to fit to data. See [references/model.md](references/model.md).

Start linear (cheap, interpretable); go non-linear when the questions below apply.

## Choosing between linear and non-linear

Effective connectivity = sum over all paths of length n of the product of weights = n-th power of the connectivity matrix; it equals the steady state of a *linear* rate model. It is the average influence, independent of state. The non-linear model (`tanh(relu(.))`) adds context-dependence. They diverge in two regimes:

- **Saturation (tanh)**: stronger input can *lower* output. E.g. input drives an excitatory and an inhibitory interneuron onto a target; if the excitatory one saturates first, more input only adds inhibition.
- **Non-negativity (ReLU)**: net effective inhibition can *fail to inhibit*. If an interneuron on the inhibitory route is itself net-inhibited, it is silenced at 0 and cannot pass the inhibition on, so the target may be net-excited where signed effective connectivity predicts inhibition.

Use the model when the question is about a specific stimulus/state, when many inhibitory routes are involved, or when asking "what input best drives X". Saliency in a linear model equals signed effective connectivity; in the non-linear model it predicts perturbation responses better, increasingly so at longer path lengths.

## Interpretation rules (from whole-brain analyses)

- **The connectome is dense.** In FAFB, ~70% of random cell-type pairs connect via one intermediate at threshold 0; at 5 hops essentially all pairs connect. Don't conclude "not connected" from direct connectivity alone; check several path lengths.
- **Don't threshold away weak connections by default.** Most connections are weak, yet many weak inputs from cells of the same type can together form a large share of a target's input. Threshold for *display* (convention: 1% of the target's input, i.e. weight 0.01, stronger than ~85% of connections), not for computing influence. When thresholding every edge in a path, shorter paths win; without threshold, effective connectivity is comparable across lengths 2-5 because path count grows ~2-3 orders of magnitude per hop and offsets per-hop decay. So don't default to shortest paths.
- **Loops** (paths revisiting a neuron) are kept by default; they usually contribute little. Use `enumerate_paths(loop_mode=...)` / `effconn_without_loops` to exclude them.
- **Sign**: excitatory/inhibitory comes from predicted neurotransmitter of the *presynaptic* neuron (`sign` column, +1/-1). Two inhibitory steps count as excitation (disinhibition), which only works if the intermediate is active - a reason to check with the model.
- **Raw effective connectivity shrinks with path length** (products of weights in [0,1]) - compare within a path length, or use `root=True` ("equivalent per-hop strength") knowing rooted values *grow* with length because of convergence.
- **Group by cell type and side** (`type_side`) for interpretable results; laterality matters (e.g. LPLC1 excites the *contralateral* DNa05).
- **Axon-dendrite split** connectomes remove axo-axonic and dendro-dendritic connections; use them when signal-flow direction matters. Effective connectivity then decreases more with path length.
- Weights are input proportions: column sums are <= 1, and the denominator is the post neuron's *total* input. Subsetting the connectome before normalising changes meaning; keep whole-brain normalisation.

## Compute environment

Runs on laptop, workstation, HPC, or Colab; GPU accelerates matrix multiplication and the model but is optional.
- Device: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. Note `compress_paths` defaults `device=cuda`; pass the device explicitly on CPU machines.
- Whole-brain all-to-all effective connectivity: `compress_paths(inprop, n, chunkSize=..., save_to_disk=True, save_path=...)`. Memory ~ sparse matrix + a few dense chunks of `chunkSize x n_neurons` float32 (2000 x 140k ~ 1.1 GB each). Lower `chunkSize` on small RAM/GPU. Precompute once (HPC/GPU), save `.npz` per step, reload with `coin.read_precomputed`.
- For a specific source/target question, prefer the targeted functions (`conn_by_path_length`, `find_paths_of_length` + `effective_conn_from_paths`): seconds on a laptop, no precompute needed.
- In scripts (HPC, no notebook), set `display_output=False`, save plotly figures with `fig.write_html(...)`, and `plot_paths(..., interactive=False, save_plot=True, show=False)`.
- Optional extras: `pip install "connectome_interpreter[get_ngl_link]"` (neuroglancer links), `[compute_flow_hitting_time]` (navis, information-flow layers), `[plot_layered_paths]` (pyvis), `[wandb]`.

## Getting help on a function

Signatures and docstrings are thorough: `help(coin.find_paths_of_length)`. When unsure of an argument, read the docstring rather than guessing.
