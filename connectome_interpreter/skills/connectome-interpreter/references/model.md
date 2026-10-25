# Non-linear model, saliency, activation maximisation, training

## The model

`MultilayeredNetwork` unrolls the whole connectome over time: every layer is a timepoint containing all neurons; signed connectome weights connect adjacent layers. Update (per neuron i):

r_i(t+1) = (tau-1)/tau * r_i(t) + 1/tau * tanh( relu( a_i * sum_j w_ji r_j(t) + b_i ) ) + I_i(t), clamped to [0, 1]

- `a` excitability = `tanh_steepness` (default 5; per group via `slope_dict`), `b` bias/baseline (`default_bias`, `bias_dict`), `tau` persistence (default 10, `tau_dict`; tau=1 means only current input matters), `threshold` (default 0.01) zeroes activity below it.
- ReLU = neurons cannot be negatively active; tanh = saturation. These two produce the divergences from linear effective connectivity (see SKILL.md).
- Parameters are shared within groups given by `idx_to_group` (e.g. `idx_to_type`); dicts map group name -> value.
- `divisive_normalization={pre_group: [post_groups]}` + `divisive_strength`: those inhibitory connections scale the post's slope instead of subtracting. Subtractive inhibition is already divisive in effect when it scales with the target's own activity (e.g. A -> B -> A feedback).
- `slope_dict` with `(pre_group, post_group)` tuple keys = per-connection gains (trainable).
- Steady state after an input pulse depends mainly on excitability, little on the stimulus or tau.

## Building and running

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sensory_indices = meta.idx[meta.super_class == "sensory"].values   # model inputs; adapt column/category per dataset
W = coin.add_sign(inprop, idx_to_sign)          # signs rows by presynaptic sign AND transposes (post rows) for torch
model = coin.MultilayeredNetwork(W, sensory_indices, num_layers=10, threshold=0.01,
                                 tanh_steepness=5, tau=1, idx_to_group=idx_to_type,
                                 device=device).to(device)

# input: (n_sensory, num_layers) or (batch, n_sensory, num_layers), ordered as sensory_indices
inp = np.zeros((len(sensory_indices), 10), dtype=np.float32); inp[:, 0] = ...  # pulse at t=0
acts = model(torch.tensor(inp, device=device))   # (neurons, layers) or (batch, neurons, layers)
acts = acts.detach().cpu().numpy()
```

- Inputs can be any neuron set (sensory neurons, visual projection neurons, MBONs, ...), the "optogenetic" view that enables contextual receptive fields.
- From a stimulus table: `inp = coin.input_from_df(df, sensory_indices, idx_to_group, num_layers, timepoints=[0, 1])` where df rows are groups (e.g. glomeruli, map ORN idx -> glomerulus), columns are stimuli (batches). E.g. `df = coin.load_dataset("DoOR_adult")`.
- Clamp neurons: `model(inp, manipulate={layer: {group_or_idx: value}})` (or per batch `{batch: {layer: {...}}}`), e.g. silence a cell type with 0.
- Long runs: `checkpoint_steps` trades compute for memory when gradients are needed.
- Stimulus builders: `coin.looming_stimulus(start_coords, all_coords, n_time)`, `coin.make_sine_stim(...)`.

Reading out:
```python
coin.get_neuron_activation(acts, neuron_indices, batch_names=..., idx_to_group=idx_to_type)  # tidy per group/time
coin.get_input_activation(inp, sensory_indices, idx_to_group, activation_threshold=0.1)
fig = coin.plot_timeseries(df)                                   # interactive, slider over batches
paths = coin.activations_to_df(inprop, inp, acts, sensory_indices, inidx_mapping=idx_to_type,
                               outidx_mapping=idx_to_type, activation_threshold=0.1,
                               connectivity_threshold=0.01)     # "paths that matter": active + strong edges
paths = coin.get_activations_for_path(paths_from_pathfinding, acts, model_in=inp,
                                      sensory_indices=sensory_indices, idx_to_group=idx_to_type)
coin.plot_paths(paths, sort_by_activation=True)                  # nodes coloured by activation
coin.activity_by_column(...) / coin.plot_activity_by_column(...)  # visual: activity per eye column
coin.get_ngl_link(df_of_activity_indexed_by_root_id)            # activity in neuroglancer
```
With many timepoints `activations_to_df` gets complex: restrict to a window or few targets.

## Saliency and gradients: which inputs/intermediates mattered

```python
sal = coin.saliency(model, inp_tensor, neurons_of_interest={layer: [idx, ...]}, method="vanilla")  # same shape as input
g = coin.get_gradients(model, inp_tensor, monitor_neurons=["DA1_lPN", ...], target_neurons={9: ["SMP108"]},
                       batch_names=[...], normalize_per_layer=False)   # group names if model has idx_to_group
```
- Gradient of target w.r.t. inputs *or any intermediate neuron*, in the current activity state. `"input_x_gradient"` weights by current activity.
- In a linear model saliency = signed effective connectivity; in the non-linear model it predicts perturbation effects better, especially at longer path lengths. tau=1 isolates per-path-length contributions (layer k back = path length k).
- Gradients can explode/vanish across many layers: `normalize_per_layer=True` for relative patterns.

## Activation maximisation: which input best drives a target

```python
targets = coin.TargetActivation({9: {target_idx: 1.0}})        # {layer: {neuron_idx: value}}; DataFrame
                                                               # (batch, layer, neuron, value) for per-batch targets
opt_in, out, act_loss, in_reg, out_reg, snapshots = coin.activation_maximisation(
    model, targets, num_iterations=200, learning_rate=0.1,
    in_reg_lambda=1e-6, out_reg_lambda=0, device=device, seed=0)
```
- Gradient descent on the input pattern; regularisers favour sparse inputs (`in_reg_lambda`) and sparse overall activity (`out_reg_lambda`) to yield a minimal circuit. In tests, larger learning rates and small regularisation gave the lowest activation loss; a small non-zero `in_reg_lambda` helped.
- Targets can be time-varying patterns, deactivations (value 0), or contrasts between groups/batches.
- Inputs need not be sensory: set e.g. MBONs as `sensory_indices` to find the MBON pattern that best drives a DN type (a contextual receptive field).
- `coin.guess_optimal_stimulus(inprop, sensory_indices, idx_to_sign, targets, longest_plen=5)` gives a linear (signed effective connectivity) guess; useful as `input_tensor=` initialisation. Activation maximisation found sparser inputs that drove targets more strongly than such guesses.
- Inspect results with `get_input_activation(opt_in, ...)` and `activations_to_df`.

## Training to data

```python
coin.train_model(model, inputs, targets_df, num_epochs=100, learning_rate=0.01,
                 param_reg_lambda=0.01, train_slopes=True, train_biases=True, train_tau=True)
```
- `inputs`: (batch, n_sensory, num_layers). `targets_df` columns `batch`, `neuron_idx`, optional `layer` (timestep; if absent, fit the mean over time), `value`.
- Fits group-shared slopes, biases, tau, divisive strengths (and pair gains if `slope_dict` has tuple keys). `param_reg_lambda` keeps parameters near initial values. Train/validation split via `train_fraction`; `wandb=True` for logging.
- Use `coin.training_mode(model)` context for manual training loops.

## Caveats to state with model results

The model is a hypothesis generator: point neurons, no neuromodulation, one-timestep synaptic delay for all connections, sign from predicted neurotransmitter. Report the parameters used (slope, tau, threshold, layers, input), and check robustness across a few slope/tau values.
