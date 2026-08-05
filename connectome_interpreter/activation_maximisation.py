from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import contextlib
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from tqdm import tqdm
from scipy import sparse
from scipy.sparse import coo_matrix
import plotly.graph_objects as go
from plotly.colors import qualitative
import plotly.colors

from .compress_paths import result_summary, signed_conn_by_path_length_data

from .utils import (
    adjacency_df_to_el,
    arrayable,
    get_activations,
    to_nparray,
    scipy_sparse_to_pytorch,
)


@dataclass
class TargetActivation:
    """
    Dataclass to handle target activations for activation maximisation.

    The target activations can be specified as a dictionary or a DataFrame.
    The dictionary should have the following structure:

    `{layer: {neuron_index: target_activation_value}}`

    The DataFrame should have the following columns:

    - 'batch': The batch index.
    - 'layer': The layer index.
    - 'neuron': The neuron index.
    - 'value': The target activation value.

    Args:
        targets (Union[Dict[int, Dict[int, float]], pd.DataFrame]): The target
            activations. If a dictionary, all batches will have the same
            target. If a DataFrame, each row represents a target activation for
            a specific batch.
        batch_size (Optional[int], optional): The number of batches. Defaults
            to None.
    """

    targets: Union[Dict[int, Dict[int, float]], pd.DataFrame]
    batch_size: Optional[int] | None = None

    def __post_init__(self):
        if isinstance(self.targets, dict):
            self.batch_size = self.batch_size or 1
            rows = []
            for layer, neurons in self.targets.items():
                for neuron, value in neurons.items():
                    for batch in range(self.batch_size):
                        rows.append(
                            {
                                "batch": batch,
                                "layer": layer,
                                "neuron": neuron,
                                "value": value,
                            }
                        )
            self.targets_df = pd.DataFrame(rows)
        else:
            required_cols = ["layer", "neuron", "value"]
            if not all(col in self.targets.columns for col in required_cols):
                raise ValueError(f"DataFrame must contain columns: {required_cols}")

            if "batch" in self.targets.columns:
                self.batch_size = self.batch_size or (self.targets["batch"].max() + 1)
                self.targets_df = self.targets.copy()
            else:
                self.batch_size = self.batch_size or 1
                self.targets_df = pd.DataFrame(
                    [
                        {**row, "batch": i}
                        for i in range(self.batch_size)
                        for _, row in self.targets.iterrows()
                    ]
                )

    def get_batch_targets(self, batch_idx: int) -> Dict[int, Dict[int, float]]:
        batch_data = self.targets_df[self.targets_df["batch"] == batch_idx]
        result = {}
        for _, row in batch_data.iterrows():
            if row["layer"] not in result:
                result[row["layer"]] = {}
            result[row["layer"]][row["neuron"]] = row["value"]
        return result


class _NetworkBase(nn.Module):
    """Shared infrastructure: __init__, _setup_trainable_parameters,
    set_param_grads, properties, divnorm, forward."""

    def __init__(
        self,
        all_weights: Union[torch.Tensor, sparse.spmatrix],
        sensory_indices: arrayable,
        num_layers: int = 2,
        threshold: float = 0.01,
        tanh_steepness: float = 5,
        idx_to_group: Optional[dict] = None,
        default_bias: float = 0.0,
        bias_dict: Optional[dict] = None,
        bias_transform: str = "abs",
        slope_dict: Optional[dict] = None,
        slope_bounds: Optional[dict] = None,
        slope_update_scales: Optional[dict] = None,
        divisive_normalization: Optional[Dict[str, List[str]]] = None,
        divisive_strength: Union[float, int, dict] = 1,
        activation_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        tau: float = 10,
        tau_dict: Optional[dict] = None,
        tau_log_scale: bool = False,
        sensory_input_mode: str = "add",
        device: Optional[torch.device] = None,
        output_clamp_max: Optional[float] = 1.0,
        output_rectify: bool = True,
        incoming_weight_budget: Optional[float] = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(all_weights, np.ndarray):
            # convert to sp.sparse matrix
            all_weights = coo_matrix(all_weights)

        # Convert all_weights to a sparse tensor if it is a scipy sparse matrix
        if isinstance(all_weights, sparse.spmatrix):
            all_weights = scipy_sparse_to_pytorch(all_weights)
        elif isinstance(all_weights, torch.Tensor):
            # if torch tensor, ensure it is sparse
            if not all_weights.is_sparse:
                all_weights = all_weights.to_sparse()
        else:
            raise TypeError(
                "all_weights must be a scipy sparse matrix, numpy array, or torch tensor."
            )
        all_weights = all_weights.coalesce()

        if bias_transform not in {"abs", "identity"}:
            raise ValueError("bias_transform must be 'abs' or 'identity'.")
        if sensory_input_mode not in {"add", "replace"}:
            raise ValueError("sensory_input_mode must be 'add' or 'replace'.")

        # change the synaptic weight to 0 for the divisive normalization pairs
        if divisive_normalization is not None:
            # idx_to_group must also be provided
            assert (
                idx_to_group is not None
            ), "idx_to_group must be provided for divisive normalization."

            pres_indices = []
            posts_indices = []
            for pre, posts in divisive_normalization.items():
                pre_idx = [idx for idx, group in idx_to_group.items() if group == pre]
                for post in posts:
                    post_idx = [
                        idx for idx, group in idx_to_group.items() if group == post
                    ]
                    # get all combinations of pre and post indices
                    these_pre, these_post = zip(
                        *[(apre, apost) for apre in pre_idx for apost in post_idx]
                    )
                    pres_indices.extend(these_pre)
                    posts_indices.extend(these_post)

            # note that post are in rows in the model
            remove_set = set(zip(posts_indices, pres_indices))

            xs, ys = all_weights._indices().cpu().numpy()

            keep = []
            for x, y in zip(xs, ys):
                if (x, y) not in remove_set:
                    keep.append(True)
                else:
                    keep.append(False)

            keep = np.array(keep)
            self.divnorm_indices = all_weights._indices()[:, ~keep].to(device)
            self.divnorm_weights = all_weights._values()[~keep].to(device)
            # sanity check: all values should be negative
            # which index of divnorm_weights is positive?
            posidx = torch.where(self.divnorm_weights > 0)[0]
            # posidx = np.where(self.divnorm_weights > 0)[0]
            if len(posidx) > 0:
                raise ValueError(
                    "Divisive normalization weights should be negative, "
                    f"but found positive weights at indices: {self.divnorm_indices[:, posidx].T}."
                )

            all_weights = torch.sparse_coo_tensor(
                all_weights._indices()[:, keep],
                all_weights._values()[keep],
                size=all_weights.shape,
            ).coalesce()

        self.all_weights = all_weights.to(device).coalesce()
        self.sensory_indices = torch.tensor(sensory_indices, device=device)
        self.num_layers = num_layers
        self.threshold = threshold
        self.tanh_steepness = tanh_steepness
        self.idx_to_group = idx_to_group
        self.activations = []
        self.custom_activation_function = activation_function
        self.default_bias = default_bias
        self.bias_transform = bias_transform
        self.divisive_normalization = divisive_normalization
        self.tau = tau
        self.tau_dict = tau_dict
        # When True, tau_param stores log(tau) so optimisation steps act
        # multiplicatively on tau; effective_tau exponentiates it back.
        self.tau_log_scale = tau_log_scale
        self.sensory_input_mode = sensory_input_mode
        # Upper clamp applied to non-sensory activations after each layer in
        # MultilayeredNetwork.forward. None disables it (unbounded rates, e.g. a
        # rectified-square model); the default 1.0 preserves the historical
        # [0, 1] output range for all existing callers.
        self.output_clamp_max = output_clamp_max

        # Hard threshold gate (>= threshold passes through, below -> 0) applied to
        # the post-layer state. False lets rates go negative, e.g. when the
        # activation function itself is signed; the default True preserves the
        # historical behaviour for all existing callers.
        #
        # This gates the *post-layer* application only. The built-in
        # activation_function applies the same gate internally as the relu of
        # tanh(relu(slope * x + bias)), and that one is NOT under this flag, so
        # with the built-in activation False still never yields negative rates --
        # it only stops sub-threshold activity being re-zeroed after the tau
        # integration, which is itself a large effect. Negative rates require a
        # custom activation_function (which bypasses the internal gate entirely).
        #
        # Honoured by MultilayeredNetwork only; LinearNetwork applies neither this
        # nor output_clamp_max, so both are silently inert there.
        self.output_rectify = output_rectify

        # slope_dict is dual-format: cell-type keys (node mode, legacy per-post
        # slope applied after the matmul) or (pre, post) tuple keys (pair mode,
        # per-connection gain folded into the weights). Detect which.
        self._slope_dict_is_pairwise = bool(slope_dict) and any(
            isinstance(k, tuple) for k in slope_dict
        )
        if self._slope_dict_is_pairwise and idx_to_group is None:
            raise ValueError(
                "idx_to_group must be provided when slope_dict has (pre, post) keys."
            )

        # Setup trainable parameters if idx_to_group is provided
        if idx_to_group is not None:
            self._setup_trainable_parameters(
                idx_to_group,
                bias_dict,
                slope_dict,
                slope_bounds,
                slope_update_scales,
                divisive_strength,
                tau_dict,
                device,
                incoming_weight_budget,
            )
        else:
            # For backward compatibility - no trainable parameters
            self.slope = None
            self.slope_pairs = ()
            self.slope_edge_indices = None
            self.slope_lower_bound = None
            self.slope_upper_bound = None
            self.slope_update_scale = None
            self.divnorm_slope_edge_indices = None
            self.raw_biases = None
            self.indices = None
            self.divisive_strength = None
            self.tau_param = None
            self.tau_indices = None
            self.incoming_weight_budget = None
            self.frozen_row_l1 = None

    def _setup_trainable_parameters(
        self,
        idx_to_group: dict,
        bias_dict,
        slope_dict,
        slope_bounds,
        slope_update_scales,
        divisive_strength,
        tau_dict,
        device,
        incoming_weight_budget=None,
    ):
        """Setup trainable slope and bias parameters grouped by neuron type.

        ``slope_dict`` is dual-format. With cell-type keys it builds a per-type
        ``self.slope`` applied after the matmul (legacy "node mode"). With
        ``(pre, post)`` tuple keys it builds a per-pair ``self.slope`` folded into
        ``effective_weights`` ("pair mode") via :meth:`_setup_slope_parameters`.
        """
        all_types = sorted(set(idx_to_group.values()))
        num_types = len(all_types)
        type2idx = {t: i for i, t in enumerate(all_types)}

        # Setup slope values (node mode only; pair mode handled below) ----
        if not self._slope_dict_is_pairwise:
            if slope_dict is None:
                slope_init = torch.full(
                    (num_types,), self.tanh_steepness, dtype=torch.float32
                )
            else:  # dict
                slope_init = torch.zeros(num_types, dtype=torch.float32)
                for group_name, slope_val in slope_dict.items():
                    if group_name in type2idx:
                        slope_init[type2idx[group_name]] = slope_val
                # Fill missing values with default
                for i, group_name in enumerate(all_types):
                    if group_name not in slope_dict:
                        slope_init[type2idx[group_name]] = self.tanh_steepness

        # Setup bias values----
        if bias_dict is None:
            bias_init = torch.full((num_types,), self.default_bias, dtype=torch.float32)
        else:  # dict
            bias_init = torch.zeros(num_types, dtype=torch.float32)
            for group_name, bias_val in bias_dict.items():
                if group_name in type2idx:
                    bias_init[type2idx[group_name]] = bias_val
            # Fill missing values with default
            for group_name in all_types:
                if group_name not in bias_dict:
                    bias_init[type2idx[group_name]] = self.default_bias

        # Setup divisive normalization strength----
        if self.divisive_normalization is not None:
            # divisive_init has the same length as the number of pres in divisive_normalization
            # it's not set for all cell types which is the case for slope and bias
            # because so far we believe that this is relatively rare, so the parameter
            # will only be set for the pres in divisive_normalization for their connections with the post
            # So for training, these particular connections are assumed to be only divisive normalising
            # and their divisive_strength is strained.
            pre_groups = sorted(self.divisive_normalization.keys())
            self.pre2dividx = {pre: i for i, pre in enumerate(pre_groups)}

            div_init = torch.tensor(
                [
                    (
                        float(divisive_strength)
                        if isinstance(divisive_strength, (int, float))
                        else float(divisive_strength[pre])
                    )
                    for pre in pre_groups
                ],
                dtype=torch.float32,
            )
            self.divisive_strength = nn.Parameter(
                div_init.to(device), requires_grad=False
            )

            # map each divnorm edge to its pre-group parameter index
            _, pre_idxs = self.divnorm_indices
            self.edge_pre_param_idx = torch.tensor(
                [self.pre2dividx[idx_to_group[int(p)]] for p in pre_idxs],
                dtype=torch.long,
                device=device,
            )
        else:
            self.divisive_strength = None
            self.edge_pre_param_idx = None

        # Setup tau values ----
        if tau_dict is None:
            tau_init = torch.full((num_types,), self.tau, dtype=torch.float32)
        else:
            tau_init = torch.full(
                (num_types,), self.tau, dtype=torch.float32
            )  # default fill
            for group_name, tau_val in tau_dict.items():
                if group_name in type2idx:
                    tau_init[type2idx[group_name]] = tau_val

        if getattr(self, "tau_log_scale", False):
            # store log(tau); effective_tau exponentiates so updates are in log space
            tau_init = torch.log(tau_init.clamp(min=1e-6))

        self.tau_param = nn.Parameter(tau_init.to(device), requires_grad=False)

        # note indices only apply to (node-mode) slope and biases, not divnorm
        self.indices = torch.tensor(
            [type2idx[idx_to_group[i]] for i in range(self.all_weights.shape[0])],
            device=device,
        )

        if self._slope_dict_is_pairwise:
            # pair mode: per-(pre, post) gain folded into the weights
            self._setup_slope_parameters(
                slope_dict,
                slope_bounds,
                slope_update_scales,
                idx_to_group,
                device,
            )
            self._setup_incoming_budget(incoming_weight_budget, device)
        else:
            # node mode (legacy): per-type slope applied after the matmul
            self.slope = nn.Parameter(slope_init.to(device), requires_grad=False)
            self.slope_pairs = ()
            self.slope_edge_indices = None
            self.slope_lower_bound = None
            self.slope_upper_bound = None
            self.slope_update_scale = None
            self.divnorm_slope_edge_indices = None
            if incoming_weight_budget is not None:
                raise ValueError(
                    "incoming_weight_budget requires pair-mode slopes "
                    "(slope_dict with (pre, post) tuple keys)."
                )
            self.incoming_weight_budget = None
            self.frozen_row_l1 = None

        self.raw_biases = nn.Parameter(bias_init.to(device), requires_grad=False)

    def _setup_slope_parameters(
        self,
        slope_dict,
        slope_bounds,
        slope_update_scales,
        idx_to_group,
        device,
    ):
        """Build the per-(pre, post) ``slope`` gain (pair mode).

        ``self.slope`` is a vector over the declared pairs; ``slope_edge_indices``
        maps each edge of ``all_weights`` to its pair (or -1), and
        ``divnorm_slope_edge_indices`` does the same for the divisive-normalization
        edges so the same gain multiplies the weight wherever it appears.
        """
        if not slope_dict:
            self.slope = None
            self.slope_pairs = ()
            self.slope_edge_indices = None
            self.slope_lower_bound = None
            self.slope_upper_bound = None
            self.slope_update_scale = None
            self.divnorm_slope_edge_indices = None
            return

        pairs = []
        init_values = []
        lower_values = []
        upper_values = []
        update_scales = []
        slope_bounds = slope_bounds or {}
        slope_update_scales = slope_update_scales or {}
        for pair, value in slope_dict.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(
                    "Pair-mode slope_dict keys must be (pre_group, post_group) tuples."
                )
            pairs.append((pair[0], pair[1]))
            init_values.append(float(value))
            lower, upper = slope_bounds.get(pair, (0.0, float("inf")))
            if float(lower) > float(upper):
                lower = upper
            lower_values.append(float(lower))
            upper_values.append(float(upper))
            update_scales.append(float(slope_update_scales.get(pair, 1.0)))

        pair_to_idx = {pair: i for i, pair in enumerate(pairs)}

        def _edge_pair_map(indices):
            post_idxs, pre_idxs = indices
            mapping = torch.full(
                (post_idxs.numel(),), -1, dtype=torch.long, device=device
            )
            for edge_i, (post_idx, pre_idx) in enumerate(zip(post_idxs, pre_idxs)):
                pair_idx = pair_to_idx.get(
                    (idx_to_group.get(int(pre_idx)), idx_to_group.get(int(post_idx)))
                )
                if pair_idx is not None:
                    mapping[edge_i] = pair_idx
            return mapping

        self.slope_pairs = tuple(pairs)
        self.slope_edge_indices = _edge_pair_map(self.all_weights._indices())
        # same gain applies to the divnorm edges (pulled out of all_weights)
        self.divnorm_slope_edge_indices = (
            _edge_pair_map(self.divnorm_indices)
            if getattr(self, "divnorm_indices", None) is not None
            else None
        )
        self.slope = nn.Parameter(
            torch.tensor(init_values, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.register_buffer(
            "slope_lower_bound",
            torch.tensor(lower_values, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "slope_upper_bound",
            torch.tensor(upper_values, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "slope_update_scale",
            torch.clamp(
                torch.tensor(update_scales, dtype=torch.float32, device=device),
                min=1e-12,
            ),
        )

    def _setup_incoming_budget(self, incoming_weight_budget, device):
        """Optional per-postsynaptic-neuron L1 budget on incoming ``|weights|``.

        When set (pair mode only), :attr:`effective_weights` reparametrizes each
        row so the sum of absolute incoming weights never exceeds
        ``incoming_weight_budget``, reaching it exactly once the row's gains sum
        to 1. Frozen edges keep their exact connectome magnitude and pre-consume
        the budget; the trainable edges share the remainder. ``None`` disables it
        and reproduces the plain per-edge behaviour (the default).

        Note that this *replaces* ``slope_bounds`` as the capping mechanism rather
        than complementing it: a box on the gain does not translate into a fixed
        box on the effective weight, because the weight also depends on the rest
        of the row. Callers using a budget should pass open bounds.
        """
        if incoming_weight_budget is None or self.slope_edge_indices is None:
            self.incoming_weight_budget = None
            self.frozen_row_l1 = None
            return
        if self.divisive_normalization is not None:
            raise NotImplementedError(
                "incoming_weight_budget is not supported together with "
                "divisive_normalization."
            )
        budget = float(incoming_weight_budget)
        if budget <= 0:
            raise ValueError("incoming_weight_budget must be positive.")

        n = self.all_weights.shape[0]
        post_idx = self.all_weights._indices()[0]
        values = self.all_weights._values()
        frozen_mask = self.slope_edge_indices < 0

        frozen_row_l1 = torch.zeros(n, dtype=values.dtype, device=device)
        if torch.any(frozen_mask):
            frozen_row_l1 = frozen_row_l1.index_add(
                0, post_idx[frozen_mask], values[frozen_mask].abs()
            )
        self.register_buffer("frozen_row_l1", frozen_row_l1)
        self.incoming_weight_budget = budget

        # Warn about post-neurons whose frozen incoming |weights| already meet the
        # budget: their trainable edges get squeezed to ~0 and the row can still
        # exceed the budget (frozen edges are never rescaled).
        has_trainable = torch.zeros(n, dtype=torch.bool, device=device)
        if torch.any(~frozen_mask):
            has_trainable[post_idx[~frozen_mask]] = True
        bad = torch.where((frozen_row_l1 >= budget) & has_trainable)[0]
        if bad.numel() > 0:
            warnings.warn(
                f"incoming_weight_budget={budget}: frozen incoming |weights| "
                f"already >= budget for post-neuron indices {bad.tolist()}; their "
                "trainable edges will be forced toward 0 and those rows may still "
                "exceed the budget."
            )

    def set_param_grads(
        self,
        slopes=False,
        raw_biases=False,
        divisive_strength=False,
        tau=False,
    ):
        """Set requires_grad for specific parameters."""
        if self.slope is not None:
            self.slope.requires_grad_(slopes)
        if self.raw_biases is not None:
            self.raw_biases.requires_grad_(raw_biases)
        if self.divisive_strength is not None:
            self.divisive_strength.requires_grad_(divisive_strength)
        if self.tau_param is not None:
            self.tau_param.requires_grad_(tau)

    @property
    def biases(self):
        """
        Get the biases of the neurons, applying absolute value if raw_biases
        are provided. If raw_biases is None, returns None.

        Returns:
            torch.Tensor or None: The biases of the neurons, or None if raw_biases
            is not set.
        """
        if self.raw_biases is None:
            return None
        if self.bias_transform == "identity":
            return self.raw_biases
        return torch.abs(self.raw_biases)

    @property
    def slope_is_pairwise(self):
        """True when ``slope`` is a per-(pre, post) gain folded into the weights."""
        return getattr(self, "slope_edge_indices", None) is not None

    @property
    def effective_slope(self):
        """Slope clamped to its bounds. Pair mode clamps to the per-pair
        [lower, upper]; node mode (no explicit bounds) applies a 1e-6 floor so a
        trained slope stays strictly positive. None if no slope."""
        if self.slope is None:
            return None
        lower = self.slope_lower_bound
        upper = self.slope_upper_bound
        if lower is None or upper is None:
            return torch.clamp(self.slope, min=1e-6)
        return torch.minimum(torch.maximum(self.slope, lower), upper)

    def rescale_slope_update_(self, before_step: torch.Tensor) -> None:
        # only meaningful in pair mode (node mode has no update scale)
        if self.slope is None or self.slope_update_scale is None:
            return
        with torch.no_grad():
            step = self.slope - before_step
            self.slope.copy_(before_step + step / self.slope_update_scale)

    def project_parameter_bounds_(self) -> None:
        # only pair-mode slope has bounds to project onto
        if not self.slope_is_pairwise:
            return
        with torch.no_grad():
            self.slope.copy_(self.effective_slope)

    @property
    def effective_weights(self):
        # Only pair-mode slope folds into the weights. Node-mode slope (None
        # slope_edge_indices) is applied after the matmul in activation_function.
        edge_param_idx = getattr(self, "slope_edge_indices", None)
        if self.slope is None or edge_param_idx is None:
            return self.all_weights

        indices = self.all_weights._indices()
        values = self.all_weights._values()
        budget = getattr(self, "incoming_weight_budget", None)

        if budget is None:
            # Default: per-edge multiplier (trainable edges scaled by their slope).
            multipliers = torch.ones_like(values)
            trainable_edge_mask = edge_param_idx >= 0
            if torch.any(trainable_edge_mask):
                multipliers = multipliers.clone()
                multipliers[trainable_edge_mask] = self.effective_slope[
                    edge_param_idx[trainable_edge_mask]
                ].to(values.dtype)
            new_values = values * multipliers
        else:
            # Per-postsynaptic-neuron L1 budget. For each post neuron j the
            # trainable incoming edge i -> j gets magnitude
            #     R_j * |m_ij| / max(1, T_j),
            # where T_j = sum_k |m_kj| over j's trainable incoming edges and
            # R_j = max(budget - frozen_row_l1_j, 0). Frozen edges keep their
            # exact magnitude, so the row |w| sum comes to
            # frozen_j + R_j * min(T_j, 1) <= budget: a hard cap the row reaches
            # exactly once T_j >= 1 rather than approaching it asymptotically.
            # Below T_j = 1 the row spends less than its budget, so the total
            # incoming drive stays learnable.
            #
            # The denominator aggregates |m|, matching the numerator, so the cap
            # holds for any gain values including negative ones -- a signed sum
            # would let opposite-sign gains cancel in the denominator while both
            # still contribute their magnitude to the row L1.
            n = self.all_weights.shape[0]
            post = indices[0]
            trainable_mask = edge_param_idx >= 0
            m_pair = self.effective_slope.to(values.dtype)
            edge_m = torch.where(
                trainable_mask,
                m_pair[edge_param_idx.clamp(min=0)],
                torch.zeros_like(values),
            )
            row_T = torch.zeros(n, dtype=values.dtype, device=values.device).index_add(
                0, post, edge_m.abs()
            )
            remaining = torch.clamp(
                self.incoming_weight_budget - self.frozen_row_l1.to(values.dtype),
                min=0.0,
            )
            row_scale = remaining / torch.clamp(row_T, min=1.0)
            trainable_new = torch.sign(values) * edge_m * row_scale[post]
            new_values = torch.where(trainable_mask, trainable_new, values)

        return torch.sparse_coo_tensor(
            indices,
            new_values,
            size=self.all_weights.shape,
            device=values.device,
        ).coalesce()

    @property
    def effective_tau(self):
        if self.tau_param is None:
            return self.tau
        if getattr(self, "tau_log_scale", False):
            # tau_param holds log(tau); exponentiate back to ms
            return torch.clamp(torch.exp(self.tau_param), min=1.0)
        return torch.clamp(self.tau_param, min=1.0)  # tau < 1 is physically meaningless

    def _apply_sensory_input(self, state: torch.Tensor, input_at_layer: torch.Tensor):
        state = state.clone()
        if self.sensory_input_mode == "add":
            state[self.sensory_indices, :] = (
                state[self.sensory_indices, :] + input_at_layer
            )
        elif self.sensory_input_mode == "replace":
            state[self.sensory_indices, :] = input_at_layer
        else:
            raise ValueError(f"Unknown sensory_input_mode: {self.sensory_input_mode}")
        return state

    def _should_apply_sensory_after_layer(self, layer: int) -> bool:
        return self.sensory_input_mode == "replace" or layer != self.num_layers - 1

    def _sensory_input_after_layer(
        self, inputs: torch.Tensor, layer: int
    ) -> torch.Tensor:
        input_layer = layer if self.sensory_input_mode == "replace" else layer + 1
        return inputs[:, :, input_layer].t()

    def _initial_full_input(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        requires_grad: bool,
    ) -> torch.Tensor:
        n_nodes = self.all_weights.size(1)
        batch_size = inputs.size(0)
        if initial_state is None:
            full_input = torch.zeros(
                n_nodes,
                batch_size,
                device=inputs.device,
                dtype=inputs.dtype,
                requires_grad=requires_grad,
            )
        else:
            full_input = torch.as_tensor(
                initial_state,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            if full_input.dim() == 1:
                if full_input.numel() != n_nodes:
                    raise ValueError("initial_state length must match the node count.")
                full_input = full_input.view(-1, 1).expand(-1, batch_size)
            elif full_input.dim() == 2:
                if full_input.shape == (batch_size, n_nodes):
                    full_input = full_input.t()
                elif full_input.shape != (n_nodes, batch_size):
                    raise ValueError(
                        "initial_state must have shape (nodes,), (batch, nodes), "
                        "or (nodes, batch)."
                    )
            else:
                raise ValueError(
                    "initial_state must have shape (nodes,), (batch, nodes), "
                    "or (nodes, batch)."
                )
            full_input = full_input.clone()

        return full_input.scatter(
            0,
            self.sensory_indices.view(-1, 1).expand(-1, batch_size),
            inputs[:, :, 0].t(),
        )

    def divnorm(
        self, slopes: Union[float, torch.Tensor], x_previous: torch.Tensor
    ) -> torch.Tensor:
        """
        Change the slopes based on divisive normalization. The new slope bewteen pre and
        post is: original_slope * (1 - pre_post_weight * pre_activation *
        divisive_strength).

        Args:
            slopes (Union[float, torch.Tensor]): The slopes to be normalized.
                If a float, it is assumed to be the same for all neurons.
            x_previous (torch.Tensor): The previous activations of the neurons.

        Returns:
            torch.Tensor: The output tensor after applying divisive normalization.
        """
        if self.divisive_normalization is None:
            return slopes
        if isinstance(slopes, float):
            slopes = torch.full(x_previous.shape, slopes, device=x_previous.device)

        # The divnorm buffers are plain tensors, so coerce them to the device the
        # model is actually running on (handles a model moved via .to() after init).
        device = x_previous.device
        post_idxs, pre_idxs = self.divnorm_indices.to(device)  # (K,), (K,)
        strengths = self.divisive_strength[self.edge_pre_param_idx.to(device)]  # (K,)

        # the pair slope multiplies the weight wherever it appears, incl. divnorm
        divnorm_weights = self.divnorm_weights.to(device)
        edge_idx = getattr(self, "divnorm_slope_edge_indices", None)
        if self.slope is not None and edge_idx is not None:
            edge_idx = edge_idx.to(device)
            mult = torch.ones_like(divnorm_weights)
            mask = edge_idx >= 0
            if torch.any(mask):
                mult = mult.clone()
                mult[mask] = self.effective_slope[edge_idx[mask]].to(
                    divnorm_weights.dtype
                )
            divnorm_weights = divnorm_weights * mult

        per_edge = (
            divnorm_weights.unsqueeze(1)  # (K, 1)
            * x_previous[pre_idxs]  # (K, batch)
            * strengths.unsqueeze(1)  # (K, 1)
        )
        per_post = torch.zeros_like(slopes).index_add_(0, post_idxs, per_edge)
        return torch.clamp(slopes * (1 + per_post), min=0.0)


class LinearNetwork(_NetworkBase):
    """
    A PyTorch module representing a multilayered neural network model with trainable
    parameters and flexible activation functions.

    This network architecture is designed to process temporal sequences of input data
    through multiple synaptic hops, with the initial layer handling only external inputs
    and subsequent layers processing external+internal input.

    Excitabililty is implemented as the slope of the tanh activation function. Users can
    specify default excitability (tanh_steepness) for all neurons, as well as specific
    excitabilities for neuron groups. In training, the excitabilities are shared per
    group specified by `idx_to_group`.

    Baseline activity is taken into account through biases. Users can specify default
    biases for all neurons, as well as specific biases for neuron groups. In training,
    the biases are shared per group specified by `idx_to_group`.

    In divisive normalisation, the synaptic connections specified in
    `divisive_normalization` no longer participate in the normal subtractive inhibition.
    Rather, they are used to change the slope of the post-synaptic neuron: the new slope
    is: max(0, original_slope * (1 + pre_post_weight * pre_activation *
    divisive_strength)). `divisive_strength` is a parameter the user can set, to specity
    how strong the divisive normalization is. `max(0, ...)` is to ensure that the slope
    does not become negative.

    Attributes:
        all_weights (torch.nn.Parameter): The connectome. Input neurons are in
            the columns.
        sensory_indices (list[int]): Indices indicating which rows/columns in
            the all_weights matrix correspond to sensory neurons.
        num_layers (int): The number of layers in the network.
        threshold (float): The activation threshold for neurons in the network.
        activations (numpy.ndarray): An array storing the activations of all (batches of)
            neurons (rows) across time steps (columns).
        custom_activation_function (Callable): A custom activation function to use
            instead of the default. The function should take the model, the input tensor
            x, and the previous activations x_previous as arguments, and return the
            output tensor after applying the activation function.
        default_bias (float): Default bias value for all neurons.
        slope (torch.nn.Parameter): Trainable parameter for the steepness of the tanh
            activation function, grouped by neuron type.
        raw_biases (torch.nn.Parameter): Trainable parameter for the biases of neurons,
            grouped by neuron type.
        indices (torch.Tensor): Indices mapping each neuron to its group for parameter
            sharing.
        divisive_strength (torch.nn.Parameter): Trainable parameter for the strength of
            divisive normalization, if applicable. In training, the parameter is shared
            per group specified by idx_to_group. This parameter only exists for the pres
            in divisive_normalization.
        divisive_normalization (Dict[str, List[str]]): A dictionary where keys are
            pre-synaptic neuron groups and values are lists of post-synaptic neuron
            groups. These *inhibitory* connections are implemented divisively instead of
            subtractively.


    Args:
        all_weights (Union[torch.Tensor, scipy.sparse.spmatrix]): The connectome. Input
            neurons are in the columns.
        sensory_indices (list[int]): A list indicating the indices of sensory neurons
            within the network.
        num_layers (int, optional): The number of temporal layers to unroll the network
            through.  Defaults to 2.
        threshold (float, optional): The threshold for activation of neurons. Defaults
            to 0.01.
        tanh_steepness (float, optional): Default steepness for tanh activation.
            Defaults to 5.
        idx_to_group (dict, optional): Mapping from neuron indices to group names for
            parameter sharing (all neurons in the same group share the same parameter).
            Defaults to None (no parameter sharing).
        default_bias (float, optional): Default bias value for all groups. Defaults to
            0.0.
        bias_dict (dict, optional): Dict mapping group names to bias values. Defaults to
            None (uses default_bias).
        slope_dict (dict, optional): Dict mapping group names to slope values. Uses
            tanh_steepness if None.
        divisive_normalization (Dict[str, List[str]], optional): A dictionary where keys
            are pre-synaptic neuron groups and values are lists of post-synaptic neuron
            groups. These *inhibitory* connections are implemented divisively instead of
            subtractively. Defaults to None.
        divisive_strength (Union[float, int, dict], optional): The strength of the
            divisive normalization. If a float or int, it applies to all connections.
            If a dict, it maps pre-synaptic neuron groups to their specific divisive
            strengths. Defaults to 1.
        activation_function (Callable, optional): Custom activation function. If None,
            uses default implementation.
        tau (float, optional): Time constant. Higher tau results in slower changes.
            Minimum 1, where the activation at the current time step is solely
            determined by the current input. Defaults to 10.
        device (torch.device, optional): Device for computation.
    """

    def activation_function(
        self,
        x: torch.Tensor,
        x_previous: Optional[torch.Tensor] = None,
        slopes_full: Optional[torch.Tensor] = None,
        biases_full: Optional[torch.Tensor] = None,
        taus_full=None,
    ) -> torch.Tensor:
        """
        Apply the activation function to the input tensor. Currently it is
        tanh(relu(slope * x + bias)). x = weights @ inputs (done outside the activation
        function).

        Args:
            x (torch.Tensor): The input tensor to the activation function.
            x_previous (Optional[torch.Tensor]): The previous activations of the neurons.
            slopes_full (Optional[torch.Tensor]): Pre-expanded slopes, shape
                (num_neurons, 1). If None, computed from self.effective_slope[self.indices].
            biases_full (Optional[torch.Tensor]): Pre-expanded biases, shape
                (num_neurons, 1). If None, computed from self.biases[self.indices].
            taus_full: Pre-expanded taus, shape (num_neurons, 1), or scalar.
                If None, computed from self.effective_tau[self.indices].

        Returns:
            torch.Tensor: The output tensor after applying the activation function.
        """
        if self.custom_activation_function is not None:
            return self.custom_activation_function(self, x, x_previous)

        # --- slopes ---
        if self.slope_is_pairwise:
            # pair-mode gain is folded into effective_weights; no post-matmul slope
            slopes = 1.0
        elif slopes_full is not None:
            slopes = slopes_full.expand(-1, x.shape[1])
        elif self.slope is None:
            slopes = self.tanh_steepness
        else:
            slopes = (
                self.effective_slope[self.indices]
                .view(-1, 1)
                .expand(-1, x.shape[1])
                .to(x.device)
            )

        slopes = self.divnorm(slopes, x_previous)

        # --- biases ---
        if biases_full is not None:
            biases = biases_full.expand(-1, x.shape[1])
        elif self.biases is None:
            biases = self.default_bias
        else:
            biases = (
                self.biases[self.indices]
                .view(-1, 1)
                .expand(-1, x.shape[1])
                .to(x.device)
            )

        # x = self.tau * x_previous + slopes * x + biases
        x = slopes * x + biases

        # # Thresholded relu
        # x = torch.where(x >= self.threshold, x, torch.zeros_like(x))

        # --- taus ---
        if taus_full is not None:
            taus = (
                taus_full.expand(-1, x.shape[1])
                if isinstance(taus_full, torch.Tensor)
                else taus_full
            )
        elif self.tau_param is not None:
            taus = (
                self.effective_tau[self.indices]
                .view(-1, 1)
                .expand(-1, x.shape[1])
                .to(x.device)
            )
        else:
            taus = self.tau

        x = 1 / taus * x + (taus - 1) / taus * x_previous

        return x

    def _forward_chunk(
        self,
        x,
        start_layer,
        end_layer,
        inputs,
        slopes_full,
        biases_full,
        taus_full,
        weights,
    ):
        """Run a chunk of timesteps. For use with gradient checkpointing.

        Returns:
            Flat tuple (x_final, *per_layer_xs). Each per_layer_x is the raw
            timestep activation with shape (neurons, batch) — NOT transposed.
            Transposing (or stacking) here would make the per-layer tensors
            siblings of a shared parent, breaking the parent→child edge that
            autograd.grad needs to walk layer-wise.
        """
        chunk_acts = []
        for alayer in range(start_layer, end_layer):
            x_previous = x
            x = torch.sparse.mm(weights, x)
            x = self.activation_function(
                x,
                x_previous=x_previous,
                slopes_full=slopes_full,
                biases_full=biases_full,
                taus_full=taus_full,
            )
            if self._should_apply_sensory_after_layer(alayer):
                x = self._apply_sensory_input(
                    x,
                    self._sensory_input_after_layer(inputs, alayer),
                )
            chunk_acts.append(x)  # <-- no .t()
        return (x, *chunk_acts)

    def forward(
        self,
        inputs: torch.Tensor,
        manipulate: Optional[
            Union[
                Dict[int, Dict[Union[int, str], float]],
                Dict[int, Dict[int, Dict[Union[int, str], float]]],
            ]
        ] = None,
        checkpoint_steps: int = 0,
        return_layer_list: bool = False,
        initial_state: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            return_layer_list (bool): If True, return List[Tensor] of length
                num_layers, each shape (neurons, batch), on GPU with grad_fn
                preserved. Required for layer-wise gradient attribution via
                torch.autograd.grad — stacking/transposing makes per-layer
                tensors siblings of a shared parent, which breaks the graph
                path autograd.grad needs. When True, self.activations is not
                populated.
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, device=self.all_weights.device)
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.all_weights.device)

        # Handle 2D inputs by expanding to 3D
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)  # Add batch dimension
            single_input = True
        else:
            single_input = False

        if manipulate is not None:
            # check if manipulate is per-batch
            if not all(
                isinstance(act_dict, dict)
                for _, this_batch in manipulate.items()
                for _, act_dict in this_batch.items()
            ):
                batch_manipulate = manipulate.copy()
                # add batch dimension
                manipulate = dict.fromkeys(range(inputs.shape[0]), batch_manipulate)

            if self.idx_to_group is None:
                manipulate_idx = {
                    b: {
                        layer: {int(c): a for c, a in act_dict.items()}
                        for layer, act_dict in this_batch.items()
                    }
                    for b, this_batch in manipulate.items()
                }
            else:
                # convert group names to indices
                manipulate_idx = {}
                for b, this_batch in manipulate.items():
                    manipulate_idx[b] = {}
                    for layer, act_dict in this_batch.items():
                        manipulate_idx[b][layer] = {
                            idx: act_dict[grp]
                            for idx, grp in self.idx_to_group.items()
                            if grp in act_dict
                        }

        if self.slope_is_pairwise:
            # pair-mode gain is in effective_weights; activation uses unit slope
            slopes_full = None
            biases_full = (
                self.biases[self.indices].view(-1, 1)
                if self.biases is not None
                else None
            )
        elif self.slope is not None:
            slopes_full = self.effective_slope[self.indices].view(-1, 1)
            biases_full = self.biases[self.indices].view(-1, 1)
        else:
            slopes_full = None
            biases_full = None

        if self.tau_param is not None:
            taus_full = self.effective_tau[self.indices].view(-1, 1)
        else:
            taus_full = None
        weights = self.effective_weights

        req_grad = inputs.requires_grad
        _needs_grad = req_grad or any(
            p.requires_grad for p in self.parameters() if p is not None
        )

        use_ckpt = checkpoint_steps > 0 and _needs_grad and manipulate is None

        full_input = self._initial_full_input(inputs, initial_state, req_grad)

        # ---- Layer 0 ----
        x = torch.sparse.mm(weights, full_input)
        x = self.activation_function(
            x,
            x_previous=full_input,
            slopes_full=slopes_full,
            biases_full=biases_full,
            taus_full=taus_full,
        )

        if self._should_apply_sensory_after_layer(0):
            x = self._apply_sensory_input(
                x,
                self._sensory_input_after_layer(inputs, 0),
            )

        if manipulate is not None:
            x = x.clone()
            for b, _ in manipulate.items():
                if 0 in manipulate_idx[b]:
                    for neuron_idx, target_act in manipulate_idx[b][0].items():
                        x[neuron_idx, b] = target_act

        # ---- Remaining layers ----
        if return_layer_list:
            per_layer_acts = [x]  # <-- no .t()

            if use_ckpt:
                for chunk_start in range(1, self.num_layers, checkpoint_steps):
                    chunk_end = min(chunk_start + checkpoint_steps, self.num_layers)
                    result = torch_checkpoint(
                        self._forward_chunk,
                        x,
                        chunk_start,
                        chunk_end,
                        inputs,
                        slopes_full,
                        biases_full,
                        taus_full,
                        weights,
                        use_reentrant=False,
                    )
                    x = result[0]
                    per_layer_acts.extend(result[1:])
            else:
                for alayer in range(1, self.num_layers):
                    x_previous = x.clone()
                    x = torch.sparse.mm(weights, x)
                    x = self.activation_function(
                        x,
                        x_previous=x_previous,
                        slopes_full=slopes_full,
                        biases_full=biases_full,
                        taus_full=taus_full,
                    )
                    if self._should_apply_sensory_after_layer(alayer):
                        x = self._apply_sensory_input(
                            x,
                            self._sensory_input_after_layer(inputs, alayer),
                        )
                    if manipulate is not None:
                        x = x.clone()
                        for b, _ in manipulate.items():
                            if alayer in manipulate_idx[b]:
                                for neuron_idx, target_act in manipulate_idx[b][
                                    alayer
                                ].items():
                                    x[neuron_idx, b] = target_act
                    per_layer_acts.append(x)  # <-- no .t()

            del inputs, x
            torch.cuda.empty_cache()
            return per_layer_acts

        # ---- default (stacked) path ----
        if use_ckpt:
            act_chunks = [x.t().unsqueeze(-1)]
            for chunk_start in range(1, self.num_layers, checkpoint_steps):
                chunk_end = min(chunk_start + checkpoint_steps, self.num_layers)
                result = torch_checkpoint(
                    self._forward_chunk,
                    x,
                    chunk_start,
                    chunk_end,
                    inputs,
                    slopes_full,
                    biases_full,
                    taus_full,
                    weights,
                    use_reentrant=False,
                )
                x = result[0]
                # result[1:] are (neurons, batch) — transpose each and stack along time
                act_chunks.append(torch.stack([cx.t() for cx in result[1:]], dim=-1))
            self.activations = torch.cat(act_chunks, dim=-1).cpu()
        else:
            acts = [x.t().cpu()]
            for alayer in range(1, self.num_layers):
                x_previous = x.clone()
                x = torch.sparse.mm(weights, x)
                x = self.activation_function(
                    x,
                    x_previous=x_previous,
                    slopes_full=slopes_full,
                    biases_full=biases_full,
                    taus_full=taus_full,
                )
                if self._should_apply_sensory_after_layer(alayer):
                    x = self._apply_sensory_input(
                        x,
                        self._sensory_input_after_layer(inputs, alayer),
                    )
                if manipulate is not None:
                    x = x.clone()
                    for b, _ in manipulate.items():
                        if alayer in manipulate_idx[b]:
                            for neuron_idx, target_act in manipulate_idx[b][
                                alayer
                            ].items():
                                x[neuron_idx, b] = target_act
                acts.append(x.t().cpu())
            self.activations = torch.stack(acts, dim=-1)

        del inputs, x
        torch.cuda.empty_cache()

        if single_input:
            self.activations = self.activations.squeeze(0)
        return self.activations


class MultilayeredNetwork(_NetworkBase):
    """
    A PyTorch module representing a multilayered neural network model with trainable
    parameters and flexible activation functions.

    This network architecture is designed to process temporal sequences of input data
    through multiple synaptic hops, with the initial layer handling only external inputs
    and subsequent layers processing external+internal input.

    Excitabililty is implemented as the slope of the tanh activation function. Users can
    specify default excitability (tanh_steepness) for all neurons, as well as specific
    excitabilities for neuron groups. In training, the excitabilities are shared per
    group specified by `idx_to_group`.

    Baseline activity is taken into account through biases. Users can specify default
    biases for all neurons, as well as specific biases for neuron groups. In training,
    the biases are shared per group specified by `idx_to_group`.

    In divisive normalisation, the synaptic connections specified in
    `divisive_normalization` no longer participate in the normal subtractive inhibition.
    Rather, they are used to change the slope of the post-synaptic neuron: the new slope
    is: max(0, original_slope * (1 + pre_post_weight * pre_activation *
    divisive_strength)). `divisive_strength` is a parameter the user can set, to specity
    how strong the divisive normalization is. `max(0, ...)` is to ensure that the slope
    does not become negative.

    Attributes:
        all_weights (torch.nn.Parameter): The connectome. Input neurons are in
            the columns.
        sensory_indices (list[int]): Indices indicating which rows/columns in
            the all_weights matrix correspond to sensory neurons.
        num_layers (int): The number of layers in the network.
        threshold (float): The activation threshold for neurons in the network.
        activations (numpy.ndarray): An array storing the activations of all (batches of)
            neurons (rows) across time steps (columns).
        custom_activation_function (Callable): A custom activation function to use
            instead of the default. The function should take the model, the input tensor
            x, and the previous activations x_previous as arguments, and return the
            output tensor after applying the activation function.
        default_bias (float): Default bias value for all neurons.
        slope (torch.nn.Parameter): Trainable parameter for the steepness of the tanh
            activation function, grouped by neuron type.
        raw_biases (torch.nn.Parameter): Trainable parameter for the biases of neurons,
            grouped by neuron type.
        indices (torch.Tensor): Indices mapping each neuron to its group for parameter
            sharing.
        divisive_strength (torch.nn.Parameter): Trainable parameter for the strength of
            divisive normalization, if applicable. In training, the parameter is shared
            per group specified by idx_to_group. This parameter only exists for the pres
            in divisive_normalization.
        divisive_normalization (Dict[str, List[str]]): A dictionary where keys are
            pre-synaptic neuron groups and values are lists of post-synaptic neuron
            groups. These *inhibitory* connections are implemented divisively instead of
            subtractively.


    Args:
        all_weights (Union[torch.Tensor, scipy.sparse.spmatrix]): The connectome. Input
            neurons are in the columns.
        sensory_indices (list[int]): A list indicating the indices of sensory neurons
            within the network.
        num_layers (int, optional): The number of temporal layers to unroll the network
            through.  Defaults to 2.
        threshold (float, optional): The threshold for activation of neurons. Defaults
            to 0.01.
        tanh_steepness (float, optional): Default steepness for tanh activation.
            Defaults to 5.
        idx_to_group (dict, optional): Mapping from neuron indices to group names for
            parameter sharing (all neurons in the same group share the same parameter).
            Defaults to None (no parameter sharing).
        default_bias (float, optional): Default bias value for all groups. Defaults to
            0.0.
        bias_dict (dict, optional): Dict mapping group names to bias values. Defaults to
            None (uses default_bias).
        slope_dict (dict, optional): Dict mapping group names to slope values. Uses
            tanh_steepness if None.
        divisive_normalization (Dict[str, List[str]], optional): A dictionary where keys
            are pre-synaptic neuron groups and values are lists of post-synaptic neuron
            groups. These *inhibitory* connections are implemented divisively instead of
            subtractively. Defaults to None.
        divisive_strength (Union[float, int, dict], optional): The strength of the
            divisive normalization. If a float or int, it applies to all connections.
            If a dict, it maps pre-synaptic neuron groups to their specific divisive
            strengths. Defaults to 1.
        activation_function (Callable, optional): Custom activation function. If None,
            uses default implementation.
        tau (float, optional): Time constant. Higher tau results in slower changes.
            Minimum 1, where the activation at the current time step is solely
            determined by the current input. Defaults to 10.
        device (torch.device, optional): Device for computation.
    """

    def activation_function(
        self,
        x: torch.Tensor,
        x_previous: Optional[torch.Tensor] = None,
        slopes_full: Optional[torch.Tensor] = None,
        biases_full: Optional[torch.Tensor] = None,
        taus_full=None,
    ) -> torch.Tensor:
        """
        Apply the activation function to the input tensor. Currently it is
        tanh(relu(slope * x + bias)). x = weights @ inputs (done outside the activation
        function).

        Args:
            x (torch.Tensor): The input tensor to the activation function.
            x_previous (Optional[torch.Tensor]): The previous activations of the neurons.
            slopes_full (Optional[torch.Tensor]): Pre-expanded slopes, shape
                (num_neurons, 1). If None, computed from self.effective_slope[self.indices].
            biases_full (Optional[torch.Tensor]): Pre-expanded biases, shape
                (num_neurons, 1). If None, computed from self.biases[self.indices].
            taus_full: Pre-expanded taus, shape (num_neurons, 1), or scalar.
                If None, computed from self.effective_tau[self.indices].

        Returns:
            torch.Tensor: The output tensor after applying the activation function.
        """
        if self.custom_activation_function is not None:
            return self.custom_activation_function(self, x, x_previous)

        # --- slopes ---
        if self.slope_is_pairwise:
            # pair-mode gain is folded into effective_weights; no post-matmul slope
            slopes = 1.0
        elif slopes_full is not None:
            slopes = slopes_full.expand(-1, x.shape[1])
        elif self.slope is None:
            slopes = self.tanh_steepness
        else:
            slopes = (
                self.effective_slope[self.indices]
                .view(-1, 1)
                .expand(-1, x.shape[1])
                .to(x.device)
            )

        slopes = self.divnorm(slopes, x_previous)

        # --- biases ---
        if biases_full is not None:
            biases = biases_full.expand(-1, x.shape[1])
        elif self.biases is None:
            biases = self.default_bias
        else:
            biases = (
                self.biases[self.indices]
                .view(-1, 1)
                .expand(-1, x.shape[1])
                .to(x.device)
            )

        x = slopes * x + biases

        # Thresholded relu
        # x = torch.where(x >= self.threshold, x, torch.zeros_like(x))
        # this propagates NaN honesty
        x = torch.relu(x - self.threshold) + self.threshold * (x >= self.threshold).to(
            x.dtype
        )

        # --- taus ---
        if taus_full is not None:
            taus = (
                taus_full.expand(-1, x.shape[1])
                if isinstance(taus_full, torch.Tensor)
                else taus_full
            )
        elif self.tau_param is not None:
            taus = (
                self.effective_tau[self.indices]
                .view(-1, 1)
                .expand(-1, x.shape[1])
                .to(x.device)
            )
        else:
            taus = self.tau

        x = 1 / taus * torch.tanh(x) + (taus - 1) / taus * x_previous

        return x

    def _forward_chunk(
        self,
        x,
        start_layer,
        end_layer,
        inputs,
        slopes_full,
        biases_full,
        taus_full,
        weights,
    ):
        """Returns flat tuple (x_final, *per_layer_xs), each (neurons, batch)."""
        chunk_acts = []
        for alayer in range(start_layer, end_layer):
            x_previous = x
            x = torch.sparse.mm(weights, x)
            x = self.activation_function(
                x,
                x_previous=x_previous,
                slopes_full=slopes_full,
                biases_full=biases_full,
                taus_full=taus_full,
            )
            if self._should_apply_sensory_after_layer(alayer):
                x = self._apply_sensory_input(
                    x,
                    self._sensory_input_after_layer(inputs, alayer),
                )
                # x = torch.where(x >= self.threshold, x, torch.zeros_like(x))
                # this propagates NaN honesty
                if self.output_rectify:
                    x = torch.relu(x - self.threshold) + self.threshold * (
                        x >= self.threshold
                    ).to(x.dtype)
                x = (
                    x
                    if self.output_clamp_max is None
                    else torch.clamp(x, max=self.output_clamp_max)
                )
                if self.sensory_input_mode == "replace":
                    x = self._apply_sensory_input(
                        x,
                        self._sensory_input_after_layer(inputs, alayer),
                    )
            chunk_acts.append(x)
        return (x, *chunk_acts)

    def forward(
        self,
        inputs: torch.Tensor,
        manipulate: Optional[
            Union[
                Dict[int, Dict[Union[int, str], float]],
                Dict[int, Dict[int, Dict[Union[int, str], float]]],
            ]
        ] = None,
        checkpoint_steps: int = 0,
        return_layer_list: bool = False,
        initial_state: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            return_layer_list (bool): If True, return List[Tensor] of length
                num_layers, each shape (neurons, batch), on GPU with grad_fn
                preserved. Required for layer-wise gradient attribution via
                torch.autograd.grad — stacking/transposing makes per-layer
                tensors siblings of a shared parent, which breaks the graph
                path autograd.grad needs. When True, self.activations is not
                populated.
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, device=self.all_weights.device)
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.all_weights.device)

        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
            single_input = True
        else:
            single_input = False

        if manipulate is not None:
            if not all(
                isinstance(act_dict, dict)
                for _, this_batch in manipulate.items()
                for _, act_dict in this_batch.items()
            ):
                batch_manipulate = manipulate.copy()
                manipulate = dict.fromkeys(range(inputs.shape[0]), batch_manipulate)

            if self.idx_to_group is None:
                manipulate_idx = {
                    b: {
                        layer: {int(c): a for c, a in act_dict.items()}
                        for layer, act_dict in this_batch.items()
                    }
                    for b, this_batch in manipulate.items()
                }
            else:
                manipulate_idx = {}
                for b, this_batch in manipulate.items():
                    manipulate_idx[b] = {}
                    for layer, act_dict in this_batch.items():
                        manipulate_idx[b][layer] = {
                            idx: act_dict[grp]
                            for idx, grp in self.idx_to_group.items()
                            if grp in act_dict
                        }

        if self.slope_is_pairwise:
            # pair-mode gain is in effective_weights; activation uses unit slope
            slopes_full = None
            biases_full = (
                self.biases[self.indices].view(-1, 1)
                if self.biases is not None
                else None
            )
        elif self.slope is not None:
            slopes_full = self.effective_slope[self.indices].view(-1, 1)
            biases_full = self.biases[self.indices].view(-1, 1)
        else:
            slopes_full = None
            biases_full = None

        if self.tau_param is not None:
            taus_full = self.effective_tau[self.indices].view(-1, 1)
        else:
            taus_full = None
        weights = self.effective_weights

        req_grad = inputs.requires_grad
        _needs_grad = req_grad or any(
            p.requires_grad for p in self.parameters() if p is not None
        )

        use_ckpt = checkpoint_steps > 0 and _needs_grad and manipulate is None

        full_input = self._initial_full_input(inputs, initial_state, req_grad)

        # ---- Layer 0 ----
        x = torch.sparse.mm(weights, full_input)
        x = self.activation_function(
            x,
            x_previous=full_input,
            slopes_full=slopes_full,
            biases_full=biases_full,
            taus_full=taus_full,
        )

        if self._should_apply_sensory_after_layer(0):
            x = self._apply_sensory_input(
                x,
                self._sensory_input_after_layer(inputs, 0),
            )
            # x = torch.where(x >= self.threshold, x, torch.zeros_like(x))
            # this propagates NaN honesty
            if self.output_rectify:
                x = torch.relu(x - self.threshold) + self.threshold * (
                    x >= self.threshold
                ).to(x.dtype)
            x = (
                x
                if self.output_clamp_max is None
                else torch.clamp(x, max=self.output_clamp_max)
            )
            if self.sensory_input_mode == "replace":
                x = self._apply_sensory_input(
                    x,
                    self._sensory_input_after_layer(inputs, 0),
                )

        if manipulate is not None:
            x = x.clone()
            for b, _ in manipulate.items():
                if 0 in manipulate_idx[b]:
                    for neuron_idx, target_act in manipulate_idx[b][0].items():
                        x[neuron_idx, b] = target_act

        # ---- Remaining layers ----
        if return_layer_list:
            per_layer_acts = [x]

            if use_ckpt:
                for chunk_start in range(1, self.num_layers, checkpoint_steps):
                    chunk_end = min(chunk_start + checkpoint_steps, self.num_layers)
                    result = torch_checkpoint(
                        self._forward_chunk,
                        x,
                        chunk_start,
                        chunk_end,
                        inputs,
                        slopes_full,
                        biases_full,
                        taus_full,
                        weights,
                        use_reentrant=False,
                    )
                    x = result[0]
                    per_layer_acts.extend(result[1:])
            else:
                for alayer in range(1, self.num_layers):
                    x_previous = x.clone()
                    x = torch.sparse.mm(weights, x)
                    x = self.activation_function(
                        x,
                        x_previous=x_previous,
                        slopes_full=slopes_full,
                        biases_full=biases_full,
                        taus_full=taus_full,
                    )
                    if self._should_apply_sensory_after_layer(alayer):
                        x = self._apply_sensory_input(
                            x,
                            self._sensory_input_after_layer(inputs, alayer),
                        )
                        # x = torch.where(x >= self.threshold, x, torch.zeros_like(x))
                        # this propagates NaN honesty
                        if self.output_rectify:
                            x = torch.relu(x - self.threshold) + self.threshold * (
                                x >= self.threshold
                            ).to(x.dtype)
                        x = (
                            x
                            if self.output_clamp_max is None
                            else torch.clamp(x, max=self.output_clamp_max)
                        )
                        if self.sensory_input_mode == "replace":
                            x = self._apply_sensory_input(
                                x,
                                self._sensory_input_after_layer(inputs, alayer),
                            )
                    if manipulate is not None:
                        x = x.clone()
                        for b, _ in manipulate.items():
                            if alayer in manipulate_idx[b]:
                                for neuron_idx, target_act in manipulate_idx[b][
                                    alayer
                                ].items():
                                    x[neuron_idx, b] = target_act
                    per_layer_acts.append(x)

            del inputs, x
            torch.cuda.empty_cache()
            return per_layer_acts

        # ---- default (stacked) path ----
        if use_ckpt:
            act_chunks = [x.t().unsqueeze(-1)]
            for chunk_start in range(1, self.num_layers, checkpoint_steps):
                chunk_end = min(chunk_start + checkpoint_steps, self.num_layers)
                result = torch_checkpoint(
                    self._forward_chunk,
                    x,
                    chunk_start,
                    chunk_end,
                    inputs,
                    slopes_full,
                    biases_full,
                    taus_full,
                    weights,
                    use_reentrant=False,
                )
                x = result[0]
                # result[1:] are (neurons, batch) — transpose each and stack along time
                act_chunks.append(torch.stack([cx.t() for cx in result[1:]], dim=-1))
            self.activations = torch.cat(act_chunks, dim=-1).cpu()
        else:
            acts = [x.t().cpu()]
            for alayer in range(1, self.num_layers):
                x_previous = x.clone()
                x = torch.sparse.mm(weights, x)
                x = self.activation_function(
                    x,
                    x_previous=x_previous,
                    slopes_full=slopes_full,
                    biases_full=biases_full,
                    taus_full=taus_full,
                )
                if self._should_apply_sensory_after_layer(alayer):
                    x = self._apply_sensory_input(
                        x,
                        self._sensory_input_after_layer(inputs, alayer),
                    )
                    # x = torch.where(x >= self.threshold, x, torch.zeros_like(x))
                    # this propagates NaN honesty
                    if self.output_rectify:
                        x = torch.relu(x - self.threshold) + self.threshold * (
                            x >= self.threshold
                        ).to(x.dtype)
                    x = (
                        x
                        if self.output_clamp_max is None
                        else torch.clamp(x, max=self.output_clamp_max)
                    )
                    if self.sensory_input_mode == "replace":
                        x = self._apply_sensory_input(
                            x,
                            self._sensory_input_after_layer(inputs, alayer),
                        )
                if manipulate is not None:
                    x = x.clone()
                    for b, _ in manipulate.items():
                        if alayer in manipulate_idx[b]:
                            for neuron_idx, target_act in manipulate_idx[b][
                                alayer
                            ].items():
                                x[neuron_idx, b] = target_act
                acts.append(x.t().cpu())
            self.activations = torch.stack(acts, dim=-1)

        del inputs, x
        torch.cuda.empty_cache()

        if single_input:
            self.activations = self.activations.squeeze(0)
        return self.activations


# Note: input gradients are handled in the activation_maximisation and saliency
# functions themselves, so no context manager for that here
@contextlib.contextmanager
def training_mode(
    model: MultilayeredNetwork,
    train_slopes=True,
    train_biases=True,
    divisive_strength=True,
    train_tau=True,
):
    """Context manager for training mode - enables gradients for slopes and biases."""
    # If biases are about to be trained and any sit on the |x|=0 kink,
    # nudge them off so abs() gradients can flow. Without this, raw_biases
    # initialised at default_bias=0 stay stuck forever.
    if (
        train_biases
        and model.raw_biases is not None
        and getattr(model, "bias_transform", "abs") == "abs"
    ):
        with torch.no_grad():
            zero_mask = model.raw_biases == 0
            if zero_mask.any():
                model.raw_biases[zero_mask] = 1e-6

    model.set_param_grads(
        slopes=train_slopes,
        raw_biases=train_biases,
        divisive_strength=divisive_strength,
        tau=train_tau,
    )
    try:
        yield model
    finally:
        model.set_param_grads(
            slopes=False,
            raw_biases=False,
            divisive_strength=False,
            tau=False,
        )


def train_model(
    model: MultilayeredNetwork,
    inputs: torch.Tensor,
    targets: pd.DataFrame,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    param_reg_lambda: float = 0.01,
    wandb: bool = False,
    wandb_project_name: str = "connectome_interpreter",
    train_fraction: float = 0.8,
    seed: int = 42,
    train_slopes: bool = True,
    train_biases: bool = True,
    train_divisive_strength: bool = True,
    train_tau: bool = True,
    rescale_slope_updates: bool = False,
    project_parameter_bounds: bool = False,
    checkpoint_steps: int = 50,
    activation_loss_fn: Union[str, Callable] = "mse",
    initial_state: Optional[torch.Tensor] = None,
    target_node_groups: Optional[dict] = None,
    extra_parameters: Optional[List[torch.nn.Parameter]] = None,
    input_transform: Optional[Callable] = None,
    output_transform: Optional[Callable] = None,
):
    """
    Train the model to approximate the targets, while keeping the model parameter change
    minumum (param_reg_lambda decdes how strongly). The loss is the mean squared error
    (default) between the model activations and the target activations.

    Args:
        model (MultilayeredNetwork): The model to train.
        inputs (torch.Tensor): The input data. Shape: (batch_size, num_input_neurons,
            num_layers).
        targets (pd.DataFrame): The target activations. A DataFrame with columns:
            "batch", "neuron_idx", "layer" (optional), "value". The "layer" column
            specifies the timestep (0-indexed). If 'layer' is not present, the average
            activation across all timesteps is used.
        num_epochs (int, optional): Number of training epochs. Defaults to 100.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to
            0.01.
        param_reg_lambda (float, optional): Regularization parameter for the model
            parameters. Defaults to 0.01.
        wandb (bool, optional): Whether to use Weights & Biases logging.
        wandb_project_name (str, optional): Project name for wandb.
        train_fraction (float, optional): Fraction of data to use for training.
        seed (int, optional): Random seed for reproducibility.
        train_biases (bool, optional): Whether to train biases. Defaults to True.
            Note: biases use abs(raw_biases) internally to keep them positive. When
            enabled, any raw_biases sitting exactly at 0 are nudged to 1e-6 to break the
            abs() kink where gradients vanish.
        train_divisive_strength (bool, optional): Whether to train divisive strength. Defaults to True.
        train_tau (bool, optional): Whether to train tau. Defaults to True.
        train_slopes (bool, optional): Whether to train the slope gain. In pair mode
            (tuple-keyed slope_dict) this is the per-(pre, post) edge gain; in node
            mode the per-cell-type slope. Defaults to True.
        rescale_slope_updates (bool, optional): Whether to rescale each optimizer
            step for the (pair-mode) slope by model.slope_update_scale. Defaults to False.
        project_parameter_bounds (bool, optional): Whether to project model parameter
            bounds after each optimizer step if the model provides that hook. Defaults
            to False.
        activation_loss_fn (str or callable, optional): Loss function for activations.
            Either "mae", "mse" (default), or a callable with signature fn(pred:
            torch.Tensor, target: torch.Tensor) -> torch.Tensor returning a scalar loss.
        initial_state (torch.Tensor, optional): Initial full-network state for each
            batch. Shape can be (nodes,), (batch, nodes), or (nodes, batch). Sensory
            nodes are still overwritten by the first input sample.
        target_node_groups (dict, optional): Fit group (e.g. cell-type) averages
            instead of individual nodes. Maps ``group_id -> sequence of member node
            indices``. When provided, the ``targets`` DataFrame's ``neuron_idx``
            column holds ``group_id`` values, and each group's prediction is the mean
            activation over its member nodes (``outputs[:, members, :].mean(dim=1)``)
            before the loss is computed. Works for both ``LinearNetwork`` and
            ``MultilayeredNetwork``. Defaults to None (per-node targets, legacy
            behavior).
        extra_parameters (list of torch.nn.Parameter, optional): Additional
            trainable parameters that live outside ``model`` (e.g. a luminance
            log-offset epsilon used by ``input_transform``). They are added to the
            optimizer, included in the L1 parameter-change regularization (anchored
            to their value at call time), and gradient-clipped alongside the model
            parameters. Defaults to None.
        input_transform (callable, optional): ``fn(inputs) -> inputs`` applied to
            the (train and validation) input tensor inside every epoch, just before
            the forward pass. Re-applied each step so a differentiable transform
            driven by ``extra_parameters`` (e.g. ``log(brightness + epsilon)``)
            receives gradients. Defaults to None (inputs used as given).
        output_transform (callable, optional): ``fn(outputs) -> outputs`` applied
            to the model output ``(batch, num_neurons, num_layers)`` right after
            every forward pass (train and validation), before the targets are
            gathered. Used to append a fixed observation model to the dynamics --
            e.g. a causal sensor/indicator convolution along the time axis (see
            ``sensor_kernel.make_sensor_output_transform``) so the fit compares a
            forward-modelled measurement, not the raw latent, to the data. Must
            preserve the shape and be differentiable wrt ``outputs``. Defaults to
            None (outputs used as given).
    """
    if output_transform is not None and not callable(output_transform):
        raise TypeError("output_transform must be callable or None.")

    def train_test_split(inputs, targets, train_fraction):
        """Split the inputs and targets into training and validation sets."""
        if not 0 < train_fraction <= 1:
            raise ValueError("train_fraction must be in the interval (0, 1].")
        if train_fraction == 1:
            train_indices = np.arange(inputs.shape[0])
            val_indices = np.array([], dtype=int)
        else:
            train_num = max(1, int(inputs.shape[0] * train_fraction))
            # random selection of train indices
            train_indices = np.random.choice(
                range(inputs.shape[0]), train_num, replace=False
            )
            # target is the rest
            val_indices = np.setdiff1d(range(inputs.shape[0]), train_indices)

        train_inputs = inputs[train_indices]
        val_inputs = inputs[val_indices]

        train_targets = targets[targets["batch"].isin(train_indices)].copy()
        train_targets.loc[:, ["batch"]] = pd.Categorical(
            train_targets["batch"], categories=list(train_indices)
        )
        train_targets = train_targets.sort_values(by="batch")
        # change to local batch indices
        batch2local_batch = {b: i for i, b in enumerate(train_indices)}
        train_targets.loc[:, ["batch"]] = train_targets.batch.map(batch2local_batch)

        val_targets = targets[targets["batch"].isin(val_indices)].copy()
        val_targets.loc[:, ["batch"]] = pd.Categorical(
            val_targets["batch"], categories=list(val_indices)
        )
        val_targets = val_targets.sort_values(by="batch")
        # change to local batch indices
        batch2local_batch = {b: i for i, b in enumerate(val_indices)}
        val_targets.loc[:, ["batch"]] = val_targets.batch.map(batch2local_batch)

        return (
            train_inputs,
            val_inputs,
            train_targets,
            val_targets,
            train_indices,
            val_indices,
        )

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Extra (model-external) trainable parameters and optional input transform.
    extra_parameters = list(extra_parameters) if extra_parameters is not None else []
    if input_transform is not None and not callable(input_transform):
        raise TypeError("input_transform must be callable or None.")

    # Resolve the loss function once
    if isinstance(activation_loss_fn, str):
        loss_fns = {
            "mae": lambda pred, tgt: torch.abs(pred - tgt).mean(),
            "mse": lambda pred, tgt: ((pred - tgt) ** 2).mean(),
        }
        if activation_loss_fn not in loss_fns:
            raise ValueError(
                f"Unknown activation_loss_fn '{activation_loss_fn}'. "
                f"Use one of {list(loss_fns)} or pass a callable."
            )
        loss_fn = loss_fns[activation_loss_fn]
    elif callable(activation_loss_fn):
        loss_fn = activation_loss_fn
    else:
        raise TypeError(
            "activation_loss_fn must be a string or callable, "
            f"got {type(activation_loss_fn).__name__}"
        )

    # Check column names of targets - support both old and new format
    required_cols = ["batch", "neuron_idx", "value"]
    if not all(col in targets.columns for col in required_cols):
        raise ValueError(
            "Targets DataFrame must contain columns: 'batch', 'neuron_idx', 'value'"
            "Optionally include 'layer' for time series targets."
        )

    # Optional group-mean reduction: targets' "neuron_idx" holds group ids and each
    # group's prediction is the mean over its member node indices, so we fit a
    # population/cell-type average rather than individual nodes.
    group_order = None
    group_members = None
    group_id_to_row = None
    if target_node_groups is not None:
        if len(target_node_groups) == 0:
            raise ValueError("target_node_groups must be non-empty when provided.")
        group_order = list(target_node_groups.keys())
        group_id_to_row = {g: i for i, g in enumerate(group_order)}
        group_members = [
            torch.as_tensor(list(target_node_groups[g]), dtype=torch.long)
            for g in group_order
        ]
        for g, members in zip(group_order, group_members):
            if members.numel() == 0:
                raise ValueError(
                    f"target_node_groups[{g!r}] has no member node indices."
                )
        missing = set(targets["neuron_idx"].unique()) - set(group_order)
        if missing:
            raise ValueError(
                f"targets 'neuron_idx' values {sorted(missing)[:5]} are not keys of "
                "target_node_groups."
            )

    if wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Please install it using 'pip install wandb' or set wandb=False."
            ) from exc

        wandb.init(project=wandb_project_name)

    device = model.all_weights.device
    # Move data to device
    if not torch.is_tensor(inputs):
        inputs = torch.tensor(inputs, device=device)
    else:
        inputs = inputs.to(device)

    # Use training mode context manager
    with training_mode(
        model,
        train_slopes,
        train_biases,
        train_divisive_strength,
        train_tau,
    ):
        # Model parameters plus any model-external trainable parameters (e.g. epsilon).
        trainable_parameters = list(model.parameters()) + extra_parameters
        optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)

        # Training history
        history = {
            "loss": [],
            "activation_loss": [],
            "param_reg_loss": [],
            "val_loss": [],
        }

        initial_params = [param.clone() for param in trainable_parameters]

        # train validation split
        (
            train_inputs,
            val_inputs,
            train_targets,
            val_targets,
            train_indices,
            val_indices,
        ) = train_test_split(inputs, targets, train_fraction)

        batch_idx = torch.tensor(
            train_targets["batch"].astype(int).values,
            dtype=torch.long,
        )
        if target_node_groups is not None:
            neuron_idx = torch.tensor(
                train_targets["neuron_idx"].map(group_id_to_row).astype(int).values,
                dtype=torch.long,
            )
        else:
            neuron_idx = torch.tensor(
                train_targets["neuron_idx"].astype(int).values,
                dtype=torch.long,
            )
        if "layer" in train_targets.columns:
            layer_idx = torch.tensor(
                train_targets["layer"].astype(int).values,
                dtype=torch.long,
            )
        target_vals = torch.tensor(
            train_targets["value"].values,
            dtype=torch.float32,
        )

        has_validation = len(val_targets) > 0
        if has_validation:
            batch_idx_val = torch.tensor(
                val_targets["batch"].astype(int).values,
                dtype=torch.long,
            )
            if target_node_groups is not None:
                neuron_idx_val = torch.tensor(
                    val_targets["neuron_idx"].map(group_id_to_row).astype(int).values,
                    dtype=torch.long,
                )
            else:
                neuron_idx_val = torch.tensor(
                    val_targets["neuron_idx"].astype(int).values,
                    dtype=torch.long,
                )
            if "layer" in val_targets.columns:
                layer_idx_val = torch.tensor(
                    val_targets["layer"].astype(int).values,
                    dtype=torch.long,
                )
            target_vals_val = torch.tensor(
                val_targets["value"].values,
                dtype=torch.float32,
            )

        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()
            # Forward pass. The transform is re-applied each epoch so a
            # differentiable transform (driven by extra_parameters) gets gradients.
            outputs = model(
                train_inputs
                if input_transform is None
                else input_transform(train_inputs),
                checkpoint_steps=checkpoint_steps,
                initial_state=initial_state,
            )  # shape: (train_num, num_neurons, num_layers)
            if output_transform is not None:
                outputs = output_transform(outputs)

            output_device = outputs.device
            if target_node_groups is not None:
                # Reduce to per-group means: (batch, num_groups, num_layers).
                outputs = torch.stack(
                    [
                        outputs[:, members.to(output_device), :].mean(dim=1)
                        for members in group_members
                    ],
                    dim=1,
                )
            batch_idx_out = batch_idx.to(output_device)
            neuron_idx_out = neuron_idx.to(output_device)
            if "layer" in train_targets.columns:
                actual = outputs[
                    batch_idx_out,
                    neuron_idx_out,
                    layer_idx.to(output_device),
                ]
            else:
                actual = outputs[batch_idx_out, neuron_idx_out, :].mean(dim=-1)
            activation_loss = loss_fn(actual, target_vals.to(output_device))

            # regularization loss
            param_reg_loss = 0
            for param, param0 in zip(trainable_parameters, initial_params):
                param_reg_loss += (param - param0).abs().mean()

            param_reg_loss = param_reg_lambda * param_reg_loss

            loss = activation_loss + param_reg_loss
            loss.backward()
            grad_params = [p for p in trainable_parameters if p.requires_grad]
            # Skip the step on a non-finite loss OR a non-finite gradient. The
            # gradient check must precede clip_grad_norm_, because clipping an inf
            # gradient produces NaN (inf * clip_coef, with clip_coef -> 0), which
            # optimizer.step() would then write permanently into the parameters.
            grads_finite = all(
                p.grad is None or torch.isfinite(p.grad).all() for p in grad_params
            )
            if not torch.isfinite(loss) or not grads_finite:
                reason = (
                    "non-finite loss"
                    if not torch.isfinite(loss)
                    else "non-finite gradient"
                )
                print(f"[epoch {epoch}] {reason}, skipping step")
                optimizer.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(grad_params, max_norm=1.0)
            slope_before = (
                model.slope.detach().clone()
                if rescale_slope_updates
                and getattr(model, "slope", None) is not None
                and getattr(model, "slope_is_pairwise", False)
                else None
            )
            optimizer.step()
            if slope_before is not None:
                model.rescale_slope_update_(slope_before)
            if project_parameter_bounds and hasattr(model, "project_parameter_bounds_"):
                model.project_parameter_bounds_()

            # validation loss
            if has_validation:
                with torch.no_grad():
                    val_outputs = model(
                        val_inputs
                        if input_transform is None
                        else input_transform(val_inputs),
                        initial_state=initial_state,
                    )
                    if output_transform is not None:
                        val_outputs = output_transform(val_outputs)
                    val_output_device = val_outputs.device
                    if target_node_groups is not None:
                        val_outputs = torch.stack(
                            [
                                val_outputs[:, members.to(val_output_device), :].mean(
                                    dim=1
                                )
                                for members in group_members
                            ],
                            dim=1,
                        )
                    batch_idx_val_out = batch_idx_val.to(val_output_device)
                    neuron_idx_val_out = neuron_idx_val.to(val_output_device)
                    if "layer" in val_targets.columns:
                        actual = val_outputs[
                            batch_idx_val_out,
                            neuron_idx_val_out,
                            layer_idx_val.to(val_output_device),
                        ]
                    else:
                        actual = val_outputs[
                            batch_idx_val_out, neuron_idx_val_out, :
                        ].mean(dim=-1)
                    val_activation_loss = loss_fn(
                        actual, target_vals_val.to(val_output_device)
                    )
                    if wandb:
                        wandb.log({"val_activation_loss": val_activation_loss.item()})
            else:
                val_activation_loss = activation_loss.detach()

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: Loss = {loss.item()}, "
                    f"Activation Loss = {activation_loss.item()}, "
                    f"Parameter Regularization Loss = {param_reg_loss.item()}, "
                    f"Validation Activation Loss = {val_activation_loss.item()}"
                )

            if wandb:
                wandb.log(
                    {
                        "learning_rate": learning_rate,
                        "param_reg_lambda": param_reg_lambda,
                        "activation_loss": activation_loss.item(),
                        "param_reg_loss": param_reg_loss.item(),
                    }
                )

            history["loss"].append(loss.item())
            history["activation_loss"].append(activation_loss.item())
            history["param_reg_loss"].append(param_reg_loss.item())
            history["val_loss"].append(val_activation_loss.item())

    # Free up memory
    torch.cuda.empty_cache()
    # clear the computational graph of activations
    with torch.no_grad():
        model.activations = model.activations.detach()

    # return the model and the training history
    return (
        model,
        history,
        train_inputs,
        val_inputs,
        train_targets,
        val_targets,
        train_indices,
        val_indices,
    )


def activation_maximisation(
    model: MultilayeredNetwork,
    target_activations: TargetActivation,
    input_tensor: Optional[torch.Tensor] = None,
    num_iterations: int = 50,
    learning_rate: float = 0.1,
    # regularization
    in_reg_lambda: float = 0.01,
    out_reg_lambda: float = 0.01,
    custom_reg_functions: Optional[
        Dict[str, Callable[[torch.Tensor], torch.Tensor]]
    ] = None,
    # early stopping
    early_stopping: bool = True,
    stopping_threshold: float = 1e-5,
    n_runs: int = 10,
    use_tqdm: bool = True,
    print_output: bool = True,
    device: Optional[torch.device] = None,
    wandb: bool = False,
    seed: Optional[int] = None,
    normalize_gradients: bool = False,
    quiet: bool = False,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[float],
    List[float],
    List[float],
    List[np.ndarray],
]:
    """
    Performs activation maximisation on a given model to identify input patterns that
    result in the target activations.

    This is done by adjusting the input tensor over `num_iterations` using gradient
    descent, while also regularising the overall input and output (to keep activated
    neurons sparse).

    Args:
        model: A PyTorch model with `activations`, `sensory_indices`, and `threshold`
            attributes.
        target_activations (TargetActivation): Target activations specification.
        input_tensor (torch.Tensor, optional): The initial tensor to optimize. If None,
            a random tensor is created.  Defaults to None.
        num_iterations (int, optional): The number of iterations to run the optimization
            for. Defaults to 50.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults
            to 0.1.
        in_reg_lambda (float, optional): The coefficient for input regularization.
            Defaults to 0.01.
        out_reg_lambda (float, optional): The coefficient for output regularization.
            Defaults to 0.01.
        custom_reg_functions (Dict[str, Callable[[torch.Tensor]], optional): A
            dictionary with keys 'in' and 'out' that map to functions that calculate the
            input and output regularization losses, respectively. If None, the default
            regularization function (L1 plus L2) is used. Defaults to None.
        early_stopping (bool, optional): Whether to stop the optimization early if the
            difference between the biggest and the smallest loss within the last n_runs
            falls below `stopping_threshold`. Defaults to True.
        stopping_threshold (float, optional): The threshold for early stopping. Defaults
            to 1e-5.
        n_runs (int, optional): The number of runs to consider for early stopping.
            Defaults to 10.
        report_memory_usage (bool, optional): Whether to report GPU memory usage during
            optimization. Defaults to False.
        device: The device to run the optimization on. If None, automatically selects a
            device. Defaults to None.
        wandb (bool, optional): Whether to log optimization details to Weights & Biases
            (https://wandb.ai/site/). Defaults to True.  Requires wandb to be installed.
        seed (int, optional): Random seed for reproducible input tensor initialization.
            Only used when input_tensor is None. Defaults to None (no seed set).
        normalize_gradients (bool, optional): Whether to normalize gradients per
            timepoint to mitigate vanishing gradients for longer path lengths. When
            True, gradients are normalized by their L2 norm for each timepoint
            independently. This helps ensure that inputs targeting neurons at deeper
            layers receive comparable gradient magnitudes to those at earlier layers.
            Defaults to False.

    Returns:
        tuple: A tuple containing:

            - numpy.ndarray: The optimized input as a numpy array.
            - numpy.ndarray: The output of the model after optimization as a numpy array.
            - list(float): A list of output activation losses over iterations.
            - list(float): A list of input activation regularization losses over iterations.
            - list(float): A list of output activation regularization losses over iterations.
            - list(numpy.ndarray): A list of input tensor snapshots taken during optimization.


    Examples:

        .. code-block:: python

            # Single target for multiple batches
            targets_dict = {
                # layer 0: neuron 1 -> 0.5, neuron 2 -> 0.8
                0: {1: 0.5, 2: 0.8},
                # layer 1: neuron 0 -> 0.3
                1: {0: 0.3}
            }
            targets = TargetActivation(targets=targets_dict, batch_size=4)
            inputs, outputs, *losses = activation_maximisation(model, targets)

            # Different targets per batch using DataFrame
            targets_df = pd.DataFrame([
                {'batch': 0, 'layer': 0, 'neuron': 1, 'value': 0.5},
                {'batch': 0, 'layer': 0, 'neuron': 2, 'value': 0.8},
                {'batch': 1, 'layer': 1, 'neuron': 0, 'value': 0.3}
            ])
            # batch_size inferred
            targets = TargetActivation(targets=targets_df)
            results = activation_maximisation(model, targets)

            # Custom regularization
            def sparse_reg(x):
                return torch.sum(torch.abs(x))
            custom_reg = {'in': sparse_reg, 'out': sparse_reg}
            results = activation_maximisation(
                model, targets, custom_reg_functions=custom_reg
            )
    """

    def default_reg(x):
        return torch.norm(x, p=1) + torch.norm(x, p=2)

    def calculate_activation_loss(activations, batch_idx):
        """Calculate loss for specific batch using its target activations"""
        batch_idx = int(batch_idx)
        batch_targets = target_activations.get_batch_targets(batch_idx)
        loss = torch.tensor(0.0, device=device)
        n_neurons = 0

        for layer, neuron_targets in batch_targets.items():
            for neuron_index, target_value in neuron_targets.items():
                actual_value = activations[batch_idx, int(neuron_index), int(layer)]
                loss += (actual_value - target_value) ** 2
                n_neurons += 1

        return loss / n_neurons if n_neurons > 0 else loss

    # Setup and validation code
    if wandb:
        try:
            import wandb as wandb_lib
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Please install it using 'pip install wandb' or set wandb=False."
            ) from exc
        wandb_lib.init(project="connectome_interpreter")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = target_activations.batch_size

    if input_tensor is None:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        input_tensor = torch.rand(
            (batch_size, len(model.sensory_indices), model.num_layers),
            requires_grad=True,
            device=device,
        )
    else:
        # if np.ndarray, convert to tensor
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.tensor(
                input_tensor, device=device, dtype=torch.float32
            )

        if input_tensor.shape != (
            batch_size,
            len(model.sensory_indices),
            model.num_layers,
        ):
            raise ValueError(
                f"Expected input shape (batch_size={batch_size}, "
                f"num_input_neurons={len(model.sensory_indices)}, "
                f"num_layers={model.num_layers})"
            )
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([input_tensor], lr=learning_rate)

    input_snapshots = []
    act_loss = []
    out_reg_losses = []
    in_reg_losses = []
    losses = []

    for iteration in tqdm(range(num_iterations), disable=quiet):
        optimizer.zero_grad()
        # Forward pass
        _ = model(input_tensor)

        # Take snapshot every 5 iterations
        if iteration % 5 == 0:
            snapshot = input_tensor.clone().detach().cpu().numpy()
            snapshot = np.where(snapshot >= model.threshold, snapshot, 0)
            snapshot = np.tanh(snapshot)
            input_snapshots.append(snapshot)

        # Calculate activation loss
        activation_loss = torch.mean(
            torch.stack(
                [
                    calculate_activation_loss(model.activations, batch_idx)
                    for batch_idx in range(batch_size)
                ]
            )
        )

        # Regularization loss
        if custom_reg_functions and "in" in custom_reg_functions:
            in_reg_loss = in_reg_lambda * custom_reg_functions["in"](input_tensor)
        else:
            in_reg_loss = in_reg_lambda * default_reg(input_tensor)

        if custom_reg_functions and "out" in custom_reg_functions:
            out_reg_loss = out_reg_lambda * custom_reg_functions["out"](
                model.activations
            )
        else:
            out_reg_loss = out_reg_lambda * default_reg(model.activations)

        loss = activation_loss + in_reg_loss + out_reg_loss
        losses.append(loss.item())

        # Early stopping
        if early_stopping and (iteration > n_runs):
            if np.max(losses[-n_runs:]) - np.min(losses[-n_runs:]) < stopping_threshold:
                break

        if wandb:
            wandb_lib.log(
                {
                    "activation_loss": activation_loss.item(),
                    "in_regularisation_loss": in_reg_loss.item(),
                    "out_regularisation_loss": out_reg_loss.item(),
                    "loss": loss.item(),
                }
            )

        act_loss.append(activation_loss.item())
        out_reg_losses.append(out_reg_loss.item())
        in_reg_losses.append(in_reg_loss.item())

        # Backward pass and optimization
        loss.backward()

        # Normalize gradients per timepoint to mitigate vanishing gradients
        if normalize_gradients:
            with torch.no_grad():
                # Compute L2 norm per timepoint (last dim), keepdim for broadcasting
                # input_tensor.grad shape: (batch, neurons, timepoints)
                norms = input_tensor.grad.norm(dim=(0, 1), keepdim=True) + 1e-8
                input_tensor.grad.div_(norms)

        optimizer.step()

        if not quiet and iteration % 10 == 0:
            print(
                f"Iteration {iteration}: Activation Loss = {activation_loss.item()}, "
                f"Input regularization Loss = {in_reg_loss.item()}, "
                f"Output regularization Loss = {out_reg_loss.item()}"
            )

        torch.cuda.empty_cache()

    # Final processing
    if not quiet:
        print(
            f"Final - Activation loss: {act_loss[-1]}, Input reg: {in_reg_losses[-1]}, Output reg: {out_reg_losses[-1]}"
        )

    input_tensor = torch.clamp(input_tensor, -1.0, 1.0)

    output_after = model(input_tensor).cpu().detach().numpy()
    input_tensor = input_tensor.cpu().detach().numpy()

    # Clear computational graph
    with torch.no_grad():
        model.activations = model.activations.detach()

    # Handle single batch case
    if batch_size == 1:
        input_tensor = input_tensor[0]
        output_after = output_after[0]
        input_snapshots = [snap[0] for snap in input_snapshots]

    return (
        input_tensor,
        output_after,
        act_loss,
        out_reg_losses,
        in_reg_losses,
        input_snapshots,
    )


def saliency(
    model: MultilayeredNetwork,
    input_tensor: torch.Tensor,
    neurons_of_interest: Dict[int, List[int]],
    method: str = "vanilla",
    normalize: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Computes saliency maps: given the current input, to what extent will each element's
    change result in change in the activations of the neurons of interest?

    Args:
        model: A MultilayeredNetwork model
        input_tensor: Input tensor to analyze
        neurons_of_interest: Dictionary mapping layer indices to lists of neuron indices
            to analyze: {layer_idx: [neuron_indices]}. Layer_idx = 0 corresponds to the
            first layer after the input layer.
        method: Saliency computation method ("vanilla" or "input_x_gradient")
        normalize: Whether to normalize saliency maps by their maximum value
        device: Computation device

    Returns:
        torch.Tensor:
            Saliency maps with same shape as input tensor
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensor if needed
    if isinstance(input_tensor, np.ndarray):
        input_tensor = torch.tensor(input_tensor, device=device)

    # Ensure input requires gradients
    input_tensor = input_tensor.clone().detach().to(device)
    input_tensor.requires_grad = True

    # Run forward pass
    _ = model(input_tensor)

    # Sum activations of neurons we care about
    activation_signal = 0
    for layer, neuron_indices in neurons_of_interest.items():
        for neuron_idx in neuron_indices:
            if input_tensor.dim() == 3:  # batched input
                for batch_idx in range(input_tensor.shape[0]):
                    activation_signal = (
                        activation_signal
                        + model.activations[batch_idx, neuron_idx, layer]
                    )
            else:  # single input
                activation_signal = (
                    activation_signal + model.activations[neuron_idx, layer]
                )

    # Compute gradients by backpropagation
    activation_signal.backward()

    # Get the gradients based on selected method
    if method == "vanilla":
        saliency_maps = input_tensor.grad.clone()
    elif method == "input_x_gradient":
        saliency_maps = input_tensor * input_tensor.grad
    else:
        raise ValueError(f"Unknown saliency method: {method}")

    if normalize and torch.max(saliency_maps) > 0:
        if saliency_maps.dim() == 3:  # Normalize each batch separately
            for i in range(saliency_maps.shape[0]):
                if torch.max(saliency_maps[i]) > 0:
                    saliency_maps[i] = saliency_maps[i] / torch.max(saliency_maps[i])
        else:
            saliency_maps = saliency_maps / torch.max(saliency_maps)

    return saliency_maps


def get_gradients(
    model: MultilayeredNetwork,
    input_tensor: torch.Tensor,
    monitor_neurons: arrayable,
    target_neurons: Dict[int, List[Union[int, str]]],
    monitor_layers: Optional[List[int]] = None,
    method: str = "vanilla",
    batch_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    checkpoint_steps: int = 50,
    normalize_per_layer: bool = False,
) -> pd.DataFrame:
    """
    Compute gradients of target neurons with regard to monitor neurons, i.e. the rate of
    change of target neuron activations with respect to monitor neurons.

    Args:
        model: MultilayeredNetwork instance.
        input_tensor: Input tensor to the model. Shape: (batch_size, num_input_neurons,
            num_timesteps) or (num_input_neurons, num_timesteps)
        monitor_neurons: List of neuron indices to monitor gradients for. If `model`'s
            `idx_to_group` attribute is not None (e.g. idx_to_type), these should be
            group names.
        target_neurons: Dictionary mapping layer/timepoint indices to lists of neurons.
            If `model`'s `idx_to_group` attribute is not None (e.g. idx_to_type), these
            should be group names. e.g. {5: ['DA1_lPN']} means calculating gradients of
            layer 5 neurons in group 'DA1_lPN'.
        monitor_layers: List of layer indices to monitor gradients at. If None, monitor
            all layers.
        method: Gradient computation method. "vanilla" or "input_x_gradient".
        batch_names: Optional list of batch names corresponding to the input tensor.
            If not provided, defaults to ["batch_0", "batch_1", ...].
        device: Optional torch device to run the computations on. If None, uses CUDA if
            available, else CPU.
        checkpoint_steps (int): Number of layers to include in each checkpoint when
            using gradient checkpointing. Adjust based on available memory and model
            size. Defaults to 50.
        normalize_per_layer (bool): If True, divide each layer's gradient tensor by its
            own L2 norm before returning. This mitigates exploding/vanishing gradients
            across time (early layers can have gradients many orders of magnitude larger
            than late ones when the per-step Jacobian's spectral radius is far from 1).
            With this enabled, cross-layer comparisons reflect the *relative pattern* of
            which inputs matter most at each layer, not the absolute magnitude. Defaults
            to False.

    Returns:
        pd.DataFrame:
            DataFrame containing gradients with columns: 'group', 'batch_name', 'time_0',
            'time_1', ..., 'time_N'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert method in ["vanilla", "input_x_gradient"]

    if isinstance(input_tensor, np.ndarray):
        input_tensor = torch.tensor(input_tensor, device=device)
    input_tensor = input_tensor.clone().detach().to(device)
    input_tensor.requires_grad = True

    monitor_neurons = to_nparray(monitor_neurons)
    target_neurons = {
        layer: to_nparray(neurons) for layer, neurons in target_neurons.items()
    }

    if model.idx_to_group is not None:
        # turn into indices
        target_neurons = {
            layer: [
                idx for idx, grp in model.idx_to_group.items() if grp in set(n.tolist())
            ]
            for layer, n in target_neurons.items()
        }
        monitor_set = set(monitor_neurons.tolist())
        monitor_indices = [
            idx for idx, grp in model.idx_to_group.items() if grp in monitor_set
        ]
    else:
        monitor_indices = list(monitor_neurons)

    if monitor_layers is None:
        monitor_layers = list(range(model.num_layers))
    else:
        monitor_layers = sorted(set(int(l) for l in monitor_layers))

    layer_acts = model(
        input_tensor,
        checkpoint_steps=checkpoint_steps,
        return_layer_list=True,
    )
    # each layer_acts[L]: (neurons, batch), on the forward path

    # Target CAN be a slice — it's a descendant of all earlier layer_acts,
    # so walking back from it will traverse every earlier layer_acts tensor.
    target_pieces = [
        layer_acts[layer][neuron_indices, :]
        for layer, neuron_indices in target_neurons.items()
    ]
    target = torch.cat(target_pieces)

    # Probes MUST be the full path tensors, not slices. A slice is a fresh
    # child of layer_acts[L]; it's not on any forward path to target, so
    # autograd.grad would return None. Slice the gradient afterwards instead.
    probes = [layer_acts[L] for L in monitor_layers]

    grads = torch.autograd.grad(target.mean(), probes, allow_unused=True)

    noted_grads = {}
    for layer, g, probe in zip(monitor_layers, grads, probes):
        if g is None:
            # Only happens if probe layer is at or after target (not upstream)
            g = torch.zeros_like(probe)
        if normalize_per_layer:
            g = g / (g.norm() + 1e-12)
        # g shape: (neurons, batch). Select monitor rows NOW (safe — it's a
        # plain tensor op on a detached gradient, no autograd implications).
        noted_grads[layer] = g[monitor_indices, :].detach().cpu()

    if batch_names is None:
        batch_names = (
            [f"batch_{i}" for i in range(input_tensor.shape[0])]
            if input_tensor.dim() == 3
            else ["batch_0"]
        )

    dfs = []
    for layer in monitor_layers:
        df = pd.DataFrame(
            noted_grads[layer].numpy(),
            columns=batch_names,
            index=(
                monitor_indices
                if model.idx_to_group is not None
                else list(monitor_neurons)
            ),
        )
        if model.idx_to_group is not None:
            # turn back to groups
            df.index = df.index.map(model.idx_to_group)
            df = df.groupby(df.index).sum()
        df = df.melt(
            var_name="batch_name", value_name="grad", ignore_index=False
        ).reset_index()
        df.rename(columns={"index": "group"}, inplace=True)
        df["layer"] = layer
        dfs.append(df)

    dfs = pd.concat(dfs, ignore_index=True)
    dfs = (
        dfs.pivot(index=["group", "batch_name"], columns="layer", values="grad")
        .reset_index()
        .rename(columns=lambda x: f"time_{x}" if isinstance(x, int) else x)
    )

    if method == "input_x_gradient":
        with torch.no_grad():
            out = model(input_tensor, checkpoint_steps=checkpoint_steps)
        acts = get_neuron_activation(
            out, monitor_neurons, batch_names, model.idx_to_group
        )
        dfs = (
            dfs.set_index(["group", "batch_name"]).sort_index()
            * acts.set_index(["group", "batch_name"]).sort_index()
        ).reset_index()

    torch.cuda.empty_cache()

    return dfs


def activations_to_df(
    inprop,
    model_input: np.ndarray,
    out: np.ndarray,
    sensory_indices: List[int],
    inidx_mapping: dict | None = None,
    outidx_mapping: dict | None = None,
    activation_threshold: float = 0,
    connectivity_threshold: float = 0,
    high_ram: bool = True,
) -> pd.DataFrame:
    """
    Generates a dataframe representing the paths in a layered plot,filtering
    by activation and connectivity thresholds.

    This function takes the direct connectivity matrix (inprop), input
    neuron activity, output neuron activity, indices for sensory
    neurons, and mapping between input and output indices to groups. It
    generates a dataframe that represents the paths through the network
    layers.

    Args:
        inprop (scipy.sparse matrix or numpy.ndarray): Matrix
            representing the synaptic strengths between neurons, can be
            dense or sparse. Presynaptic is in the rows, postsynaptic in the
            columns.
        model_input (numpy.ndarray): A 2D array representing input to
            the network. Neurons are in the rows, timepoints in the columns.
            Only the first timepoint is used, since `out` is expected to
            have activity of all neurons, including input neurons.
        out (numpy.ndarray): A 2D array representing the output from the
            network. The second dimension represents timepoints.
        sensory_indices (list of int): A list of indices corresponding
            to sensory neurons in `inprop`.
        inidx_mapping (dict, optional): A dictionary mapping indices in
            `inprop` to new indices (e.g. cell type). If None, indices are
            not remapped. Defaults to None.
        outidx_mapping (dict, optional): A dictionary mapping indices in `out`
            to new indices. If None, `inidx_mapping` is used for mapping.
            Defaults to None.
        activation_threshold (float, optional): A threshold value for
            activation. Neurons with activations below this threshold are not
            considered. Defaults to 0.
        connectivity_threshold (float, optional): A threshold for filtering
            connections. Connections with weights below this threshold are
            ignored. Defaults to 0.
        high_ram (bool, optional): Whether to use a high RAM implementation
            (which is slightly faster). This implementation gets direct
            connections between *all* relevant neurons at once, instead of
            within each layer. Defaults to True.

    Returns:
        pandas.DataFrame:
            A dataframe representing the paths in the network. Each row is a connection,
            with columns for 'pre' and 'post' neuron indices, 'layer', and their
            respective activations ('pre_activation', 'post_activation').
    """
    all_indices = list(range(inprop.shape[0]))

    if inidx_mapping is None:
        inidx_mapping = {idx: idx for idx in all_indices}

    if outidx_mapping is None:
        outidx_mapping = inidx_mapping

    # move to CPU in case it's still on GPU
    if torch.is_tensor(model_input):
        model_input = model_input.cpu().numpy()
    if torch.is_tensor(out):
        out = out.detach().cpu().numpy()

    print("Getting activations...")
    # get activations from input and out, based on the mappings provided by
    # inidx_mapping and outidx_mapping
    sensory_act = get_activations(
        model_input,
        sensory_indices,
        inidx_mapping,
        threshold=activation_threshold,
    )
    all_act = get_activations(
        out, all_indices, outidx_mapping, threshold=activation_threshold
    )

    print(
        "Getting connectivity... If this takes a while, consider increasing "
        "the connectivity/activation threshold."
    )

    if high_ram:
        # get all pre and post indices across layers
        post_groups = {key for _, d in all_act.items() for key in d}
        pre_groups = set(sensory_act[0].keys()).union(post_groups)
        pre_indices = [idx for idx, key in inidx_mapping.items() if key in pre_groups]
        post_indices = [
            idx for idx, key in outidx_mapping.items() if key in post_groups
        ]

        # if there are literally no neurons
        if len(pre_indices) == 0 or len(post_indices) == 0:
            raise ValueError(
                "No neurons found. Consider lowering the activation threshold."
            )

        # get connectivity
        conn = result_summary(
            inprop,
            inidx=pre_indices,
            outidx=post_indices,
            inidx_map=inidx_mapping,
            outidx_map=outidx_mapping,
            display_output=False,
        )
        # turn to edgelist, and filter
        conn_el = adjacency_df_to_el(conn, threshold=connectivity_threshold)

    # make paths df
    paths = []
    for layer in range(out.shape[1]):
        if layer == 0:
            # the initial layer only has input neuron activations
            pre = sensory_act[layer]
            post = all_act[layer]
        else:
            # all the input, not just the optimised external input
            pre = all_act[layer - 1]
            post = all_act[layer]

        pre_stringkeys = {str(key): val for key, val in pre.items()}
        post_stringkeys = {str(key): val for key, val in post.items()}

        if high_ram:
            # index the big connectivity matrix for this layer
            connections = conn_el[
                conn_el.pre.isin(pre_stringkeys.keys())
                & conn_el.post.isin(post_stringkeys.keys())
            ]

        else:
            # pre and post are already grouped by inidx and outidx_mapping
            # so need to recover the indices using pre, inidx_map, post,
            # outidx_map
            pre_indices = [
                idx for idx, val in inidx_mapping.items() if val in pre.keys()
            ]
            post_indices = [
                idx for idx, val in outidx_mapping.items() if val in post.keys()
            ]

            conn = result_summary(
                inprop,
                inidx=pre_indices,
                outidx=post_indices,
                inidx_map=inidx_mapping,
                outidx_map=outidx_mapping,
                display_output=False,
            )
            # turn to edgelist, and filter
            connections = adjacency_df_to_el(conn, threshold=connectivity_threshold)

        # so that direct connectivity is layer 1
        connections.loc[:, ["layer"]] = layer + 1
        connections.loc[:, ["pre_activation"]] = connections.pre.map(pre_stringkeys)
        connections.loc[:, ["post_activation"]] = connections.post.map(post_stringkeys)
        if connections.shape[0] > 0:
            paths.append(connections)
        else:
            print(f"Warning: No connections found in layer {layer+1}.")
    paths = pd.concat(paths)
    return paths


def activations_to_df_batched(
    inprop,
    opt_in: np.ndarray,
    out: np.ndarray,
    sensory_indices: List[int],
    inidx_mapping: dict | None = None,
    outidx_mapping: dict | None = None,
    activation_threshold: float = 0,
    connectivity_threshold: float = 0,
    high_ram: bool = True,
) -> pd.DataFrame:
    """
    Generates a dataframe representing the paths in a layered plot,
    filtering by activation and connectivity thresholds.

    This function takes the direct connectivity matrix (inprop), optimal
    input neuron activity, output neuron activity, indices for sensory
    neurons, and mapping between input and output indices to groups. It
    generates a dataframe that represents the paths through the network
    layers.

    Args:
        inprop (scipy.sparse matrix or numpy.ndarray): Matrix representing the
            synaptic strengths between neurons, can be dense or sparse.
            Presynaptic is in the rows, postsynaptic in the columns.
        opt_in (numpy.ndarray): A **3D** array representing optimal input to
            the network. The first dimension represents the batch size, the
            second dimension represents input neurons, and the third dimension
            represents timepoints. Only the first timepoint is used, since
            `out` is expected to have activity of all neurons, including input
            neurons.
        out (numpy.ndarray): A **3D** array representing the activation of all
            neurons. The first dimension represents the batch size, the second
            dimension represents all neurons, and the third dimension
            represents timepoints.
        sensory_indices (list of int): A list of indices corresponding to
            sensory neurons in `inprop`.
        inidx_mapping (dict, optional): A dictionary mapping indices in
            `inprop` to new indices (e.g. cell type). If None, indices are
            not remapped. Defaults to None.
        outidx_mapping (dict, optional): A dictionary mapping indices in `out`
            to new indices. If None, `inidx_mapping` is used for mapping.
            Defaults to None.
        activation_threshold (float, optional): A threshold value for
            activation. Neurons with activations below this threshold are not
            considered. Defaults to 0.
        connectivity_threshold (float, optional): A threshold for filtering
            connections. Connections with weights below this threshold are
            ignored. Defaults to 0.
        high_ram (bool, optional): Whether to use a high RAM implementation
            (which is slightly faster). This implementation gets direct
            connections between *all* relevant neurons at once, instead of
            within each layer. Defaults to True.

    Returns:
        pandas.DataFrame:
            A dataframe representing the paths in the network. Each row is a connection,
            with columns for 'pre' and 'post' neuron indices, 'layer', and their
            respective activations ('pre_activation', 'post_activation').
    """
    # use activations_to_df for each batch
    paths = []
    # assume the first dimnesion of opt_in and out is the batch size
    for i in tqdm(range(opt_in.shape[0])):
        path = activations_to_df(
            inprop,
            opt_in[i],
            out[i],
            sensory_indices,
            inidx_mapping,
            outidx_mapping,
            activation_threshold,
            connectivity_threshold,
            high_ram,
        )
        path.loc[:, ["batch"]] = i
        paths.append(path)
    return pd.concat(paths)


def input_from_df(
    df: pd.DataFrame,
    sensory_indices: list,
    idx_to_group: dict,
    num_layers: int,
    timepoints: Union[int, List[int], np.ndarray, set] = 0,
    dtype: type = np.float32,
) -> np.ndarray:
    """
    Make well-formatted input for the model, based on defined vectors of
    input neuron activation (df). The function returens a 3D tensor,
    with the first dimension being the batch size, the second dimension
    being the number of input neurons of the model, and the third
    dimension being the number of layers.

    Args:
        df : pd.DataFrame
            Rows correspond to the values in `idx_to_group`, and columns
            correspond to the batches (number of columns = number of batch).
            For instance, rows of df can be olfactory glomeruli, and columns
            can be odours. df is the reaction of each glomerulus to each odour.
        sensory_indices : list
            The indices of sensory neurons. For instance, these could be the
            indices of individual olfactory receptor neurons.
        idx_to_group : dict
            A dictionary that maps indices to group. For instance, this could
            map from indices of indivudal olfactory receptor neuron to the
            glomerulus they innervate (rows of df).
        num_layers : int
            The number of layers in the model.
        timepoints : int or list or np.ndarray or set, optional
            The timepoints to put the input in. For instance, if timepoints=[0, 1], then the same input will be put in timepoint 0 and 1. Defaults to 0.
        dtype : type, optional
            The data type of the output array. Defaults to np.float32.

    Returns:
        np.ndarray:
            The input for the model.

    """
    # first initialise empty input
    inarray = np.zeros((df.shape[1], len(sensory_indices), num_layers), dtype=dtype)

    # Ensure timepoints is iterable
    if isinstance(timepoints, int):
        timepoints = [timepoints]
    elif isinstance(timepoints, (np.ndarray, set)):
        timepoints = list(timepoints)

    for l in range(df.shape[1]):
        grp2act = df.iloc[:, l]
        grp2act = dict(zip(grp2act.index, grp2act.values.flatten()))
        actvec = [
            grp2act[idx_to_group[idx]] if idx_to_group[idx] in grp2act else 0
            for idx in sensory_indices
        ]
        for t in timepoints:
            inarray[l, :, t] = actvec
    return inarray


def get_neuron_activation(
    activations: torch.Tensor | npt.NDArray,
    neuron_indices: arrayable,
    batch_names: arrayable | None = None,
    idx_to_group: dict | None = None,
) -> pd.DataFrame:
    """
    Get the activations for specified indices across timepoints, include batch name and
    group information when available. Memory efficiency provided by Claude.

    Args:
        activations (torch.Tensor | numpy.ndarray): Output activation from the model.
            Shape should be (batch_size, num_neurons, num_timepoints) or (num_neurons,
            num_timepoints).
        neuron_indices (arrayable): The indices of the neurons to get activations for.
        batch_names (arrayable, optional): The names of the batches. Defaults to None.
            If activations.ndim == 3, then this should be supplied. If not, batch names
            will be e.g. 'batch_0', 'batch_1', etc.
        idx_to_group (dict, optional): A dictionary mapping indices to groups. Defaults
            to None.

    Returns:
        pd.DataFrame:
            The activations for the neurons, with the first columns being batch_names,
            neuron_indices, and group. The rest are the timesteps.
    """
    # CPU torch -> numpy is a zero-copy view; only GPU tensors copy.
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()

    n_timepoints = activations.shape[-1]
    in_dtype = activations.dtype

    if idx_to_group is None:
        idx_to_group = {idx: idx for idx in range(activations.shape[-2])}

    # Bucket once. Insertion order = output row order.
    group_to_indices: dict[object, list[int]] = defaultdict(list)
    for idx in neuron_indices:
        group_to_indices[idx_to_group[idx]].append(idx)
    groups = list(group_to_indices.keys())
    n_groups = len(groups)

    # ---- 2D ---------------------------------------------------------------
    if activations.ndim == 2:
        if batch_names is not None:
            print("batch_names is ignored for 2D activations.")

        result = np.empty((n_groups, n_timepoints), dtype=in_dtype)
        for gi, g in enumerate(groups):
            # float64 sum -> stable mean, then cast back
            result[gi] = (
                activations[group_to_indices[g], :].sum(axis=0, dtype=np.float64)
                / len(group_to_indices[g])
            ).astype(in_dtype, copy=False)

        df = pd.DataFrame(result, columns=[f"time_{i}" for i in range(n_timepoints)])
        df.insert(0, "group", groups)
        return df

    # ---- 3D ---------------------------------------------------------------
    n_batches = activations.shape[0]
    if batch_names is None:
        batch_names = [f"batch_{i}" for i in range(n_batches)]
    batch_names = list(to_nparray(batch_names, unique=False))
    if n_batches != len(batch_names):
        raise ValueError(
            "Length of batch_names has to be the same as activations.shape[0]."
        )

    # Row layout: row b*n_groups + gi corresponds to (batch=b, group=gi),
    # written via the gi::n_groups stride. Matches the original column order
    # (batch_name varies slowest, group varies within each batch).
    result = np.empty((n_batches * n_groups, n_timepoints), dtype=in_dtype)
    for gi, g in enumerate(groups):
        inds = group_to_indices[g]
        # (n_batches, |g|, n_timepoints) -> sum over neurons -> (n_batches, n_timepoints)
        mean_val = activations[:, inds, :].sum(axis=1, dtype=np.float64) / len(inds)
        result[gi::n_groups] = mean_val.astype(in_dtype, copy=False)

    out_batch_names = [bn for bn in batch_names for _ in range(n_groups)]
    out_groups = groups * n_batches

    df = pd.DataFrame(result, columns=[f"time_{i}" for i in range(n_timepoints)])
    df.insert(0, "batch_name", out_batch_names)
    df.insert(1, "group", out_groups)
    return df


def get_activations_for_path(
    path: pd.DataFrame,
    activations: torch.Tensor | npt.NDArray,
    model_in: torch.Tensor | npt.NDArray | None = None,
    sensory_indices: arrayable | None = None,
    idx_to_group: dict | None = None,
    activation_start: int = 0,
    aggregate: str | None = None,
) -> pd.DataFrame:
    """
    Get the activations for the pre and post neurons in the path, based on
    the activations of the model and the input.

    Args:
        path (pd.DataFrame): A dataframe representing the paths in the network.
            Each row is a connection, with columns for 'pre' and 'post' neuron
            indices, and 'layer'.
        activations (torch.Tensor | numpy.ndarray): The activations of the model.
            Shape should be (num_neurons, num_layers). If the shape is (num_neurons, 1)
            or (num_neurons,), then the same activation will be used for all layers in
            the path.
        model_in (torch.Tensor | numpy.ndarray): The input to the model. Shape
            should be (num_neurons, something) -  only the first column
            (num_neurons, 0), is used, when there is 1 in 'layer' in `path`. It
            is otherwise not used. Note: if aggregate is not None, then the input will
            be ignored.
        sensory_indices (arrayable): The indices of sensory neurons.
        idx_to_group (dict, optional): A dictionary mapping indices from the
            model to the groups in path (e.g. cell type). Defaults to None.
        activation_start (int, optional): Which layer corresponds to the start
            layer of path. By default, activation_start = 0, the sensory-only
            layer in the model corresponds to path.pre[path.layer == 1]. If you
            want activations[:,1] (i.e. two timesteps forward) to correspond to
            path.pre[path.layer == 1], set activation_start = 2. If you want
            the last timepoint to correspond to the last layer in path, set
            activation_start = activations.shape[1] - path.layer.max().
        aggregate (str or None, optional): Whether to aggregate activations across time
            first before mapping to path. If e.g. 'mean', will take the mean across
            timepoints. If None (default), each layer in the path corresponds to one
            timepoint in the activations.

    Returns:
        pd.DataFrame:
            The activations for the pre and post neurons in the path.
    """
    pathdf = path.copy()

    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().detach().numpy()
    if isinstance(model_in, torch.Tensor):
        model_in = model_in.cpu().detach().numpy()

    # single column = already aggregated; otherwise aggregate across time if requested
    if activations.ndim == 1:
        activations = activations[:, None]
    if activations.ndim > 2:
        raise ValueError(
            f"activations must be 1D or 2D, but got shape {activations.shape}."
        )
    aggregated = aggregate is not None or activations.shape[1] == 1
    if aggregate is not None and activations.shape[1] > 1:
        fn = aggregate if callable(aggregate) else getattr(np, aggregate)
        activations = np.asarray(fn(activations, axis=1)).reshape(-1, 1)

    if idx_to_group is not None:
        # turn value into string
        idx_to_group = {k: str(v) for k, v in idx_to_group.items()}
    else:
        idx_to_group = {idx: idx for idx in range(activations.shape[0])}

    # if starting later, just bump up the layer numbers
    pathdf.loc[:, ["layer"]] += activation_start

    out_df = []
    for l in sorted(pathdf.layer.unique()):  # layer number starts at 1
        layer_path = pathdf[pathdf.layer == l]

        # pre activations
        prenodes = set(layer_path.pre)
        pre_indices = [idx for idx, group in idx_to_group.items() if group in prenodes]
        if l == 1 and not aggregated:
            # raise error if sensory_indices or model_in doesn't exist
            if sensory_indices is None or model_in is None:
                raise ValueError(
                    "sensory_indices and model_in must be provided when layer == 1."
                )
            # need to get local indices for sensory_indices
            global2local = {idx: i for i, idx in enumerate(sensory_indices)}
            local_indices = [global2local[idx] for idx in pre_indices]
            pre_activations = model_in[local_indices, 0]
        else:
            pre_activations = activations[pre_indices, 0 if aggregated else l - 2]
        pre_group_act = pd.DataFrame(
            {"idx": pre_indices, "activation": pre_activations}
        )
        pre_group_act.loc[:, ["group"]] = [idx_to_group[idx] for idx in pre_indices]
        # mean per group
        pre_group_act = pre_group_act.groupby("group").activation.mean()

        # post activations
        postnodes = set(layer_path.post)
        post_indices = [
            idx for idx, group in idx_to_group.items() if group in postnodes
        ]
        post_activations = activations[post_indices, 0 if aggregated else l - 1]
        post_group_act = pd.DataFrame(
            {"idx": post_indices, "activation": post_activations}
        )
        post_group_act.loc[:, ["group"]] = [idx_to_group[idx] for idx in post_indices]
        # mean per group
        post_group_act = post_group_act.groupby("group").activation.mean()

        layer_path.loc[:, ["pre_activation"]] = layer_path.pre.map(pre_group_act)
        layer_path.loc[:, ["post_activation"]] = layer_path.post.map(post_group_act)
        out_df.append(layer_path)

    out = pd.concat(out_df)
    # reset layer numbers
    out.loc[:, ["layer"]] -= activation_start
    return out


def get_activations_for_el(
    el: pd.DataFrame,
    activations: torch.Tensor | npt.NDArray,
    idx_to_group: dict | None = None,
    aggregate: str | None = None,
) -> pd.DataFrame:
    """
    Add node activations to an edgelist with non-discrete (continuous) layers,
    such as the output of `external_paths.layered_el()` (columns 'pre', 'post',
    'pre_layer', 'post_layer'). Written by Claude, adapted from
    `get_activations_for_path()`.

    Unlike `get_activations_for_path()`, layers here are continuous (in
    'pre_layer'/'post_layer'), so activations cannot be indexed by layer. A single
    (aggregated) activation per group is used: each group is given the mean
    activation across its member neurons. The resulting 'pre_activation' and
    'post_activation' columns are consumed directly by `plot_paths()` in flow mode.

    Args:
        el (pd.DataFrame): Edgelist with 'pre' and 'post' columns holding group
            labels (as produced by `layered_el()`).
        activations (torch.Tensor | numpy.ndarray): Aggregated activations of shape
            (num_neurons, num_cols) or (num_neurons,). A single column is used as-is;
            if there is more than one column, `aggregate` must be set to collapse
            across columns (e.g. across time).
        idx_to_group (dict, optional): Maps model indices (rows of `activations`) to
            the group labels used in `el.pre`/`el.post`. If None, indices are used
            directly as group labels. Defaults to None.
        aggregate (str or callable, optional): How to aggregate across columns of
            `activations` (e.g. 'mean'). Required when `activations` has >1 column;
            ignored otherwise. Defaults to None.

    Returns:
        pd.DataFrame:
            A copy of `el` with added 'pre_activation' and 'post_activation' columns.
            Groups absent from `activations`/`idx_to_group` get NaN.
    """
    eldf = el.copy()

    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().detach().numpy()
    activations = np.asarray(activations)
    if activations.ndim == 1:
        activations = activations[:, None]
    if activations.ndim > 2:
        raise ValueError(f"activations must be 1D or 2D, got shape {activations.shape}")
    if activations.shape[1] > 1:
        if aggregate is None:
            raise ValueError(
                "activations has >1 column; set `aggregate` (e.g. 'mean') to "
                "collapse across columns."
            )
        fn = aggregate if callable(aggregate) else getattr(np, aggregate)
        activations = np.asarray(fn(activations, axis=1)).reshape(-1, 1)
    act = activations[:, 0]

    if idx_to_group is None:
        idx_to_group = {i: i for i in range(act.shape[0])}

    # mean activation per group, indexed by the group labels found in el.pre/el.post
    idx = np.fromiter(idx_to_group.keys(), dtype=int)
    groups = np.array([idx_to_group[i] for i in idx], dtype=object)
    group_act = pd.Series(act[idx]).groupby(groups).mean()

    eldf["pre_activation"] = eldf.pre.map(group_act)
    eldf["post_activation"] = eldf.post.map(group_act)
    return eldf


def activated_path_for_ngl(path):
    """
    Convert a path DataFrame [with 'pre_activation' and 'post_activation'
    columns] to a format suitable for Neuroglancer visualization
    (`get_ngl_link(df_format='long')`). Neurons are coloured by their
    activation.

    Args:
        path (pd.DataFrame): A DataFrame containing the columns 'pre', 'post',
            'layer', 'pre_activation', and 'post_activation' (standard output
            from function `activations_to_df()` and
            `get_activations_for_path()`).

    Returns:
        pd.DataFrame:
            A DataFrame with columns 'neuron_id', 'layer', and 'activation', suitable
            for Neuroglancer visualization.
    """
    dfs = []
    for l in path.layer.unique():
        path_l = path[path.layer == l]
        if l == path.layer.min():
            df = pd.DataFrame(
                {
                    "neuron_id": path_l.pre,
                    "layer": l,
                    "activation": path_l.pre_activation,
                }
            )
            # drop duplicate rows
            df = df.drop_duplicates()
            dfs.append(df)

        df = pd.DataFrame(
            {
                "neuron_id": path_l.post,
                "layer": l + 1,
                "activation": path_l.post_activation,
            }
        )
        df = df.drop_duplicates()
        dfs.append(df)
    return pd.concat(dfs)


def get_input_activation(
    model_in: npt.NDArray | torch.Tensor,
    sensory_indices: arrayable,
    idx_to_group: dict,
    selected_indices: arrayable | None = None,
    activation_threshold: float = 0,
    batch_names: arrayable | None = None,
) -> pd.DataFrame:
    """
    Get selected input activation.

    Args:
        model_in (numpy.ndarray | torch.Tensor): The input to the model. Shape should be
            (num_neurons, num_timepoints) for 2D or (batch_size, num_neurons,
            num_timepoints) for 3D.
        sensory_indices (arrayable): All the indices of sensory neurons. Length should
            be equal to the first dimension of model_in (2D) or second dimension (3D).
        idx_to_group (dict): A dictionary mapping indices from the model to the groups
            of interest (e.g. cell type). max(idx_to_group.keys()) should be equal to
            number of units in the model.
        selected_indices (arrayable, optional): The indices of the neurons to select.
            Defaults to None. If None, all neurons are selected.
        activation_threshold (float, optional): A threshold value for the absolute value
            of activation. Neurons with activations below this threshold are not
            considered. Defaults to 0.
        batch_names (arrayable, optional): The names of the batches. Defaults to None.
            If model_in.ndim == 3, then this should be supplied. If not, batch names
            will be e.g. 'batch_0', 'batch_1', etc.

    Notes:

        - If there are two neurons in a cell type, one passes the activation threshold and the other doesn't, only the one that passes the threshold is selected.


    Returns:
        pd.DataFrame:
            The input activation for the model. For 2D input, columns are 'group' and
            'time_0', 'time_1', etc. For 3D input, columns are 'batch_name', 'group' and
            'time_0', 'time_1', etc.
    """
    sensory_indices = to_nparray(sensory_indices)

    if isinstance(model_in, torch.Tensor):
        model_in = model_in.cpu().detach().numpy()

    # check the shape of model_in, if not correct raise an error
    if len(model_in.shape) not in [2, 3]:
        raise ValueError(
            f"Expected input shape (num_neurons={len(sensory_indices)}, num_timepoints) "
            f"or (batch_size, num_neurons={len(sensory_indices)}, num_timepoints)"
        )

    global2local = {glo: lo for lo, glo in enumerate(sensory_indices)}
    local2global = {lo: glo for lo, glo in enumerate(sensory_indices)}

    # selection boolean
    if selected_indices is not None:
        selected_local = [global2local[idx] for idx in selected_indices]
        selection_bool = np.zeros(len(sensory_indices), dtype=bool)
        selection_bool[selected_local] = True
    else:
        selection_bool = np.ones(len(sensory_indices), dtype=bool)

    # Handle 2D case
    if model_in.ndim == 2:
        # message if batch_names is not None
        if batch_names is not None:
            print("batch_names is ignored for 2D model_in.")

        # activation boolean
        # if any >= threshold for each row
        activation_bool = np.any(np.abs(model_in) >= activation_threshold, axis=1)

        selection = selection_bool & activation_bool

        local_selected_indices = [
            i for i, s in zip(range(len(sensory_indices)), selection) if s
        ]
        global_selected_indices = [local2global[i] for i in local_selected_indices]
        groups = [idx_to_group[idx] for idx in global_selected_indices]

        # out dataframe
        out = pd.DataFrame(model_in[selection, :], index=groups)
        out.index.name = "group"
        # group by group/index, take average
        out = out.groupby("group").mean().reset_index()
        # change column names
        out.columns = ["group"] + [f"time_{i}" for i in range(model_in.shape[1])]
        return out

    # Handle 3D case
    if model_in.ndim == 3:
        if batch_names is None:
            # make batch names
            batch_names = [f"batch_{i}" for i in range(model_in.shape[0])]

        batch_names = list(to_nparray(batch_names, unique=False))

        # make sure length matches
        if model_in.shape[0] != len(batch_names):
            raise ValueError(
                "Length of batch_names has to be the same as model_in.shape[0]."
            )

        # Process each batch
        batch_dfs = []
        for batch_idx, batch_name in enumerate(batch_names):
            batch_data = model_in[batch_idx]

            # activation boolean for this batch
            # if any >= threshold for each row
            activation_bool = np.any(np.abs(batch_data) >= activation_threshold, axis=1)

            selection = selection_bool & activation_bool

            local_selected_indices = [
                i for i, s in zip(range(len(sensory_indices)), selection) if s
            ]
            global_selected_indices = [local2global[i] for i in local_selected_indices]
            groups = [idx_to_group[idx] for idx in global_selected_indices]

            # out dataframe for this batch
            if len(groups) > 0:
                batch_df = pd.DataFrame(
                    batch_data[selection, :],
                    index=groups,
                    columns=[f"time_{i}" for i in range(model_in.shape[2])],
                )
                batch_df.index.name = "group"
                # group by group/index, take average
                batch_df = batch_df.groupby("group").mean().reset_index()
                batch_df["batch_name"] = batch_name
                batch_dfs.append(batch_df)

        if len(batch_dfs) == 0:
            # No activations passed the threshold
            return pd.DataFrame(
                columns=["batch_name", "group"]
                + [f"time_{i}" for i in range(model_in.shape[2])]
            )

        out = pd.concat(batch_dfs, ignore_index=True)
        # change column names and reorder
        time_cols = [f"time_{i}" for i in range(model_in.shape[2])]
        out = out[["batch_name", "group"] + time_cols]

        return out


def add_sign(inprop: sparse.spmatrix, idx_to_sign: dict):
    """
    Add sign to the inprop matrix based on the idx_to_sign dictionary, *and transpose
    it* such that the pre is in the columns (required by torch).

    Args:
        inprop (scipy.sparse matrix): The matrix to which the sign will be added.
        idx_to_sign (dict): A dictionary mapping indices to their sign (-1 or 1).

    Returns:
        scipy.sparse matrix:
            The matrix with the sign added.
    """
    # add negative connections
    neg_indices = [idx for idx, val in idx_to_sign.items() if val == -1]
    inpropcoo = inprop.tocoo().copy()
    mask = np.isin(inpropcoo.row, neg_indices)

    data = inpropcoo.data.copy()
    data[mask] = -inpropcoo.data[mask].copy()
    # make a signed sparse matrix
    inprop_signed = coo_matrix(
        (data, (inpropcoo.row, inpropcoo.col)), shape=inpropcoo.shape
    ).transpose()
    return inprop_signed


def activity_by_column(
    activity,
    idx_to_group: dict,
    idx_to_column: dict,
    selected_group: arrayable,
    model_input: Optional[Union[np.ndarray, torch.Tensor]] = None,
    sensory_indices: Optional[arrayable] = None,
):
    """
    Given the input arguments, return a dataframe with columns 'normalised_column',
    'cell_group', 'time_point', 'activation', and 'time_step'. This gives the raw data
    for `plot_activity_by_column()`. Note, if both model_input and sensory_indices are
    provided, the first timepoint of model_input is also included (time_step = 0). Other
    timesteps are already included in `activity`.

    Args:
        activity (torch.Tensor | numpy.ndarray): The activity of the model. Shape should
            be (num_neurons, num_timepoints).
        idx_to_group (dict): A dictionary mapping indices from the model to the groups
            of interest (e.g. cell type). max(idx_to_group.keys()) should be equal to
            number of units in the model.
        idx_to_column (dict): A dictionary mapping indices from the model to the columns
            of interest (e.g. column in the central complex).
        selected_group (arrayable): The groups to select from the activity. This should
            be a list of groups that are present in `idx_to_group.values()`.
        model_input (numpy.ndarray | torch.Tensor, optional): The input to the model.
            Shape should be (num_neurons, num_timepoints). If provided, the first
            timepoint of model_input is also included in the output dataframe
            (time_step = 0). Defaults to None.
        sensory_indices (arrayable, optional): The indices of sensory neurons.
            If provided, it should be a list of indices that are present in
            `idx_to_group`. If provided, it must also be provided with `model_input`.
            Defaults to None.

    Returns:
        pd.DataFrame:
            A dataframe with columns 'normalised_column', 'cell_group', 'time_point',
            'activation', and 'time_step'. The first timepoint of `model_input` is also
            included if `model_input` is provided.
    """

    # if one of model_input or sensory_indices is provided, the other must also be provided
    if (model_input is None) != (sensory_indices is None):
        raise ValueError(
            "If one of model_input or sensory_indices is provided, the other must also be provided."
        )

    column_acts = []
    input_acts = []  # only the first timestep is used. The rest are in `activity`

    for group in selected_group:
        indices = [idx for idx, g in idx_to_group.items() if g == group]
        column_act = get_neuron_activation(
            activity, indices, idx_to_group=idx_to_column
        )

        # Normalize group
        column_act["group"] = column_act["group"] / column_act["group"].max()
        column_act["cell_group"] = group

        # Melt the time columns into long format
        time_columns = [col for col in column_act.columns if col.startswith("time_")]

        melted = pd.melt(
            column_act,
            id_vars=["group", "cell_group"],
            value_vars=time_columns,
            var_name="time_point",
            value_name="activation",
        )

        column_acts.append(melted)

        if sensory_indices is not None:
            if len(set(indices) & set(sensory_indices)) > 0:
                input_act = get_input_activation(
                    model_input,
                    sensory_indices,
                    idx_to_column,
                    selected_indices=set(indices) & set(sensory_indices),
                )[["time_0"]].reset_index()
                input_act["group"] = input_act["group"] / input_act["group"].max()
                input_act["cell_group"] = group
                input_act = input_act.melt(
                    id_vars=["group", "cell_group"],
                    var_name="time_point",
                    value_name="activation",
                )
                input_acts.append(input_act)

    # Concatenate all melted data
    column_acts = pd.concat(column_acts, ignore_index=True)
    column_acts.rename(columns={"group": "normalised_column"}, inplace=True)

    # Extract time step number and sort
    column_acts["time_step"] = (
        column_acts["time_point"].str.extract(r"time_(\d+)").astype(int)
    )
    column_acts.time_step = column_acts.time_step + 1

    if len(input_acts) > 0:
        input_acts = pd.concat(input_acts, ignore_index=True)
        input_acts.rename(columns={"group": "normalised_column"}, inplace=True)
        input_acts["time_step"] = 0
        column_acts = pd.concat([column_acts, input_acts], ignore_index=True)

    column_acts = column_acts.sort_values(
        ["time_step", "cell_group", "normalised_column"]
    )

    return column_acts


def plot_activity_by_column(
    activity,
    idx_to_group: dict,
    idx_to_column: dict,
    selected_group: arrayable,
    plot_type: str = "line",
    model_input: Optional[Union[np.ndarray, torch.Tensor]] = None,
    sensory_indices: Optional[arrayable] = None,
    figsize: tuple = (800, 600),
    global_min: Optional[float] = None,
    global_max: Optional[float] = None,
):
    """
    Take output from `activity_by_column()` and plot the activity per neuron group, per
    time step, per column. The x axis is the normalised column, normalised within each
    neuron group.

    Args:
        activity (torch.Tensor | numpy.ndarray): The activity of the model. Shape should
            be (num_neurons, num_timepoints).
        idx_to_group (dict): A dictionary mapping indices from the model to the groups
            of interest (e.g. cell type). max(idx_to_group.keys()) should be equal to
            number of units in the model.
        idx_to_column (dict): A dictionary mapping indices from the model to the columns
            of interest (e.g. column in the central complex).
        selected_group (arrayable): The groups to select from the activity. This should
            be a list of groups that are present in `idx_to_group.values()`.
        plot_type (str, optional): The type of plot to create. Can be either 'scatter'
            or 'line'. Defaults to 'line'.
        model_input (numpy.ndarray | torch.Tensor, optional): The input to the model.
            Shape should be (num_neurons, num_timepoints). If provided, the first
            timepoint of model_input is also included in the output dataframe
            (time_step = 0). Defaults to None.
        sensory_indices (arrayable, optional): The indices of sensory neurons.
            If provided, it should be a list of indices that are present in
            `idx_to_group`. If provided, it must also be provided with `model_input`.
            Defaults to None.
        figsize (tuple, optional): The size of the figure in pixels. Defaults to (800,
            600).
        global_min (float, optional): The minimum value for the y-axis. If None, the
            minimum value is set to the smaller of 0, and the minimum activation value
            across all groups and time steps. Defaults to None.
        global_max (float, optional): The maximum value for the y-axis. If None, the
            maximum value is set to the bigger of 1, and the maximum activation value
            across all groups and time steps. Defaults to None.

    Returns:
        plotly.graph_objects.Figure:
            A Plotly figure object.
    """

    column_acts = activity_by_column(
        activity,
        idx_to_group,
        idx_to_column,
        selected_group,
        model_input=model_input,
        sensory_indices=sensory_indices,
    )

    # Get unique values
    unique_groups = column_acts["cell_group"].unique()
    unique_times = sorted(column_acts["time_step"].unique())

    # Create color mapping
    colors = plotly.colors.qualitative.Plotly[: len(unique_groups)]
    color_map = {group: colors[i] for i, group in enumerate(unique_groups)}

    # Create figure
    fig = go.Figure()

    # Add traces for each time step and group combination
    for time_step in unique_times:
        time_data = column_acts[column_acts["time_step"] == time_step]

        for group in unique_groups:
            group_data = time_data[time_data["cell_group"] == group]

            if plot_type == "scatter":
                fig.add_trace(
                    go.Scatter(
                        x=group_data["normalised_column"],
                        y=group_data["activation"],
                        mode="markers",
                        name=group,
                        legendgroup=group,  # Group legend entries
                        showlegend=True,
                        visible=bool(
                            time_step == unique_times[0]
                        ),  # Convert to Python bool
                        marker=dict(color=color_map[group], size=8),
                    )
                )
            else:  # line
                fig.add_trace(
                    go.Scatter(
                        x=group_data["normalised_column"],
                        y=group_data["activation"],
                        mode="lines+markers",
                        name=group,
                        legendgroup=group,
                        showlegend=True,
                        visible=bool(
                            time_step == unique_times[0]
                        ),  # Convert to Python bool
                        line=dict(color=color_map[group]),
                        marker=dict(size=6),
                    )
                )

    # Create slider steps
    steps = []
    for i, time_step in enumerate(unique_times):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": f"Neural Activity - Time Step: {time_step}"},
            ],
            label=str(time_step),
        )

        # Make traces visible for current time step
        start_idx = i * len(unique_groups)
        end_idx = start_idx + len(unique_groups)
        for j in range(start_idx, end_idx):
            step["args"][0]["visible"][j] = True

        steps.append(step)

    # Add slider
    sliders = [
        dict(
            active=0, currentvalue={"prefix": "Time Step: "}, pad={"t": 50}, steps=steps
        )
    ]

    fig.update_layout(
        sliders=sliders,
        title=f"Neural Activity - Time Step: {unique_times[0]}",
        xaxis_title="Normalised Column",
        yaxis_title="Activation Value",
        width=figsize[0],
        height=figsize[1],
    )
    if global_min is None:
        global_min = min(0, column_acts["activation"].min())
    if global_max is None:
        global_max = max(1, column_acts["activation"].max())
    fig.update_yaxes(range=[global_min, global_max])

    return fig


def plot_timeseries(
    df: pd.DataFrame,
    style: Optional[dict] = None,
    sizing: Optional[dict] = None,
    x_label: str = "Time",
    y_label: str = "Activation",
    title: Optional[str] = None,
    slider_dim: Optional[str] = "batch",
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    scatter_mode: str = "lines",
) -> go.Figure:
    """
    Generate an interactive time-series plot of neural activations, based on a dataframe
    that's like the output of get_neuron_activation().

    If both **'batch_name'** and **'group'** columns exist, the **slider_dim**
    argument decides which dimension the slider animates over.

    Args:
        df (pd.DataFrame): Must contain **'group'** plus one or more **'time_*'**
            columns (e.g. 'time_0', 'time_1' …). A **'batch_name'** column is
            optional.
        style (Optional[dict]): Plot styling; keys:

            - 'font_type': str, default='Arial'
            - 'linecolor': str, default='black'
            - 'papercolor': str, default='rgba(255,255,255,255)'

        sizing (Optional[dict]): Layout sizing; keys (px / pt):

            - 'fig_width': int, default=600
            - 'fig_height': int, default=400
            - 'fig_margin': int, default=0
            - 'fsize_ticks_pt': int, default=12
            - 'fsize_title_pt': int, default=16
            - 'markersize': int, default=8
            - 'ticklen': int, default=5
            - 'tickwidth': int, default=1
            - 'axislinewidth': int, default=1.5
            - 'markerlinewidth': int, default=1

        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        title (Optional[str]): Figure title.
        slider_dim (str | None): Dimension for slider if both are present.

            - 'batch' (default): slider toggles **batch_name**, traces coloured by group.
            - 'group': slider toggles **group**, traces coloured by batch_name.
            - None: no slider (all traces in one panel).

        ymin (Optional[float]): Minimum y-axis value. If None, set to minimum activation.
        ymax (Optional[float]): Maximum y-axis value. If None, set to maximum activation.
        scatter_mode (str): Plot mode for traces, e.g. 'lines', 'markers', 'lines+markers'.

    Returns:
        fig (go.Figure)
    """
    # ---------- defaults ----------------------------------------------------
    style_d = {
        "font_type": "Arial",
        "linecolor": "black",
        "papercolor": "rgba(255,255,255,255)",
    }
    sizing_d = {
        "fig_width": 600,
        "fig_height": 400,
        "fig_margin": 0,
        "fsize_ticks_pt": 12,
        "fsize_title_pt": 16,
        "markersize": 8,
        "ticklen": 5,
        "tickwidth": 1,
        "axislinewidth": 1.5,
        "markerlinewidth": 1,
    }
    if style:
        style_d.update(style)
    if sizing:
        sizing_d.update(sizing)
    s, z = style_d, sizing_d

    # ---------- sanity checks ----------------------------------------------
    time_cols = sorted(
        [c for c in df.columns if c.startswith("time_")],
        key=lambda c: int(c.split("_")[1]),
    )
    if not time_cols:
        raise ValueError("No 'time_*' columns found.")
    if "group" not in df.columns:
        raise ValueError("'group' column required.")

    has_batch = "batch_name" in df.columns
    if not has_batch:
        slider_dim = None  # can't animate over batch without data
    elif slider_dim not in {"batch", "group", None}:
        raise ValueError("slider_dim must be 'batch', 'group', or None.")

    x_vals = [int(c.split("_")[1]) for c in time_cols]

    y_data = df[time_cols].values.flatten()
    y_range = [
        ymin if ymin is not None else y_data.min(),
        ymax if ymax is not None else y_data.max(),
    ]

    # ---------- figure ------------------------------------------------------
    fig = go.Figure(
        layout=dict(
            width=z["fig_width"],
            height=z["fig_height"] + (100 if slider_dim else 0),
            margin=dict(l=50, r=20, t=(60 if title else 20), b=50 + z["fig_margin"]),
            paper_bgcolor=s["papercolor"],
            plot_bgcolor=s["papercolor"],
            font=dict(family=s["font_type"], size=z["fsize_ticks_pt"]),
            xaxis=dict(
                title=x_label,
                linecolor=s["linecolor"],
                linewidth=z["axislinewidth"],
                ticklen=z["ticklen"],
                tickwidth=z["tickwidth"],
            ),
            yaxis=dict(
                title=y_label,
                linecolor=s["linecolor"],
                linewidth=z["axislinewidth"],
                ticklen=z["ticklen"],
                tickwidth=z["tickwidth"],
                range=y_range,
            ),
            title=(
                dict(text=title, x=0.5, font=dict(size=z["fsize_title_pt"]))
                if title
                else None
            ),
            showlegend=True,
            legend=dict(borderwidth=0),
            sliders=[],
        )
    )

    # ---------- helper ------------------------------------------------------
    def make_trace(name, y, color_i):
        return go.Scatter(
            x=x_vals,
            y=y,
            mode=scatter_mode,
            name=str(name),
            legendgroup=str(name),
            line=dict(
                color=qualitative.Plotly[color_i % len(qualitative.Plotly)],
                width=z["axislinewidth"],
            ),
            marker=dict(
                size=z["markersize"],
                line=dict(width=z["markerlinewidth"], color=s["linecolor"]),
            ),
        )

    # ---------- no slider /  single dim ------------------------------------
    if slider_dim is None:
        # colour by group
        for gi, (grp, gdf) in enumerate(df.groupby("group")):
            fig.add_trace(make_trace(grp, gdf[time_cols].iloc[0], gi))
        return fig

    # ---------- slider: batch OR group -------------------------------------
    outer_key = "batch_name" if slider_dim == "batch" else "group"
    inner_key = "group" if slider_dim == "batch" else "batch_name"

    frames, steps = [], []
    for oi, (o_val, outer_df) in enumerate(df.groupby(outer_key)):
        traces = []
        for ii, (i_val, inner_df) in enumerate(outer_df.groupby(inner_key)):
            traces.append(make_trace(i_val, inner_df[time_cols].iloc[0], ii))
        frames.append(go.Frame(data=traces, name=str(oi)))
        steps.append(
            dict(
                label=o_val,
                method="animate",
                args=[
                    [str(oi)],
                    {"mode": "immediate", "frame": {"duration": 0, "redraw": True}},
                ],
            )
        )
        if oi == 0:
            fig.add_traces(traces)

    fig.frames = frames
    fig.update_layout(
        sliders=[
            dict(
                active=0,
                steps=steps,
                x=0.05,
                len=0.9,
                y=-0.08,
            )
        ]
    )
    return fig


def guess_optimal_stimulus(
    inprop: sparse.spmatrix,
    sensory_indices: arrayable,
    idx_to_sign: dict,
    target_activations: TargetActivation,
    longest_plen: int = 5,
):
    """
    Guesses optimal stimulus based on signed effective connectivity (no non-linearity).
    Since it calculates effective connectivity on the fly (behind the scene using
    `signed_effective_conn_from_paths()`), it would be good to have a small-ish path
    length (e.g. 5).

    When the layer number in target_activations is > longest_plen, the function returns
    a stimulus of shape ((batch,) num_sensory_neurons, longest_plen), instead of shape
    ((batch,) num_sensory_neurons, layer), appropriate for the stimulus immediately
    before the large layer number in target_activations.

    e.g. if target_activations has layer 10, and longest_plen = 5, the user should first
    make a random stimulus of shape ((batch,) num_sensory_neurons, 10), and then replace
    the last 5 timesteps with the output of this function.

    Args:
        inprop (sparse.spmatrix): The inprop matrix of the model. Pre is in the rows,
            post in the columns. There are only positive values in this matrix, the sign
            information is added using idx_to_sign.
        sensory_indices (arrayable): The sensory indices of the model.
        idx_to_sign (dict): A dictionary mapping indices to their sign (-1 or 1).
        target_activations (TargetActivation): The target activations.
        longest_plen (int, optional): The longest path length to consider. Defaults to 5.

    Returns:
        np.ndarray:
            The guessed optimal stimulus. Shape is ((batch,) num_sensory_neurons,
            min(longest_plen, max_layer_in_target_activations)).
    """

    all_layers = set()
    for batch in range(target_activations.batch_size):
        batch_targets = target_activations.get_batch_targets(batch)
        all_layers = all_layers | set(batch_targets.keys())

    max_layer = int(max(all_layers))
    stimuli = []

    nlayer = int(min(longest_plen, max_layer + 1))

    for batch in range(target_activations.batch_size):
        # shape (num_sensory_neurons, max_layer + 1) (0-indexing)
        this_stimulus = pd.DataFrame(
            data=np.zeros((len(sensory_indices), nlayer)),
            index=sensory_indices,
            columns=range(nlayer),
        )
        this_stimulus.index = this_stimulus.index.astype(str)

        batch_targets = target_activations.get_batch_targets(batch)
        # iterate through each layer for this batch
        for layer, neuron_targets in batch_targets.items():
            post_indices = list(neuron_targets.keys())
            # Convert neuron indices to int then string (pandas iterrows converts to float)
            neuron_targets = {str(int(k)): v for k, v in neuron_targets.items()}
            # excitatory/inhibitory effective connectivity
            excs, inhbs = signed_conn_by_path_length_data(
                inprop,
                sensory_indices,
                post_indices,
                int(min(layer + 1, longest_plen)),
                idx_to_sign,
            )
            # scale by target activations, then sum across post neurons
            excs = [
                exc.multiply(exc.columns.map(neuron_targets).values, axis=1).sum(axis=1)
                for exc in excs
            ]
            inhbs = [
                inhb.multiply(inhb.columns.map(neuron_targets).values, axis=1).sum(
                    axis=1
                )
                for inhb in inhbs
            ]

            # now loop through each path length and add to this_stimulus
            # e.g. len(excs) = 3 means path lengths 1, 2, 3
            for plen in range(len(excs)):
                # e.g. this_stimulus[2] += excs[0] - inhbs[0]
                this_stimulus[len(excs) - (1 + plen)] += excs[plen].reindex(
                    this_stimulus.index, fill_value=0
                )
                this_stimulus[len(excs) - (1 + plen)] -= inhbs[plen].reindex(
                    this_stimulus.index, fill_value=0
                )
        stimuli.append(this_stimulus.values.astype(np.float32))
    return np.stack(stimuli)


def legacy_activation_function(
    self, x: torch.Tensor, x_previous: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply the activation function to the input tensor. Currently it is
    tanh(relu(slope * x + bias)). x = weights @ inputs (done outside the activation
    function). Note if using this, then tau should be a number between 0 and 1, as it
    represents the extent to which the activation persists to the next timestep.

    Args:
        x (torch.Tensor): The input tensor to the activation function.
        x_previous (Optional[torch.Tensor]): The previous activations of the neurons.

    Returns:
        torch.Tensor:
            The output tensor after applying the activation function.
    """
    if self.custom_activation_function is not None:
        return self.custom_activation_function(self, x, x_previous)

    if self.slope_is_pairwise:
        slopes = 1.0  # pair-mode gain folded into effective_weights
    elif self.slope is None:
        slopes = self.tanh_steepness
    else:
        slopes = self.effective_slope[self.indices]  # shape: (num_neurons,)
        # shape: (num_neurons, batch_size)
        slopes = slopes.view(-1, 1).expand(-1, x.shape[1]).to(x.device)

    slopes = self.divnorm(slopes, x_previous)

    if self.biases is None:
        biases = self.default_bias
    else:
        biases = self.biases[self.indices]
        biases = biases.view(-1, 1).expand(-1, x.shape[1]).to(x.device)

    x = self.tau * x_previous + slopes * x + biases
    # x = slopes * x + biases

    # Thresholded relu
    # x = torch.where(x >= self.threshold, x, torch.zeros_like(x))
    # this propagates NaN honesty
    x = torch.relu(x - self.threshold) + self.threshold * (x >= self.threshold).to(
        x.dtype
    )

    # Tanh activation
    x = torch.tanh(x)

    return x


def legacy_activation_function_linear(
    self, x: torch.Tensor, x_previous: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply the activation function to the input tensor. Currently it is
    tanh(relu(slope * x + bias)). x = weights @ inputs (done outside the activation
    function). Note if using this, then tau should be a number between 0 and 1, as it
    represents the extent to which the activation persists to the next timestep.

    Args:
        x (torch.Tensor): The input tensor to the activation function.
        x_previous (Optional[torch.Tensor]): The previous activations of the neurons.

    Returns:
        torch.Tensor:
            The output tensor after applying the activation function.
    """
    if self.custom_activation_function is not None:
        return self.custom_activation_function(self, x, x_previous)

    if self.slope_is_pairwise:
        slopes = 1.0  # pair-mode gain folded into effective_weights
    elif self.slope is None:
        slopes = self.tanh_steepness
    else:
        slopes = self.effective_slope[self.indices]  # shape: (num_neurons,)
        # shape: (num_neurons, batch_size)
        slopes = slopes.view(-1, 1).expand(-1, x.shape[1]).to(x.device)

    slopes = self.divnorm(slopes, x_previous)

    if self.biases is None:
        biases = self.default_bias
    else:
        biases = self.biases[self.indices]
        biases = biases.view(-1, 1).expand(-1, x.shape[1]).to(x.device)

    x = self.tau * x_previous + slopes * x + biases
    # x = slopes * x + biases

    # # Thresholded relu
    # x = torch.where(x >= self.threshold, x, torch.zeros_like(x))

    # # Tanh activation
    # x = torch.tanh(x)

    return x
