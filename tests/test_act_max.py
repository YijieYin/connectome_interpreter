import unittest

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
import torch

from connectome_interpreter.activation_maximisation import (
    MultilayeredNetwork,
    TargetActivation,
    activation_maximisation,
    activations_to_df,
    activations_to_df_batched,
    get_neuron_activation,
    get_gradients,
)


class TestMultilayeredNetwork(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_neurons = 10
        self.num_sensory = 4
        self.num_layers = 3
        self.batch_size = 2

        # Create a dense matrix and convert it to a scipy sparse matrix
        dense_weights = np.random.rand(self.num_neurons, self.num_neurons)
        dense_weights = dense_weights / dense_weights.sum(axis=1, keepdims=True)
        dense_weights[:, :3] = -dense_weights[:, :3]
        self.all_weights = csr_matrix(dense_weights)  # Convert to scipy sparse matrix
        self.sensory_indices = list(range(self.num_sensory))

        self.model = MultilayeredNetwork(
            self.all_weights, self.sensory_indices, num_layers=self.num_layers
        ).to(self.device)

    def test_initialization(self):
        # self.assertEqual(self.model.num_layers, self.num_layers)
        # self.assertEqual(len(self.model.sensory_indices), self.num_sensory)
        # self.assertTrue(
        #     torch.equal(
        #         self.model.all_weights.to_dense(),
        #         torch.tensor(self.all_weights.toarray(), device=self.device),
        #     )
        # )
        self.assertEqual(self.model.num_layers, self.num_layers)
        self.assertEqual(len(self.model.sensory_indices), self.num_sensory)

        # Convert both to numpy arrays for comparison
        model_weights = self.model.all_weights.to_dense().cpu().numpy()
        expected_weights = self.all_weights.toarray()

        # Use numpy's allclose for a more tolerant comparison
        self.assertTrue(
            np.allclose(model_weights, expected_weights, rtol=1e-5, atol=1e-5),
            "Weights matrices are not equal within tolerance",
        )

    def test_forward_pass_2d(self):
        print("testing forward pass 2d")
        input_tensor = torch.rand(self.num_sensory, self.num_layers).to(self.device)
        output = self.model(input_tensor)

        expected_shape = (self.num_neurons, self.num_layers)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.all(output >= -1) and torch.all(output <= 1))

    def test_forward_pass_3d(self):
        print("testing forward pass 3d")
        input_tensor = torch.rand(
            self.batch_size, self.num_sensory, self.num_layers
        ).to(self.device)
        output = self.model(input_tensor)

        expected_shape = (self.batch_size, self.num_neurons, self.num_layers)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.all(output >= -1) and torch.all(output <= 1))


class TestTargetActivation(unittest.TestCase):
    def setUp(self):
        self.dict_targets = {0: {1: 0.5, 2: 0.8}, 1: {0: 0.3}}
        self.df_targets = pd.DataFrame(
            [
                {"batch": 0, "layer": 0, "neuron": 1, "value": 0.5},
                {"batch": 0, "layer": 0, "neuron": 2, "value": 0.8},
                {"batch": 1, "layer": 1, "neuron": 0, "value": 0.3},
            ]
        )

    def test_dict_initialization(self):
        target = TargetActivation(targets=self.dict_targets, batch_size=2)
        self.assertEqual(target.batch_size, 2)

        batch_targets = target.get_batch_targets(0)
        self.assertEqual(batch_targets[0][1], 0.5)

    def test_df_initialization(self):
        target = TargetActivation(targets=self.df_targets)
        self.assertEqual(target.batch_size, 2)

        batch_targets = target.get_batch_targets(1)
        self.assertEqual(batch_targets[1][0], 0.3)


class TestActivationMaximisation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a dense matrix and convert it to a scipy sparse matrix
        dense_weights = np.random.rand(10, 10)
        dense_weights = dense_weights / dense_weights.sum(axis=1, keepdims=True)
        dense_weights[:, :3] = -dense_weights[:, :3]
        self.all_weights = csr_matrix(dense_weights)  # Convert to scipy sparse matrix

        self.model = MultilayeredNetwork(
            self.all_weights,
            sensory_indices=[0, 1, 2, 3],
            num_layers=3,
        ).to(self.device)

        self.targets = TargetActivation(
            {0: {0: 0.5, 1: 0.8}, 1: {2: 0.3}}, batch_size=2
        )

    def test_basic_optimization(self):
        print("testing basic optimization")

        result = activation_maximisation(
            self.model,
            self.targets,
            num_iterations=10,
            in_reg_lambda=1e-3,
            out_reg_lambda=1e-3,
            wandb=False,
            device=self.device,
        )

        input_tensor, output, act_losses, *_ = result
        expected_shape = (2, 4, 3)  # (batch_size, num_sensory, num_layers)
        self.assertEqual(input_tensor.shape, expected_shape)
        self.assertTrue(act_losses[-1] <= act_losses[0])

    def test_custom_regularization(self):
        print("testing custom regularization")

        custom_reg = {
            "in": lambda x: torch.sum(torch.abs(x)),
            "out": lambda x: torch.sum(torch.abs(x)),
        }

        result = activation_maximisation(
            self.model,
            self.targets,
            custom_reg_functions=custom_reg,
            num_iterations=10,
            wandb=False,
            device=self.device,
        )

        _, _, _, out_reg_losses, in_reg_losses, _ = result
        self.assertTrue(len(in_reg_losses) > 0 and len(out_reg_losses) > 0)


class TestActivationsToDF(unittest.TestCase):
    def setUp(self):
        self.weights = np.array([[0.5, 0.3, 0.0], [0.0, 0.4, 0.2], [0.0, 0.0, 0.6]])
        self.input_act = np.array([[0.8, 0.6], [0.7, 0.5]])
        self.output_act = np.array([[0.8, 0.6], [0.7, 0.5], [0.6, 0.4]])
        self.sensory_indices = [0, 1]

    def test_basic_functionality(self):
        paths = activations_to_df(
            self.weights, self.input_act, self.output_act, self.sensory_indices
        )

        expected_columns = [
            "pre",
            "post",
            "weight",
            "layer",
            "pre_activation",
            "post_activation",
        ]
        self.assertTrue(all(col in paths.columns for col in expected_columns))
        self.assertEqual(paths["layer"].nunique(), self.output_act.shape[1])

    def test_sparse_input(self):
        sparse_weights = csr_matrix(self.weights)

        paths_dense = activations_to_df(
            self.weights, self.input_act, self.output_act, self.sensory_indices
        )
        paths_sparse = activations_to_df(
            sparse_weights,
            self.input_act,
            self.output_act,
            self.sensory_indices,
        )

        pd.testing.assert_frame_equal(paths_dense, paths_sparse)


class TestActivationsToDFBatched(unittest.TestCase):
    def setUp(self):
        self.weights = np.array([[0.5, 0.3, 0.0], [0.0, 0.4, 0.2], [0.0, 0.0, 0.6]])
        self.batched_input = np.array(
            [[[0.8, 0.6], [0.7, 0.5]], [[0.6, 0.4], [0.5, 0.3]]]
        )
        self.batched_output = np.array(
            [
                [[0.8, 0.6], [0.7, 0.5], [0.6, 0.4]],
                [[0.6, 0.4], [0.5, 0.3], [0.4, 0.2]],
            ]
        )
        self.sensory_indices = [0, 1]

    def test_batched_processing(self):
        print("testing batched proccessing")
        paths = activations_to_df_batched(
            self.weights,
            self.batched_input,
            self.batched_output,
            self.sensory_indices,
        )

        self.assertTrue("batch" in paths.columns)
        self.assertEqual(paths["batch"].nunique(), self.batched_input.shape[0])


class TestGetNeuronActivation(unittest.TestCase):

    def test_2d_output_with_groups(self):
        output = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        neuron_indices = [0, 2]
        idx_to_group = {0: "A", 2: "B"}

        df = get_neuron_activation(output, neuron_indices, idx_to_group=idx_to_group)

        expected_df = pd.DataFrame(
            {
                "group": ["A", "B"],
                "time_0": [0.1, 0.7],
                "time_1": [0.2, 0.8],
                "time_2": [0.3, 0.9],
            }
        )
        pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)

    def test_2d_output_without_groups(self):
        output = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        neuron_indices = [1]

        df = get_neuron_activation(output, neuron_indices)

        expected_df = pd.DataFrame(
            {"group": [1], "time_0": [0.4], "time_1": [0.5], "time_2": [0.6]}
        )
        pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)

    def test_3d_output_with_batch_names_and_groups(self):
        output = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        neuron_indices = [0]
        batch_names = ["batch_1", "batch_2"]
        idx_to_group = {0: "A"}

        df = get_neuron_activation(
            output,
            neuron_indices,
            batch_names=batch_names,
            idx_to_group=idx_to_group,
        )

        expected_df = pd.DataFrame(
            {
                "batch_name": ["batch_1", "batch_2"],
                "group": ["A", "A"],
                "time_0": [0.1, 0.5],
                "time_1": [0.2, 0.6],
            }
        )
        pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)

    def test_3d_output_without_batch_names(self):
        output = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        neuron_indices = [1]

        df = get_neuron_activation(output, neuron_indices)

        expected_df = pd.DataFrame(
            {
                "batch_name": ["batch_0", "batch_1"],
                "group": [1, 1],
                "time_0": [0.3, 0.7],
                "time_1": [0.4, 0.8],
            }
        )
        pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)

    def test_batch_names_mismatch(self):
        output = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        neuron_indices = [1]
        batch_names = ["batch_1"]

        with self.assertRaises(ValueError):
            get_neuron_activation(output, neuron_indices, batch_names=batch_names)


# Add these test classes to your existing test file


class TestMultilayeredNetworkEnhanced(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_neurons = 10
        self.num_sensory = 4
        self.num_layers = 3

        # Create weights and mappings
        dense_weights = np.random.rand(self.num_neurons, self.num_neurons)
        dense_weights = dense_weights / dense_weights.sum(axis=1, keepdims=True)
        self.all_weights = csr_matrix(dense_weights)
        self.sensory_indices = list(range(self.num_sensory))

        # Create idx_to_group mapping
        self.idx_to_group = {i: f"type_{i//3}" for i in range(self.num_neurons)}

    def test_trainable_parameters_initialization(self):
        """Test initialization with trainable parameters"""
        model = MultilayeredNetwork(
            self.all_weights,
            self.sensory_indices,
            idx_to_group=self.idx_to_group,
            default_bias=0.2,
            tanh_steepness=3.0,
        ).to(self.device)

        # Check that parameters exist
        self.assertIsNotNone(model.slope)
        self.assertIsNotNone(model.biases)
        self.assertIsNotNone(model.indices)

        # Check parameter shapes
        num_types = len(set(self.idx_to_group.values()))
        self.assertEqual(model.slope.shape[0], num_types)
        self.assertEqual(model.biases.shape[0], num_types)

    def test_dict_parameter_values(self):
        """Test initialization with dictionary parameter values"""
        bias_dict = {"type_0": 0.1, "type_1": 0.2, "type_2": 0.3}
        slope_dict = {"type_0": 2.0, "type_1": 4.0, "type_2": 6.0}

        model = MultilayeredNetwork(
            self.all_weights,
            self.sensory_indices,
            idx_to_group=self.idx_to_group,
            bias_dict=bias_dict,
            slope_dict=slope_dict,
        ).to(self.device)

        # Check that parameters were set correctly
        # The order depends on set() ordering, so check individual values
        unique_types = sorted(set(self.idx_to_group.values()))
        for i, type_name in enumerate(unique_types):
            if type_name in bias_dict:
                expected_bias = bias_dict[type_name]
            else:
                expected_bias = 0  # default
            self.assertAlmostEqual(model.raw_biases[i].item(), expected_bias, places=6)

    def test_custom_activation_function(self):
        """Test custom activation function"""

        def custom_activation(x):
            return torch.sigmoid(x)

        model = MultilayeredNetwork(
            self.all_weights,
            self.sensory_indices,
            activation_function=custom_activation,
        ).to(self.device)

        input_tensor = torch.rand(self.num_sensory, self.num_layers).to(self.device)
        output = model(input_tensor)

        # Check that output is in sigmoid range [0, 1]
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))

    def test_backward_compatibility(self):
        """Test that model works without new parameters (backward compatibility)"""
        model = MultilayeredNetwork(
            self.all_weights, self.sensory_indices, num_layers=self.num_layers
        ).to(self.device)

        # Check that trainable parameters are None
        self.assertIsNone(model.slope)
        self.assertIsNone(model.biases)
        self.assertIsNone(model.indices)

        # Check that forward pass still works
        input_tensor = torch.rand(self.num_sensory, self.num_layers).to(self.device)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.num_neurons, self.num_layers))

    def test_parameter_gradient_control(self):
        """Test set_param_grads method"""
        model = MultilayeredNetwork(
            self.all_weights, self.sensory_indices, idx_to_group=self.idx_to_group
        ).to(self.device)

        # Initially parameters shouldn't require grad
        self.assertFalse(model.slope.requires_grad)
        self.assertFalse(model.raw_biases.requires_grad)

        # Enable gradients
        model.set_param_grads(slopes=True, raw_biases=True)
        self.assertTrue(model.slope.requires_grad)
        self.assertTrue(model.raw_biases.requires_grad)

        # Disable gradients
        model.set_param_grads(slopes=False, raw_biases=False)
        self.assertFalse(model.slope.requires_grad)
        self.assertFalse(model.raw_biases.requires_grad)


class TestContextManagers(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dense_weights = np.random.rand(5, 5)
        self.all_weights = csr_matrix(dense_weights)
        self.sensory_indices = [0, 1]
        self.idx_to_group = {i: f"type_{i}" for i in range(5)}

    def test_training_mode_context(self):
        """Test training_mode context manager"""
        from connectome_interpreter.activation_maximisation import training_mode

        model = MultilayeredNetwork(
            self.all_weights, self.sensory_indices, idx_to_group=self.idx_to_group
        ).to(self.device)

        # Initially no gradients
        self.assertFalse(model.slope.requires_grad)
        self.assertFalse(model.raw_biases.requires_grad)

        # Inside context, gradients should be enabled
        with training_mode(model):
            self.assertTrue(model.slope.requires_grad)
            self.assertTrue(model.raw_biases.requires_grad)

        # After context, gradients should be disabled
        self.assertFalse(model.slope.requires_grad)
        self.assertFalse(model.raw_biases.requires_grad)


class TestTrainModelEnhanced(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dense_weights = np.random.rand(6, 6) * 0.1
        self.all_weights = csr_matrix(dense_weights)
        self.sensory_indices = [0, 1]
        self.idx_to_group = {i: f"type_{i//2}" for i in range(6)}

        self.model = MultilayeredNetwork(
            self.all_weights,
            self.sensory_indices,
            num_layers=2,
            idx_to_group=self.idx_to_group,
        ).to(self.device)

    def test_time_series_targets(self):
        """Test training with time series targets (layer column)"""
        from connectome_interpreter.activation_maximisation import train_model

        # Create inputs and time series targets
        inputs = torch.rand(4, 2, 2).to(self.device)
        targets = pd.DataFrame(
            [
                {"batch": 0, "neuron_idx": 2, "layer": 0, "value": 0.5},
                {"batch": 0, "neuron_idx": 2, "layer": 1, "value": 0.8},
                {"batch": 1, "neuron_idx": 3, "layer": 0, "value": 0.3},
                {"batch": 2, "neuron_idx": 4, "layer": 1, "value": 0.7},
                {"batch": 3, "neuron_idx": 5, "layer": 0, "value": 0.4},
            ]
        )

        # Train model
        result = train_model(self.model, inputs, targets, num_epochs=5, wandb=False)

        model, history, *_ = result

        # Check that training occurred
        self.assertTrue(len(history["loss"]) > 0)
        self.assertIsInstance(history["loss"][0], float)

    def test_backward_compatible_targets(self):
        """Test training with old format targets (no layer column)"""
        from connectome_interpreter.activation_maximisation import train_model

        inputs = torch.rand(4, 2, 2).to(self.device)
        targets = pd.DataFrame(
            [
                {"batch": 0, "neuron_idx": 2, "value": 0.5},
                {"batch": 1, "neuron_idx": 3, "value": 0.3},
                {"batch": 2, "neuron_idx": 4, "value": 0.7},
                {"batch": 3, "neuron_idx": 5, "value": 0.4},
            ]
        )

        # Should work without error
        result = train_model(self.model, inputs, targets, num_epochs=3, wandb=False)

        model, history, *_ = result
        self.assertTrue(len(history["loss"]) > 0)


class TestSaliencyEnhanced(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dense_weights = np.random.rand(8, 8) * 0.1
        self.all_weights = csr_matrix(dense_weights)
        self.sensory_indices = [0, 1, 2]
        self.idx_to_group = {i: f"type_{i//2}" for i in range(8)}

    def test_saliency_with_trainable_model(self):
        """Test saliency computation with trainable parameters"""
        from connectome_interpreter.activation_maximisation import saliency

        model = MultilayeredNetwork(
            self.all_weights,
            self.sensory_indices,
            num_layers=2,
            idx_to_group=self.idx_to_group,
        ).to(self.device)

        input_tensor = torch.rand(3, 2).to(self.device)
        neurons_of_interest = {0: [4, 5], 1: [6, 7]}

        saliency_maps = saliency(
            model, input_tensor, neurons_of_interest, device=self.device
        )

        # Check output shape and that gradients were computed
        self.assertEqual(saliency_maps.shape, input_tensor.shape)
        self.assertFalse(torch.allclose(saliency_maps, torch.zeros_like(saliency_maps)))

    def test_saliency_methods(self):
        """Test different saliency methods"""
        from connectome_interpreter.activation_maximisation import saliency

        model = MultilayeredNetwork(
            self.all_weights, self.sensory_indices, num_layers=2
        ).to(self.device)

        input_tensor = torch.rand(3, 2).to(self.device)
        neurons_of_interest = {0: [4], 1: [5]}

        # Test vanilla saliency
        vanilla_sal = saliency(
            model,
            input_tensor,
            neurons_of_interest,
            method="vanilla",
            device=self.device,
        )

        # Test input_x_gradient saliency
        ixg_sal = saliency(
            model,
            input_tensor,
            neurons_of_interest,
            method="input_x_gradient",
            device=self.device,
        )

        # Should produce different results
        self.assertFalse(torch.allclose(vanilla_sal, ixg_sal))

    def test_saliency_invalid_method(self):
        """Test saliency with invalid method"""
        from connectome_interpreter.activation_maximisation import saliency

        model = MultilayeredNetwork(
            self.all_weights, self.sensory_indices, num_layers=2
        ).to(self.device)

        input_tensor = torch.rand(3, 2).to(self.device)
        neurons_of_interest = {0: [4]}

        with self.assertRaises(ValueError):
            saliency(
                model,
                input_tensor,
                neurons_of_interest,
                method="invalid_method",
                device=self.device,
            )


class TestTargetActivationEnhanced(unittest.TestCase):
    def test_time_series_targets_dict(self):
        """Test TargetActivation with time series targets from dict"""
        targets_dict = {0: {1: 0.5, 2: 0.8}, 1: {0: 0.3, 3: 0.7}}  # layer 0  # layer 1
        target = TargetActivation(targets=targets_dict, batch_size=3)

        # Check that all batches have the same targets
        batch_0_targets = target.get_batch_targets(0)
        batch_2_targets = target.get_batch_targets(2)
        self.assertEqual(batch_0_targets, batch_2_targets)

        # Check layer structure
        self.assertIn(0, batch_0_targets)  # layer 0
        self.assertIn(1, batch_0_targets)  # layer 1


class TestDivisiveNormalization(unittest.TestCase):
    def setUp(self):
        # Simple 2x2 weight matrix for tests
        dense = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)
        self.weights = csr_matrix(dense)
        self.sensory_indices = [0]
        self.idx_to_group = {0: "A", 1: "B"}

    def test_missing_idx_to_group_raises(self):
        dn = {"A": ["B"]}
        with self.assertRaises(AssertionError):
            MultilayeredNetwork(
                self.weights, self.sensory_indices, divisive_normalization=dn
            )

    def test_positive_divnorm_weight_raises(self):
        dense = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)  # weight[1,0] positive
        weights = csr_matrix(dense)
        dn = {"A": ["B"]}
        with self.assertRaises(ValueError):
            MultilayeredNetwork(
                weights,
                self.sensory_indices,
                idx_to_group=self.idx_to_group,
                divisive_normalization=dn,
            )

    def test_weight_removal_and_storage(self):
        dense = np.array([[0.0, -0.5], [0.3, 0.0]], dtype=float)  # from B to A, -0.5
        weights = csr_matrix(dense)
        dn = {"B": ["A"]}
        model = MultilayeredNetwork(
            weights,
            self.sensory_indices,
            idx_to_group=self.idx_to_group,
            divisive_normalization=dn,
        )
        dense_out = model.all_weights.to_dense().cpu().numpy()
        # entry (post=0, pre=1) should be removed -> zero
        self.assertEqual(dense_out[0, 1], 0.0)
        # stored divnorm_weights should contain original -0.5
        self.assertTrue(
            torch.allclose(
                model.divnorm_weights,
                torch.tensor([-0.5], dtype=torch.float32).to(
                    model.divnorm_weights.device
                ),
                atol=1e-6,
            )
        )
        # stored indices should point to (0, 1) -> pre=1, post=0
        idx = model.divnorm_indices.cpu().numpy()
        self.assertTrue(np.array_equal(idx, np.array([[0], [1]])))

    def test_divisive_strength_param(self):
        dense = np.array([[0.0, -0.2], [0.0, 0.0]], dtype=float)
        weights = csr_matrix(dense)
        dn = {"B": ["A"]}  # one pres type
        # custom strength
        ds = {"B": 0.3}
        model = MultilayeredNetwork(
            weights,
            self.sensory_indices,
            idx_to_group=self.idx_to_group,
            divisive_normalization=dn,
            divisive_strength=ds,
        )
        # one strength value
        self.assertEqual(model.divisive_strength.numel(), 1)
        self.assertFalse(model.divisive_strength.requires_grad)
        self.assertAlmostEqual(model.divisive_strength.item(), 0.3, places=6)

    def test_divnorm_identity_when_none(self):
        dense = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
        weights = csr_matrix(dense)
        model = MultilayeredNetwork(weights, self.sensory_indices)
        slopes = torch.tensor([1.0, 2.0])
        x_prev = torch.tensor([0.5, 0.5])
        out = model.divnorm(slopes.clone(), x_prev)
        self.assertTrue(torch.allclose(out, slopes))

    def test_divnorm_effect_forward(self):
        # If pre-activation=1, weight=-1 and divisive_strength=1, new slope = 0 -> no activation
        dense = np.array(
            [
                [0.0, 0.0, 0.0],
                [-0.5, 0.0, 1.0],  # C excites B, A inhibits B
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        weights = csr_matrix(dense)
        dn = {"A": ["B"]}
        ds = {"A": 20}  # a very big divisive strength, should set slope to 0
        model = MultilayeredNetwork(
            weights,
            sensory_indices=[0, 2],
            tanh_steepness=5,
            idx_to_group={0: "A", 1: "B", 2: "C"},
            divisive_normalization=dn,
            divisive_strength=ds,
        )
        # one layer, one sensory neuron => shape (sensory, layers) = (2,2)
        inp = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        out = model(inp)
        # post neuron idx=1 should receive zero activation
        self.assertAlmostEqual(out[1, 0].item(), 0.0, places=6)

        # and on the other hand, without divisive normalization, it should > 0
        model_no_dn = MultilayeredNetwork(
            weights, sensory_indices=[0, 2], idx_to_group={0: "A", 1: "B", 2: "C"}
        )
        out_no_dn = model_no_dn(inp)
        self.assertGreater(out_no_dn[1, 0].item(), 0.0)


def _linear_activation(x):
    # custom activation: identity; ignores x_previous
    return x


class TestGetGradients(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # network: 5 neurons total; neurons 0,1 are sensory
        # paths: 0 -> 2 -> 4; 1 -> 3 (no path to 4)
        self.num_neurons = 5
        self.sensory_indices = [0, 1]
        self.num_layers = 3

        W = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        W[2, 0] = 0.7  # 0->2
        W[3, 1] = 0.6  # 1->3
        W[4, 2] = 0.5  # 2->4
        # No 3->4 edge so sensor 1 has no route to neuron 4

        self.all_weights = csr_matrix(W)

        self.model = MultilayeredNetwork(
            self.all_weights,
            sensory_indices=self.sensory_indices,
            num_layers=self.num_layers,
            threshold=0.0,  # keep inputs as provided
            activation_function=_linear_activation,  # linear system for deterministic grads
        ).to(self.device)

        # inputs: shape (batch, num_sensory, layers)
        self.batch_size = 2
        self.inputs_batched = torch.full(
            (self.batch_size, len(self.sensory_indices), self.num_layers),
            0.5,
            device=self.device,
            dtype=torch.float32,
        )
        # single example: shape (num_sensory, layers)
        self.inputs_single = torch.full(
            (len(self.sensory_indices), self.num_layers),
            0.5,
            device=self.device,
            dtype=torch.float32,
        )

    def test_returns_dataframe_shape_and_columns(self):
        monitor = [0, 1]  # indices
        target = {2: [4]}  # layer 2, neuron 4
        df = get_gradients(
            self.model,
            self.inputs_batched,
            monitor_neurons=monitor,
            target_neurons=target,
            monitor_layers=[0, 1, 2],
            idx_to_group=None,
            batch_names=[f"b{i}" for i in range(self.batch_size)],
            device=self.device,
        )
        # expected columns
        expected_time_cols = {f"time_{t}" for t in [0, 1, 2]}
        self.assertTrue({"group", "batch_name"}.issubset(df.columns))
        self.assertTrue(expected_time_cols.issubset(df.columns))
        # two monitored neurons * two batches = 4 rows
        self.assertEqual(len(df), 4)
        # group column are raw indices when idx_to_group=None
        self.assertTrue(set(df["group"].unique()).issubset({0, 1}))

    def test_gradient_path_specificity(self):
        # sensor 0 should influence target (0->2->4), sensor 1 should not
        monitor = [0, 1]
        target = {2: [4]}
        df = get_gradients(
            self.model,
            self.inputs_single,
            monitor_neurons=monitor,
            target_neurons=target,
            monitor_layers=[0, 1, 2],
            device=self.device,
        )
        # convert to wide for easy checks
        wide = df.pivot(index="group", columns="batch_name").sort_index()
        # single input -> only "batch_0" exists
        g0 = wide.loc[0].to_numpy()
        # print("g0: ", g0)
        g1 = wide.loc[1].to_numpy()

        # grads should be non-zero for sensor 0 and ~0 for sensor 1
        self.assertTrue(np.any(g0 != 0))
        self.assertTrue(np.allclose(g1, 0.0, atol=1e-8))

        # sign should be positive given positive weights
        self.assertTrue(np.all(g0 >= 0))

    def test_gradients_constant_across_batches_in_linear_case(self):
        monitor = [0, 1]
        target = {2: [4]}
        # two different batches; in linear system with identity activation the gradient is input-independent
        inp = self.inputs_batched.clone()
        inp[1] = 0.1  # change batch 2 input
        df = get_gradients(
            self.model,
            inp,
            monitor_neurons=monitor,
            target_neurons=target,
            monitor_layers=[0, 1, 2],
            batch_names=["A", "B"],
            device=self.device,
        )
        a = df[df["batch_name"] == "A"].sort_values("group")
        b = df[df["batch_name"] == "B"].sort_values("group")
        self.assertTrue(
            np.allclose(
                a.filter(like="time_").to_numpy(),
                b.filter(like="time_").to_numpy(),
                atol=1e-10,
            )
        )

    def test_idx_to_group_mapping_and_averaging(self):
        # group mapping: 0->S0, 1->S1, 2->X, 3->X, 4->T
        idx_to_group = {0: "S0", 1: "S1", 2: "X", 3: "X", 4: "T"}
        # monitor by group names
        monitor = ["S0", "S1"]
        # target is neuron 4 via group "T" at layer 2
        target = {2: ["T"]}

        df = get_gradients(
            self.model,
            self.inputs_batched,
            monitor_neurons=monitor,
            target_neurons=target,
            monitor_layers=[0, 1, 2],
            idx_to_group=idx_to_group,
            batch_names=["A", "B"],
            device=self.device,
        )

        # groups in output should be group names
        self.assertTrue(set(df["group"].unique()).issubset({"S0", "S1"}))

        # S0 should have non-zero grads; S1 should be ~0
        s0 = df[df["group"] == "S0"].filter(like="time_").to_numpy()
        s1 = df[df["group"] == "S1"].filter(like="time_").to_numpy()
        self.assertTrue(np.any(s0 != 0))
        self.assertTrue(np.allclose(s1, 0.0, atol=1e-8))

    def test_monitor_layers_subset(self):
        monitor = [0, 1]
        target = {2: [4]}
        df = get_gradients(
            self.model,
            self.inputs_single,
            monitor_neurons=monitor,
            target_neurons=target,
            monitor_layers=[1, 2],  # subset only
            device=self.device,
        )
        self.assertIn("time_1", df.columns)
        self.assertIn("time_2", df.columns)
        self.assertNotIn("time_0", df.columns)
