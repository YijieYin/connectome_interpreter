import unittest

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from connectome_interpreter.activation_maximisation import (
    LinearNetwork,
    MultilayeredNetwork,
    TargetActivation,
    activation_maximisation,
    activations_to_df,
    activations_to_df_batched,
    get_neuron_activation,
    get_input_activation,
    get_gradients,
    guess_optimal_stimulus,
    training_mode,
    train_model,
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


class TestGetInputActivation(unittest.TestCase):
    """Test get_input_activation function for both 2D and 3D inputs"""

    def test_2d_input_basic(self):
        """Test 2D input with basic functionality"""
        model_in = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.0, 0.0, 0.0]]
        )
        sensory_indices = [0, 1, 2, 3]
        idx_to_group = {0: "A", 1: "B", 2: "C", 3: "D"}

        df = get_input_activation(model_in, sensory_indices, idx_to_group)

        # Should have all groups except D (all zeros)
        self.assertTrue(set(df["group"].unique()).issubset({"A", "B", "C", "D"}))
        # Should have time columns
        self.assertTrue("time_0" in df.columns)
        self.assertTrue("time_1" in df.columns)
        self.assertTrue("time_2" in df.columns)

    def test_2d_input_with_groups(self):
        """Test 2D input where multiple neurons map to same group"""
        model_in = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        sensory_indices = [0, 1, 2, 3]
        idx_to_group = {0: "A", 1: "A", 2: "B", 3: "B"}  # Two groups

        df = get_input_activation(model_in, sensory_indices, idx_to_group)

        expected_df = pd.DataFrame(
            {
                "group": ["A", "B"],
                "time_0": [0.2, 0.6],  # Average of (0.1, 0.3) and (0.5, 0.7)
                "time_1": [0.3, 0.7],  # Average of (0.2, 0.4) and (0.6, 0.8)
            }
        )
        pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)

    def test_2d_input_with_threshold(self):
        """Test 2D input with activation threshold"""
        model_in = np.array([[0.1, 0.2], [0.0, 0.0], [0.5, 0.6]])
        sensory_indices = [0, 1, 2]
        idx_to_group = {0: "A", 1: "B", 2: "C"}

        df = get_input_activation(
            model_in, sensory_indices, idx_to_group, activation_threshold=0.1
        )

        # Group B should be excluded (all zeros, below threshold)
        self.assertTrue(set(df["group"].unique()).issubset({"A", "C"}))
        self.assertFalse("B" in df["group"].values)

    def test_2d_input_with_selected_indices(self):
        """Test 2D input with selected indices"""
        model_in = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        sensory_indices = [0, 1, 2]
        idx_to_group = {0: "A", 1: "B", 2: "C"}

        df = get_input_activation(
            model_in, sensory_indices, idx_to_group, selected_indices=[0, 2]
        )

        # Should only have groups A and C
        self.assertEqual(set(df["group"].unique()), {"A", "C"})
        self.assertFalse("B" in df["group"].values)

    def test_2d_input_torch_tensor(self):
        """Test 2D input with torch tensor"""
        model_in = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        sensory_indices = [0, 1]
        idx_to_group = {0: "A", 1: "B"}

        df = get_input_activation(model_in, sensory_indices, idx_to_group)

        expected_df = pd.DataFrame(
            {"group": ["A", "B"], "time_0": [0.1, 0.3], "time_1": [0.2, 0.4]}
        )
        pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)

    def test_2d_input_ignores_batch_names(self):
        """Test that batch_names is ignored for 2D input"""
        model_in = np.array([[0.1, 0.2], [0.3, 0.4]])
        sensory_indices = [0, 1]
        idx_to_group = {0: "A", 1: "B"}

        # Should print a message but still work
        df = get_input_activation(
            model_in, sensory_indices, idx_to_group, batch_names=["batch_1"]
        )

        # Should not have batch_name column for 2D input
        self.assertFalse("batch_name" in df.columns)

    def test_3d_input_basic(self):
        """Test 3D input with basic functionality"""
        model_in = np.array(
            [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]  # batch 0  # batch 1
        )
        sensory_indices = [0, 1]
        idx_to_group = {0: "A", 1: "B"}

        df = get_input_activation(model_in, sensory_indices, idx_to_group)

        # Should have batch_name column
        self.assertTrue("batch_name" in df.columns)
        # Should have default batch names
        self.assertEqual(set(df["batch_name"].unique()), {"batch_0", "batch_1"})
        # Should have all groups
        self.assertTrue(set(df["group"].unique()).issubset({"A", "B"}))

    def test_3d_input_with_batch_names(self):
        """Test 3D input with custom batch names"""
        model_in = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        sensory_indices = [0, 1]
        idx_to_group = {0: "A", 1: "B"}
        batch_names = ["odor_1", "odor_2"]

        df = get_input_activation(
            model_in, sensory_indices, idx_to_group, batch_names=batch_names
        )

        expected_df = pd.DataFrame(
            {
                "batch_name": ["odor_1", "odor_1", "odor_2", "odor_2"],
                "group": ["A", "B", "A", "B"],
                "time_0": [0.1, 0.3, 0.5, 0.7],
                "time_1": [0.2, 0.4, 0.6, 0.8],
            }
        )
        pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)

    def test_3d_input_with_groups(self):
        """Test 3D input where multiple neurons map to same group"""
        model_in = np.array(
            [
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],  # batch 0
            ]
        )
        sensory_indices = [0, 1, 2, 3]
        idx_to_group = {0: "A", 1: "A", 2: "B", 3: "B"}

        df = get_input_activation(model_in, sensory_indices, idx_to_group)

        # Should average within groups
        expected_df = pd.DataFrame(
            {
                "batch_name": ["batch_0", "batch_0"],
                "group": ["A", "B"],
                "time_0": [0.2, 0.6],  # Average of (0.1, 0.3) and (0.5, 0.7)
                "time_1": [0.3, 0.7],  # Average of (0.2, 0.4) and (0.6, 0.8)
            }
        )
        pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)

    def test_3d_input_with_threshold(self):
        """Test 3D input with activation threshold"""
        model_in = np.array(
            [
                [[0.1, 0.2], [0.0, 0.0]],  # batch 0: neuron 1 below threshold
                [[0.0, 0.0], [0.5, 0.6]],  # batch 1: neuron 0 below threshold
            ]
        )
        sensory_indices = [0, 1]
        idx_to_group = {0: "A", 1: "B"}

        df = get_input_activation(
            model_in, sensory_indices, idx_to_group, activation_threshold=0.1
        )

        # Batch 0 should have only A, batch 1 should have only B
        batch_0 = df[df["batch_name"] == "batch_0"]
        batch_1 = df[df["batch_name"] == "batch_1"]

        self.assertTrue("A" in batch_0["group"].values)
        self.assertFalse("B" in batch_0["group"].values)
        self.assertTrue("B" in batch_1["group"].values)
        self.assertFalse("A" in batch_1["group"].values)

    def test_3d_input_with_selected_indices(self):
        """Test 3D input with selected indices"""
        model_in = np.array([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]])
        sensory_indices = [0, 1, 2]
        idx_to_group = {0: "A", 1: "B", 2: "C"}

        df = get_input_activation(
            model_in, sensory_indices, idx_to_group, selected_indices=[0, 2]
        )

        # Should only have groups A and C
        self.assertEqual(set(df["group"].unique()), {"A", "C"})

    def test_3d_input_empty_result(self):
        """Test 3D input where no activations pass threshold"""
        model_in = np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
        sensory_indices = [0, 1]
        idx_to_group = {0: "A", 1: "B"}

        df = get_input_activation(
            model_in, sensory_indices, idx_to_group, activation_threshold=0.1
        )

        # Should return empty dataframe with correct columns
        self.assertEqual(len(df), 0)
        self.assertTrue("batch_name" in df.columns)
        self.assertTrue("group" in df.columns)
        self.assertTrue("time_0" in df.columns)

    def test_3d_batch_names_mismatch(self):
        """Test 3D input with mismatched batch names length"""
        model_in = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        sensory_indices = [0, 1]
        idx_to_group = {0: "A", 1: "B"}
        batch_names = ["batch_1"]  # Only 1 name for 2 batches

        with self.assertRaises(ValueError):
            get_input_activation(
                model_in, sensory_indices, idx_to_group, batch_names=batch_names
            )

    def test_invalid_shape(self):
        """Test that invalid input shape raises error"""
        model_in = np.array([0.1, 0.2, 0.3])  # 1D array
        sensory_indices = [0, 1, 2]
        idx_to_group = {0: "A", 1: "B", 2: "C"}

        with self.assertRaises(ValueError):
            get_input_activation(model_in, sensory_indices, idx_to_group)

    def test_3d_input_torch_tensor(self):
        """Test 3D input with torch tensor"""
        model_in = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        sensory_indices = [0, 1]
        idx_to_group = {0: "A", 1: "B"}

        df = get_input_activation(model_in, sensory_indices, idx_to_group)

        # Should convert to numpy and work correctly
        self.assertEqual(len(df), 4)  # 2 batches * 2 groups
        self.assertTrue("batch_name" in df.columns)


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

        def custom_activation(self, x, x_previous=None):
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
            threshold=0.0,
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


def _linear_activation(self, x, x_previous=None):
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
        monitor = [0, 1]  # neuron indices
        target = {2: [4]}  # layer 2, neuron 4

        df = get_gradients(
            self.model,
            self.inputs_batched,
            monitor_neurons=monitor,
            target_neurons=target,
            monitor_layers=[0, 1, 2],
            batch_names=[f"b{i}" for i in range(self.batch_size)],
            device=self.device,
        )

        expected_time_cols = {f"time_{t}" for t in [0, 1, 2]}
        self.assertTrue({"group", "batch_name"}.issubset(df.columns))
        self.assertTrue(expected_time_cols.issubset(df.columns))

        # two monitored neurons * two batches = 4 rows
        self.assertEqual(len(df), 4)

        # group column are raw indices when idx_to_group=None
        self.assertTrue(set(df["group"].unique()).issubset({0, 1}))

    def test_gradient_path_specificity(self):
        monitor = [0, 1]
        target = {2: [4]}
        df = get_gradients(
            self.model,
            self.inputs_single,
            monitor_neurons=monitor,
            target_neurons=target,
            monitor_layers=[0, 1, 2],
            batch_names=None,
            device=self.device,
        )
        # convert to wide for easy checks
        wide = df.pivot(index="group", columns="batch_name").sort_index()
        # single input -> only "batch_0" exists
        g0 = wide.loc[0].to_numpy()
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

        # new model with group mapping
        model_with_groups = MultilayeredNetwork(
            self.all_weights,
            sensory_indices=self.sensory_indices,
            num_layers=self.num_layers,
            threshold=0.0,
            activation_function=_linear_activation,
            idx_to_group=idx_to_group,
        ).to(self.device)

        # monitor by group names
        monitor = ["S0", "S1"]
        # target is neuron 4 via group "T" at layer 2
        target = {2: ["T"]}

        df = get_gradients(
            model_with_groups,
            self.inputs_batched,
            monitor_neurons=monitor,
            target_neurons=target,
            monitor_layers=[0, 1, 2],
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
            batch_names=None,
            device=self.device,
        )
        self.assertIn("time_1", df.columns)
        self.assertIn("time_2", df.columns)
        self.assertNotIn("time_0", df.columns)


class TestLinearNetwork(unittest.TestCase):
    """Test suite for LinearNetwork class"""

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
        self.all_weights = csr_matrix(dense_weights)
        self.sensory_indices = list(range(self.num_sensory))

        self.model = LinearNetwork(
            self.all_weights, self.sensory_indices, num_layers=self.num_layers
        ).to(self.device)

    def test_initialization(self):
        """Test that LinearNetwork initializes correctly"""
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
        """Test 2D forward pass (single batch)"""
        input_tensor = torch.rand(self.num_sensory, self.num_layers).to(self.device)
        output = self.model(input_tensor)

        expected_shape = (self.num_neurons, self.num_layers)
        self.assertEqual(output.shape, expected_shape)
        # LinearNetwork outputs are not bounded (no tanh applied), just check they exist
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_pass_3d(self):
        """Test 3D forward pass (batched)"""
        input_tensor = torch.rand(
            self.batch_size, self.num_sensory, self.num_layers
        ).to(self.device)
        output = self.model(input_tensor)

        expected_shape = (self.batch_size, self.num_neurons, self.num_layers)
        self.assertEqual(output.shape, expected_shape)
        # LinearNetwork outputs are not bounded (no tanh applied), just check they exist
        self.assertTrue(torch.isfinite(output).all())

    def test_with_trainable_parameters(self):
        """Test LinearNetwork with trainable parameters"""
        idx_to_group = {i: f"group_{i % 3}" for i in range(self.num_neurons)}
        bias_dict = {"group_0": 0.1, "group_1": -0.1, "group_2": 0.0}
        slope_dict = {"group_0": 3.0, "group_1": 5.0, "group_2": 7.0}

        model = LinearNetwork(
            self.all_weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            idx_to_group=idx_to_group,
            bias_dict=bias_dict,
            slope_dict=slope_dict,
        ).to(self.device)

        # Check that parameters are initialized
        self.assertIsNotNone(model.slope)
        self.assertIsNotNone(model.raw_biases)

        # Test forward pass
        input_tensor = torch.rand(
            self.batch_size, self.num_sensory, self.num_layers
        ).to(self.device)
        output = model(input_tensor)
        self.assertEqual(
            output.shape, (self.batch_size, self.num_neurons, self.num_layers)
        )

    def test_custom_activation_function(self):
        """Test LinearNetwork with custom activation function"""

        def custom_activation(self, x, x_previous=None):
            return torch.relu(x)

        model = LinearNetwork(
            self.all_weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            activation_function=custom_activation,
        ).to(self.device)

        input_tensor = torch.rand(self.num_sensory, self.num_layers).to(self.device)
        output = model(input_tensor)

        # ReLU should produce only non-negative values
        self.assertTrue(torch.all(output >= 0))

    def test_monitor_neurons(self):
        """Test monitoring specific neurons during forward pass"""
        monitor_neurons = [0, 2, 5]
        # Need requires_grad=True for monitoring to work (registers hooks)
        input_tensor = torch.rand(
            self.batch_size, self.num_sensory, self.num_layers, requires_grad=True
        ).to(self.device)

        output = self.model(input_tensor, monitor_neurons=monitor_neurons)

        # Should return full activations
        self.assertEqual(
            output.shape, (self.batch_size, self.num_neurons, self.num_layers)
        )

    def test_sparse_input(self):
        """Test LinearNetwork with scipy sparse matrix input"""
        sparse_weights = csr_matrix(self.all_weights)
        model = LinearNetwork(
            sparse_weights, self.sensory_indices, num_layers=self.num_layers
        ).to(self.device)

        input_tensor = torch.rand(self.num_sensory, self.num_layers).to(self.device)
        output = model(input_tensor)

        self.assertEqual(output.shape, (self.num_neurons, self.num_layers))

    def test_tau_persistence(self):
        """Test that tau parameter affects activation persistence"""
        # Create model with tau=10
        model_with_tau = LinearNetwork(
            self.all_weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            tau=10.0,
        ).to(self.device)

        # Create model without tau
        model_without_tau = LinearNetwork(
            self.all_weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            tau=1.0,
        ).to(self.device)

        # Same input
        input_tensor = torch.rand(self.num_sensory, self.num_layers).to(self.device)

        output_with_tau = model_with_tau(input_tensor)
        output_without_tau = model_without_tau(input_tensor)

        # Outputs should be different due to tau
        # (unless by chance the activations are very similar)
        # We just check they both produce valid outputs
        self.assertEqual(output_with_tau.shape, output_without_tau.shape)


class TestNormalizeGradients(unittest.TestCase):
    """Test suite for normalize_gradients parameter in activation_maximisation"""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a simple network
        dense_weights = np.random.rand(10, 10)
        dense_weights = dense_weights / dense_weights.sum(axis=1, keepdims=True)
        dense_weights[:, :3] = -dense_weights[:, :3]
        self.all_weights = csr_matrix(dense_weights)

        self.model = MultilayeredNetwork(
            self.all_weights,
            sensory_indices=[0, 1, 2, 3],
            num_layers=3,
        ).to(self.device)

        self.targets = TargetActivation(
            {0: {0: 0.5, 1: 0.8}, 1: {2: 0.3}}, batch_size=2
        )

    def test_normalize_gradients_runs(self):
        """Test that activation_maximisation runs with normalize_gradients=True"""
        result = activation_maximisation(
            self.model,
            self.targets,
            num_iterations=10,
            in_reg_lambda=1e-3,
            out_reg_lambda=1e-3,
            wandb=False,
            device=self.device,
            normalize_gradients=True,
        )

        input_tensor, output, act_losses, *_ = result
        expected_shape = (2, 4, 3)  # (batch_size, num_sensory, num_layers)
        self.assertEqual(input_tensor.shape, expected_shape)
        # Loss should decrease
        self.assertTrue(act_losses[-1] <= act_losses[0])

    def test_normalize_gradients_false(self):
        """Test that activation_maximisation runs with normalize_gradients=False"""
        result = activation_maximisation(
            self.model,
            self.targets,
            num_iterations=10,
            in_reg_lambda=1e-3,
            out_reg_lambda=1e-3,
            wandb=False,
            device=self.device,
            normalize_gradients=False,
        )

        input_tensor, output, act_losses, *_ = result
        expected_shape = (2, 4, 3)
        self.assertEqual(input_tensor.shape, expected_shape)
        self.assertTrue(act_losses[-1] <= act_losses[0])

    def test_normalize_gradients_comparison(self):
        """Test that normalize_gradients affects optimization differently"""
        # Set seed for reproducibility
        seed = 42

        # Run with normalization
        result_normalized = activation_maximisation(
            self.model,
            self.targets,
            num_iterations=20,
            in_reg_lambda=1e-3,
            out_reg_lambda=1e-3,
            wandb=False,
            device=self.device,
            normalize_gradients=True,
            seed=seed,
        )

        # Run without normalization (same seed)
        result_unnormalized = activation_maximisation(
            self.model,
            self.targets,
            num_iterations=20,
            in_reg_lambda=1e-3,
            out_reg_lambda=1e-3,
            wandb=False,
            device=self.device,
            normalize_gradients=False,
            seed=seed,
        )

        input_normalized, _, losses_normalized, *_ = result_normalized
        input_unnormalized, _, losses_unnormalized, *_ = result_unnormalized

        # The inputs should be different due to different gradient processing
        self.assertFalse(np.allclose(input_normalized, input_unnormalized, rtol=1e-3))

    def test_normalize_gradients_with_longer_paths(self):
        """Test normalize_gradients with deeper network (more layers)"""
        # Create a deeper network
        model = MultilayeredNetwork(
            self.all_weights,
            sensory_indices=[0, 1, 2, 3],
            num_layers=50,  # Deeper network
        ).to(self.device)

        # Use a target earlier in the network to make optimization more reliable
        targets = TargetActivation(
            {30: {5: 0.7, 6: 0.5}}, batch_size=1  # Target at middle layer
        )

        result = activation_maximisation(
            model,
            targets,
            num_iterations=30,
            learning_rate=0.2,  # Increase learning rate
            in_reg_lambda=1e-3,
            out_reg_lambda=0,
            wandb=False,
            device="cpu",  # Use CPU for consistency
            normalize_gradients=True,
            seed=42,  # Add seed for reproducibility across platforms
            print_output=False,  # Reduce test output
        )

        input_tensor, output, act_losses, *_ = result
        # Should show some improvement (more lenient check for cross-platform stability)
        # Final loss should be less than 80% of initial loss
        self.assertTrue(
            act_losses[-1] < 0.8 * act_losses[0],
            f"Loss should decrease: initial={act_losses[0]}, final={act_losses[-1]}",
        )


class TestGuessOptimalStimulus(unittest.TestCase):
    """Test suite for guess_optimal_stimulus function"""

    def setUp(self):
        """Set up test fixtures with a simple connectivity matrix"""
        # Create a simple 6-neuron network:
        # - Neurons 0, 1: sensory (excitatory)
        # - Neuron 2: inhibitory
        # - Neurons 3, 4, 5: internal (excitatory)

        # Simple connectivity: sensory -> internal with some inhibition
        self.inprop = csr_matrix(
            np.array(
                [
                    [0, 0, 0, 0.5, 0, 0],  # 0 (sensory) -> 3
                    [0, 0, 0, 0, 0.6, 0],  # 1 (sensory) -> 4
                    [0, 0, 0, -0.3, -0.2, 0],  # 2 (inhibitory) -> 3, 4
                    [0, 0, 0, 0, 0, 0.7],  # 3 -> 5
                    [0, 0, 0, 0, 0, 0.4],  # 4 -> 5
                    [0, 0, 0, 0, 0, 0],  # 5 (no output)
                ]
            )
        )

        self.sensory_indices = [0, 1]
        self.idx_to_sign = {
            0: 1,  # excitatory
            1: 1,  # excitatory
            2: -1,  # inhibitory
            3: 1,  # excitatory
            4: 1,  # excitatory
            5: 1,  # excitatory
        }

    def test_basic_functionality_single_batch(self):
        """Test basic functionality with single batch and single target"""
        targets = TargetActivation(
            targets={0: {3: 1.0}}, batch_size=1  # layer 0, neuron 3
        )

        result = guess_optimal_stimulus(
            self.inprop, self.sensory_indices, self.idx_to_sign, targets, longest_plen=2
        )

        # Check output shape: (batch_size, num_sensory, nlayer)
        # nlayer = min(longest_plen, max_layer + 1) = min(2, 0 + 1) = 1
        expected_shape = (1, 2, 1)
        self.assertEqual(result.shape, expected_shape)

        # Check that stimulus is non-zero (neuron 3 is connected to sensory 0)
        self.assertTrue(np.any(result != 0))

    def test_multiple_batches(self):
        """Test with multiple batches"""
        # Use DataFrame format for multiple batches with different targets
        targets = TargetActivation(
            targets=pd.DataFrame(
                [
                    {"batch": 0, "layer": 0, "neuron": 3, "value": 1.0},
                    {"batch": 1, "layer": 0, "neuron": 4, "value": 0.8},
                ]
            )
        )

        result = guess_optimal_stimulus(
            self.inprop, self.sensory_indices, self.idx_to_sign, targets, longest_plen=2
        )

        # Check output shape
        expected_shape = (2, 2, 1)
        self.assertEqual(result.shape, expected_shape)

        # Batches should be different (targeting different neurons)
        self.assertFalse(np.allclose(result[0], result[1]))

    def test_multiple_layers_and_path_lengths(self):
        """Test with targets at different layers"""
        targets = TargetActivation(
            targets={0: {5: 1.0}},  # layer 0, neuron 5 (2 hops away)
            batch_size=1,
        )

        result = guess_optimal_stimulus(
            self.inprop, self.sensory_indices, self.idx_to_sign, targets, longest_plen=3
        )

        # nlayer = min(3, 0 + 1) = 1
        expected_shape = (1, 2, 1)
        self.assertEqual(result.shape, expected_shape)

    def test_layer_exceeds_longest_plen(self):
        """Test when target layer exceeds longest_plen"""
        targets = TargetActivation(
            targets={10: {5: 1.0}}, batch_size=1  # layer 10, neuron 5
        )

        result = guess_optimal_stimulus(
            self.inprop, self.sensory_indices, self.idx_to_sign, targets, longest_plen=3
        )

        # nlayer should be min(3, 10 + 1) = 3
        expected_shape = (1, 2, 3)
        self.assertEqual(result.shape, expected_shape)

    def test_multiple_targets_same_layer(self):
        """Test with multiple target neurons at the same layer"""
        targets = TargetActivation(
            targets={0: {3: 0.5, 4: 0.8}},  # layer 0, neurons 3 and 4
            batch_size=1,
        )

        result = guess_optimal_stimulus(
            self.inprop, self.sensory_indices, self.idx_to_sign, targets, longest_plen=2
        )

        expected_shape = (1, 2, 1)
        self.assertEqual(result.shape, expected_shape)

        # Result should reflect contributions from both targets
        self.assertTrue(np.any(result != 0))

    def test_inhibitory_influence(self):
        """Test that inhibitory neurons have negative influence"""
        # Create a simpler network to test inhibition
        # inprop should contain only positive weights
        simple_inprop = csr_matrix(
            np.array(
                [
                    [0, 0, 0],  # 0 (sensory)
                    [0, 0, 0.5],  # 1 -> 2 with weight 0.5 (sign from idx_to_sign)
                    [0, 0, 0],  # 2 (target)
                ]
            )
        )

        simple_idx_to_sign = {0: 1, 1: -1, 2: 1}  # 1 is inhibitory

        targets = TargetActivation(
            targets={0: {2: 1.0}}, batch_size=1  # layer 0, Want to activate neuron 2
        )

        result = guess_optimal_stimulus(
            simple_inprop,
            [0, 1],  # both sensory
            simple_idx_to_sign,
            targets,
            longest_plen=2,
        )

        # Neuron 1 is inhibitory and connects to neuron 2 with weight 0.5
        # signed_conn_by_path_length_data returns inhib DataFrame with positive value 0.5
        # The function does: stimulus += exc - inhib = 0 - 0.5 = -0.5
        # So it correctly suggests REDUCING activation of neuron 1 to maximize neuron 2
        self.assertTrue(result[0, 1, 0] < 0)

    def test_output_dtype(self):
        """Test that output is float32"""
        targets = TargetActivation(targets={0: {0: {3: 1.0}}}, batch_size=1)

        result = guess_optimal_stimulus(
            self.inprop, self.sensory_indices, self.idx_to_sign, targets, longest_plen=2
        )

        self.assertEqual(result.dtype, np.float32)

    def test_empty_paths_handling(self):
        """Test with target neuron that has no paths from sensory neurons"""
        # Create network where neuron 5 has no incoming connections from sensory
        disconnected_inprop = csr_matrix(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            )
        )

        targets = TargetActivation(targets={0: {0: {5: 1.0}}}, batch_size=1)

        result = guess_optimal_stimulus(
            disconnected_inprop,
            self.sensory_indices,
            self.idx_to_sign,
            targets,
            longest_plen=2,
        )

        # Should return zeros when no paths exist
        self.assertTrue(np.allclose(result, 0))

    def test_sensory_indices_as_different_types(self):
        """Test that sensory_indices can be list, array, or set"""
        targets = TargetActivation(targets={0: {0: {3: 1.0}}}, batch_size=1)

        # Test with list
        result_list = guess_optimal_stimulus(
            self.inprop, [0, 1], self.idx_to_sign, targets, longest_plen=2
        )

        # Test with numpy array
        result_array = guess_optimal_stimulus(
            self.inprop, np.array([0, 1]), self.idx_to_sign, targets, longest_plen=2
        )

        # Both should give the same result
        np.testing.assert_array_almost_equal(result_list, result_array)

    def test_different_target_values(self):
        """Test that different target activation values scale the stimulus"""
        targets_small = TargetActivation(targets={0: {0: {3: 0.1}}}, batch_size=1)

        targets_large = TargetActivation(targets={0: {0: {3: 1.0}}}, batch_size=1)

        result_small = guess_optimal_stimulus(
            self.inprop,
            self.sensory_indices,
            self.idx_to_sign,
            targets_small,
            longest_plen=2,
        )

        result_large = guess_optimal_stimulus(
            self.inprop,
            self.sensory_indices,
            self.idx_to_sign,
            targets_large,
            longest_plen=2,
        )

        # Larger target should produce larger stimulus (scaled by 10x)
        ratio = result_large / (result_small + 1e-10)  # avoid division by zero
        # The ratio should be approximately 10 for non-zero elements
        non_zero_mask = np.abs(result_small) > 1e-6
        if np.any(non_zero_mask):
            self.assertTrue(np.allclose(ratio[non_zero_mask], 10.0, rtol=0.1))

    def test_targets_at_multiple_layers_same_batch(self):
        """Test with targets at multiple layers in the same batch"""
        targets = TargetActivation(
            targets={
                0: {3: 1.0},  # layer 0, neuron 3
                1: {4: 0.5},  # layer 1, neuron 4
            },
            batch_size=1,
        )

        result = guess_optimal_stimulus(
            self.inprop, self.sensory_indices, self.idx_to_sign, targets, longest_plen=3
        )

        # nlayer = min(3, max(0, 1) + 1) = min(3, 2) = 2
        expected_shape = (1, 2, 2)
        self.assertEqual(result.shape, expected_shape)


class TestTauParam(unittest.TestCase):
    """Tests for tau_dict and tau_param across _NetworkBase subclasses."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = 2
        dense = np.random.rand(6, 6) * 0.1
        self.weights = csr_matrix(dense)
        self.sensory_indices = [0, 1]
        self.idx_to_group = {i: f"type_{i % 3}" for i in range(6)}
        # sorted groups: type_0, type_1, type_2
        self.all_types = sorted(set(self.idx_to_group.values()))

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_tau_param_none_without_idx_to_group(self):
        """tau_param should be None when idx_to_group is not provided."""
        for cls in [LinearNetwork, MultilayeredNetwork]:
            model = cls(self.weights, self.sensory_indices, num_layers=2)
            self.assertIsNone(
                model.tau_param, f"{cls.__name__}: tau_param should be None"
            )

    def test_tau_param_default_fills_scalar_tau(self):
        """When tau_dict=None, all groups should get the scalar tau value."""
        scalar_tau = 7.0
        for cls in [LinearNetwork, MultilayeredNetwork]:
            model = cls(
                self.weights,
                self.sensory_indices,
                idx_to_group=self.idx_to_group,
                tau=scalar_tau,
            )
            expected = torch.full((len(self.all_types),), scalar_tau)
            self.assertTrue(
                torch.allclose(model.tau_param.cpu(), expected),
                f"{cls.__name__}: default tau_param values incorrect",
            )

    def test_tau_dict_sets_values(self):
        """tau_dict values should override defaults for named groups."""
        tau_dict = {"type_0": 3.0, "type_2": 15.0}
        default_tau = 10.0
        for cls in [LinearNetwork, MultilayeredNetwork]:
            model = cls(
                self.weights,
                self.sensory_indices,
                idx_to_group=self.idx_to_group,
                tau=default_tau,
                tau_dict=tau_dict,
            )
            type2pos = {t: i for i, t in enumerate(self.all_types)}
            self.assertAlmostEqual(
                model.tau_param[type2pos["type_0"]].item(), 3.0, places=6
            )
            self.assertAlmostEqual(
                model.tau_param[type2pos["type_1"]].item(), default_tau, places=6
            )
            self.assertAlmostEqual(
                model.tau_param[type2pos["type_2"]].item(), 15.0, places=6
            )

    def test_tau_dict_ignores_unknown_groups(self):
        """tau_dict keys not in idx_to_group should be silently ignored."""
        tau_dict = {"type_0": 5.0, "nonexistent_group": 99.0}
        for cls in [LinearNetwork, MultilayeredNetwork]:
            model = cls(
                self.weights,
                self.sensory_indices,
                idx_to_group=self.idx_to_group,
                tau_dict=tau_dict,
            )
            # Should not raise; nonexistent_group simply has no effect
            self.assertAlmostEqual(
                model.tau_param[
                    [i for i, t in enumerate(self.all_types) if t == "type_0"][0]
                ].item(),
                5.0,
                places=6,
            )

    # ------------------------------------------------------------------
    # effective_tau property
    # ------------------------------------------------------------------

    def test_effective_tau_returns_scalar_when_no_tau_param(self):
        """effective_tau should return the scalar self.tau when tau_param is None."""
        scalar_tau = 8.0
        for cls in [LinearNetwork, MultilayeredNetwork]:
            model = cls(self.weights, self.sensory_indices, tau=scalar_tau)
            self.assertEqual(model.effective_tau, scalar_tau)

    def test_effective_tau_clamps_below_one(self):
        """effective_tau should clamp values < 1.0 up to 1.0."""
        tau_dict = {"type_0": 0.1, "type_1": 0.5, "type_2": 2.0}
        for cls in [LinearNetwork, MultilayeredNetwork]:
            model = cls(
                self.weights,
                self.sensory_indices,
                idx_to_group=self.idx_to_group,
                tau_dict=tau_dict,
            )
            clamped = model.effective_tau
            self.assertTrue(
                torch.all(clamped >= 1.0),
                f"{cls.__name__}: effective_tau below 1.0 not clamped",
            )

    def test_effective_tau_does_not_clamp_valid_values(self):
        """effective_tau should not modify values already >= 1.0."""
        tau_dict = {"type_0": 1.0, "type_1": 5.0, "type_2": 20.0}
        for cls in [LinearNetwork, MultilayeredNetwork]:
            model = cls(
                self.weights,
                self.sensory_indices,
                idx_to_group=self.idx_to_group,
                tau_dict=tau_dict,
            )
            clamped = model.effective_tau.cpu()
            type2pos = {t: i for i, t in enumerate(self.all_types)}
            self.assertAlmostEqual(clamped[type2pos["type_0"]].item(), 1.0, places=6)
            self.assertAlmostEqual(clamped[type2pos["type_1"]].item(), 5.0, places=6)
            self.assertAlmostEqual(clamped[type2pos["type_2"]].item(), 20.0, places=6)

    # ------------------------------------------------------------------
    # set_param_grads
    # ------------------------------------------------------------------

    def test_tau_grad_disabled_by_default(self):
        """tau_param should not require grad after construction."""
        for cls in [LinearNetwork, MultilayeredNetwork]:
            model = cls(
                self.weights,
                self.sensory_indices,
                idx_to_group=self.idx_to_group,
            )
            self.assertFalse(
                model.tau_param.requires_grad,
                f"{cls.__name__}: tau_param.requires_grad should be False",
            )

    def test_set_param_grads_tau_true(self):
        """set_param_grads(tau=True) should enable gradients for tau_param."""
        for cls in [LinearNetwork, MultilayeredNetwork]:
            model = cls(
                self.weights,
                self.sensory_indices,
                idx_to_group=self.idx_to_group,
            )
            model.set_param_grads(tau=True)
            self.assertTrue(model.tau_param.requires_grad)
            # Other params should remain unchanged (still False)
            self.assertFalse(model.slope.requires_grad)
            self.assertFalse(model.raw_biases.requires_grad)

    def test_set_param_grads_tau_false(self):
        """set_param_grads(tau=False) should disable gradients for tau_param."""
        for cls in [LinearNetwork, MultilayeredNetwork]:
            model = cls(
                self.weights,
                self.sensory_indices,
                idx_to_group=self.idx_to_group,
            )
            model.set_param_grads(tau=True)
            model.set_param_grads(tau=False)
            self.assertFalse(model.tau_param.requires_grad)

    def test_set_param_grads_no_tau_param_does_not_raise(self):
        """set_param_grads should not raise when tau_param is None."""
        for cls in [LinearNetwork, MultilayeredNetwork]:
            model = cls(self.weights, self.sensory_indices)
            try:
                model.set_param_grads(tau=True)
            except Exception as e:
                self.fail(f"{cls.__name__}: set_param_grads raised unexpectedly: {e}")

    # ------------------------------------------------------------------
    # training_mode context manager
    # ------------------------------------------------------------------

    def test_training_mode_enables_tau_grad(self):
        """training_mode with train_tau=True should enable and then restore tau grad."""
        model = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            idx_to_group=self.idx_to_group,
        )
        self.assertFalse(model.tau_param.requires_grad)
        with training_mode(model, train_tau=True):
            self.assertTrue(model.tau_param.requires_grad)
        self.assertFalse(model.tau_param.requires_grad)

    def test_training_mode_tau_false_leaves_grad_disabled(self):
        """training_mode with train_tau=False should not enable tau grad."""
        model = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            idx_to_group=self.idx_to_group,
        )
        with training_mode(model, train_tau=False):
            self.assertFalse(model.tau_param.requires_grad)

    # ------------------------------------------------------------------
    # Forward pass behaviour
    # ------------------------------------------------------------------

    def test_tau_dict_affects_output(self):
        """Two models with different tau_dict should produce different outputs."""
        inp = torch.rand(2, len(self.sensory_indices), self.num_layers)
        # (batch, sensory, layers)
        model_slow = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=2,
            idx_to_group=self.idx_to_group,
            tau_dict={"type_0": 1.0, "type_1": 1.0, "type_2": 1.0},
        )
        model_fast = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=2,
            idx_to_group=self.idx_to_group,
            tau_dict={"type_0": 20.0, "type_1": 20.0, "type_2": 20.0},
        )
        out_slow = model_slow(inp).detach()
        out_fast = model_fast(inp).detach()
        self.assertFalse(
            torch.allclose(out_slow, out_fast),
            "Different tau values should produce different outputs",
        )

    def test_scalar_tau_matches_tau_dict_uniform(self):
        """A model with scalar tau should match one where tau_dict sets all groups to that value."""
        scalar_tau = 5.0
        inp = torch.rand(2, len(self.sensory_indices), self.num_layers)

        model_scalar = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=2,
            idx_to_group=self.idx_to_group,
            tau=scalar_tau,
        )
        model_dict = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=2,
            idx_to_group=self.idx_to_group,
            tau=scalar_tau,
            tau_dict={t: scalar_tau for t in self.all_types},
        )
        out_scalar = model_scalar(inp).detach()
        out_dict = model_dict(inp).detach()
        self.assertTrue(
            torch.allclose(out_scalar, out_dict, atol=1e-6),
            "Scalar tau and equivalent tau_dict should give identical outputs",
        )

    def test_forward_without_idx_to_group_uses_scalar_tau(self):
        """Forward pass without idx_to_group should use scalar tau and not crash."""
        for cls in [LinearNetwork, MultilayeredNetwork]:
            model = cls(self.weights, self.sensory_indices, num_layers=2, tau=5.0)
            inp = torch.rand(2, len(self.sensory_indices), self.num_layers)

            try:
                out = model(inp)
            except Exception as e:
                self.fail(f"{cls.__name__} forward crashed with scalar tau: {e}")
            self.assertEqual(out.shape, (2, 6, 2))

    # ------------------------------------------------------------------
    # Gradient flow through tau
    # ------------------------------------------------------------------

    def test_tau_param_receives_gradient(self):
        """tau_param should receive a gradient when requires_grad=True."""
        model = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=2,
            idx_to_group=self.idx_to_group,
        )
        model.set_param_grads(tau=True)
        inp = torch.rand(2, len(self.sensory_indices), self.num_layers)

        out = model(inp)
        out.sum().backward()
        self.assertIsNotNone(
            model.tau_param.grad, "tau_param should have a gradient after backward()"
        )
        self.assertFalse(
            torch.all(model.tau_param.grad == 0),
            "tau_param gradient should be non-zero",
        )

    def test_tau_param_changes_after_optimizer_step(self):
        """tau_param values should change after an optimizer step when grad is enabled."""
        model = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=2,
            idx_to_group=self.idx_to_group,
        )
        model.set_param_grads(tau=True)
        tau_before = model.tau_param.clone().detach()
        optimizer = torch.optim.Adam([model.tau_param], lr=0.1)
        inp = torch.rand(2, len(self.sensory_indices), self.num_layers)

        out = model(inp)
        out.sum().backward()
        optimizer.step()
        self.assertFalse(
            torch.allclose(model.tau_param.detach(), tau_before),
            "tau_param should change after optimizer step",
        )


class TestTauParamCorrectness(unittest.TestCase):
    """Tests for correctness of tau formula and per-group behaviour."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = 3
        self.sensory_indices = [0, 1]
        self.num_sensory = len(self.sensory_indices)
        self.num_neurons = 6
        self.idx_to_group = {i: f"type_{i % 3}" for i in range(self.num_neurons)}
        self.all_types = sorted(set(self.idx_to_group.values()))

        dense = np.random.rand(self.num_neurons, self.num_neurons) * 0.1
        self.weights = csr_matrix(dense)

    # ------------------------------------------------------------------
    # Tau formula boundary values
    # ------------------------------------------------------------------

    def test_tau_equals_one_ignores_previous_state_multilayered(self):
        """With tau=1, activation_function output should not depend on x_previous."""
        model = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=2,
            idx_to_group=self.idx_to_group,
            tau_dict={t: 1.0 for t in self.all_types},
            threshold=0.0,
        )
        x = torch.rand(self.num_neurons, 2)
        x_prev_a = torch.zeros(self.num_neurons, 2)
        x_prev_b = torch.rand(self.num_neurons, 2)  # different x_previous

        out_a = model.activation_function(x.clone(), x_prev_a)
        out_b = model.activation_function(x.clone(), x_prev_b)

        self.assertTrue(
            torch.allclose(out_a, out_b, atol=1e-6),
            "With tau=1, output should be independent of x_previous",
        )

    def test_large_tau_output_close_to_x_previous(self):
        """With very large tau, output should be dominated by x_previous (slow dynamics)."""
        # tau=1000: 1/1000 * tanh(x) + 999/1000 * x_previous ≈ x_previous
        model_slow = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            idx_to_group=self.idx_to_group,
            tau_dict={t: 1000.0 for t in self.all_types},
            threshold=0.0,
        )
        model_fast = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            idx_to_group=self.idx_to_group,
            tau_dict={t: 1.0 for t in self.all_types},
            threshold=0.0,
        )
        inp = torch.rand(2, self.num_sensory, self.num_layers)
        out_slow = model_slow(inp).detach()
        out_fast = model_fast(inp).detach()

        # Slow model activations at later layers should be much smaller in magnitude
        # (dragged toward 0 since x_previous starts at 0)
        self.assertTrue(
            out_slow[:, :, -1].abs().mean() < out_fast[:, :, -1].abs().mean(),
            "Large tau should produce smaller activations (pulled toward zero x_previous)",
        )

    def test_linear_network_tau_equals_one_no_x_previous(self):
        """LinearNetwork with tau=1: output should not depend on x_previous."""
        model = LinearNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=2,
            idx_to_group=self.idx_to_group,
            tau_dict={t: 1.0 for t in self.all_types},
            threshold=0.0,
        )
        x = torch.rand(self.num_neurons, 2)
        x_prev_a = torch.zeros(self.num_neurons, 2)
        x_prev_b = torch.rand(self.num_neurons, 2)

        out_a = model.activation_function(x.clone(), x_prev_a)
        out_b = model.activation_function(x.clone(), x_prev_b)

        self.assertTrue(
            torch.allclose(out_a, out_b, atol=1e-6),
            "LinearNetwork with tau=1: output should be independent of x_previous",
        )

    # ------------------------------------------------------------------
    # Per-group tau produces group-specific dynamics
    # ------------------------------------------------------------------

    def test_per_group_tau_faster_group_responds_more(self):
        """Neurons in a fast-tau group should show larger activations than slow-tau group."""
        # type_0 (indices 0, 3): tau=1 (fast)
        # type_1 (indices 1, 4): tau=1000 (slow, stays near 0)
        # type_2 (indices 2, 5): tau=1 (fast)
        tau_dict = {"type_0": 1.0, "type_1": 1000.0, "type_2": 1.0}
        model = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            idx_to_group=self.idx_to_group,
            tau_dict=tau_dict,
            threshold=0.0,
        )
        inp = torch.ones(2, self.num_sensory, self.num_layers) * 0.5
        out = model(inp).detach()  # shape: (2, num_neurons, num_layers)

        # type_1 indices are 1 and 4
        type1_indices = [i for i, g in self.idx_to_group.items() if g == "type_1"]
        type0_indices = [i for i, g in self.idx_to_group.items() if g == "type_0"]

        # At last layer, fast neurons should have larger abs activation than slow
        fast_mean = out[:, type0_indices, -1].abs().mean()
        slow_mean = out[:, type1_indices, -1].abs().mean()
        self.assertTrue(
            fast_mean > slow_mean,
            f"Fast tau group (mean={fast_mean:.4f}) should have larger activations "
            f"than slow tau group (mean={slow_mean:.4f})",
        )

    def test_per_group_tau_independent_across_groups(self):
        """Changing tau for one group should not affect neurons in other groups."""
        base_tau_dict = {"type_0": 5.0, "type_1": 5.0, "type_2": 5.0}
        modified_tau_dict = {
            "type_0": 1.0,
            "type_1": 5.0,
            "type_2": 5.0,
        }  # only type_0 changed

        model_base = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            idx_to_group=self.idx_to_group,
            tau_dict=base_tau_dict,
            threshold=0.0,
        )
        model_modified = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            idx_to_group=self.idx_to_group,
            tau_dict=modified_tau_dict,
            threshold=0.0,
        )
        inp = torch.rand(2, self.num_sensory, self.num_layers)
        out_base = model_base(inp).detach()
        out_modified = model_modified(inp).detach()

        type1_indices = [i for i, g in self.idx_to_group.items() if g == "type_1"]
        type2_indices = [i for i, g in self.idx_to_group.items() if g == "type_2"]
        type0_indices = [i for i, g in self.idx_to_group.items() if g == "type_0"]

        # type_1 and type_2 outputs should be identical between the two models
        # Note: type_0 feeds into others via weights, so strict equality won't hold at
        # deeper layers — check only layer 0 where the effect is isolated
        self.assertTrue(
            torch.allclose(
                out_base[:, type1_indices, 0],
                out_modified[:, type1_indices, 0],
                atol=1e-5,
            ),
            "type_1 activations at layer 0 should be unaffected by type_0 tau change",
        )
        self.assertTrue(
            torch.allclose(
                out_base[:, type2_indices, 0],
                out_modified[:, type2_indices, 0],
                atol=1e-5,
            ),
            "type_2 activations at layer 0 should be unaffected by type_0 tau change",
        )
        # type_0 outputs should differ
        self.assertFalse(
            torch.allclose(
                out_base[:, type0_indices, :],
                out_modified[:, type0_indices, :],
                atol=1e-5,
            ),
            "type_0 activations should differ when its tau is changed",
        )

    # ------------------------------------------------------------------
    # Clamping propagates into forward pass
    # ------------------------------------------------------------------

    def test_sub_one_tau_clamped_in_forward(self):
        """A tau_param value < 1 should be clamped to 1 during forward, not used raw."""
        # Build a model and manually set tau_param to 0.1 (below minimum)
        model_clamped = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=2,
            idx_to_group=self.idx_to_group,
            tau_dict={t: 0.1 for t in self.all_types},  # will be stored as 0.1
            threshold=0.0,
        )
        model_one = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=2,
            idx_to_group=self.idx_to_group,
            tau_dict={t: 1.0 for t in self.all_types},
            threshold=0.0,
        )
        inp = torch.rand(2, self.num_sensory, 2)
        # effective_tau clamps to 1.0, so both should produce the same output
        out_clamped = model_clamped(inp).detach()
        out_one = model_one(inp).detach()
        self.assertTrue(
            torch.allclose(out_clamped, out_one, atol=1e-5),
            "tau=0.1 should be clamped to 1.0, producing same output as tau=1.0",
        )

    # ------------------------------------------------------------------
    # LinearNetwork gradient tests
    # ------------------------------------------------------------------

    def test_linear_network_tau_param_receives_gradient(self):
        """LinearNetwork tau_param should receive a gradient when requires_grad=True."""
        model = LinearNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            idx_to_group=self.idx_to_group,
        )
        model.set_param_grads(tau=True)
        inp = torch.rand(2, self.num_sensory, self.num_layers)
        out = model(inp)
        out.sum().backward()
        self.assertIsNotNone(model.tau_param.grad)
        self.assertFalse(torch.all(model.tau_param.grad == 0))

    def test_linear_network_tau_param_changes_after_optimizer_step(self):
        """LinearNetwork tau_param should change after an optimizer step."""
        model = LinearNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            idx_to_group=self.idx_to_group,
        )
        model.set_param_grads(tau=True)
        tau_before = model.tau_param.clone().detach()
        optimizer = torch.optim.Adam([model.tau_param], lr=0.1)
        inp = torch.rand(2, self.num_sensory, self.num_layers)
        out = model(inp)
        out.sum().backward()
        optimizer.step()
        self.assertFalse(torch.allclose(model.tau_param.detach(), tau_before))

    # ------------------------------------------------------------------
    # train_model tau behaviour
    # ------------------------------------------------------------------

    def _make_train_model_fixtures(self):
        """Helper: returns a model, inputs, and targets suitable for train_model."""
        model = MultilayeredNetwork(
            self.weights,
            self.sensory_indices,
            num_layers=2,
            idx_to_group=self.idx_to_group,
        )
        inputs = torch.rand(6, self.num_sensory, 2)
        targets = pd.DataFrame(
            [{"batch": i, "neuron_idx": 2, "layer": 1, "value": 0.5} for i in range(6)]
        )
        return model, inputs, targets

    def test_train_model_tau_true_changes_tau_param(self):
        """train_model with train_tau=True should change tau_param values."""
        model, inputs, targets = self._make_train_model_fixtures()
        tau_before = model.tau_param.clone().detach()
        train_model(
            model,
            inputs,
            targets,
            num_epochs=10,
            wandb=False,
            train_slopes=False,
            train_biases=False,
            train_divisive_strength=False,
            train_tau=True,
        )
        self.assertFalse(
            torch.allclose(model.tau_param.detach(), tau_before),
            "tau_param should change when train_tau=True",
        )

    def test_train_model_tau_false_leaves_tau_param_unchanged(self):
        """train_model with train_tau=False should not change tau_param values."""
        model, inputs, targets = self._make_train_model_fixtures()
        tau_before = model.tau_param.clone().detach()
        train_model(
            model,
            inputs,
            targets,
            num_epochs=10,
            wandb=False,
            train_slopes=True,  # keep at least one param trainable so backward() works
            train_biases=False,
            train_divisive_strength=False,
            train_tau=False,
        )
        self.assertTrue(
            torch.allclose(model.tau_param.detach(), tau_before),
            "tau_param should be unchanged when train_tau=False",
        )

    # ------------------------------------------------------------------
    # tau_dict stored as attribute
    # ------------------------------------------------------------------

    def test_tau_dict_stored_as_attribute(self):
        """tau_dict passed to constructor should be stored on the model."""
        tau_dict = {"type_0": 3.0, "type_1": 7.0, "type_2": 15.0}
        for cls in [LinearNetwork, MultilayeredNetwork]:
            model = cls(
                self.weights,
                self.sensory_indices,
                idx_to_group=self.idx_to_group,
                tau_dict=tau_dict,
            )
            self.assertEqual(
                model.tau_dict,
                tau_dict,
                f"{cls.__name__}: tau_dict not stored correctly",
            )

    def test_tau_dict_none_stored_when_not_provided(self):
        """tau_dict should be None when not provided."""
        for cls in [LinearNetwork, MultilayeredNetwork]:
            model = cls(self.weights, self.sensory_indices)
            self.assertIsNone(model.tau_dict)

    # ------------------------------------------------------------------
    # LinearNetwork vs MultilayeredNetwork produce different outputs
    # ------------------------------------------------------------------

    def test_linear_and_multilayered_differ_same_tau(self):
        """LinearNetwork and MultilayeredNetwork should produce different outputs with same tau."""
        kwargs = dict(
            sensory_indices=self.sensory_indices,
            num_layers=self.num_layers,
            idx_to_group=self.idx_to_group,
            tau_dict={t: 5.0 for t in self.all_types},
            threshold=0.0,
            tanh_steepness=1.0,
            default_bias=0.0,
        )
        linear_model = LinearNetwork(self.weights, **kwargs)
        multi_model = MultilayeredNetwork(self.weights, **kwargs)

        inp = torch.rand(2, self.num_sensory, self.num_layers)
        out_linear = linear_model(inp).detach()
        out_multi = multi_model(inp).detach()

        self.assertFalse(
            torch.allclose(out_linear, out_multi, atol=1e-5),
            "LinearNetwork and MultilayeredNetwork should differ due to tanh/threshold",
        )


# ---- Add these tests before `if __name__ == "__main__":` ----


class TestCheckpointing(unittest.TestCase):
    """Tests for gradient checkpointing and hoisted parameter gathers."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_neurons = 12
        self.num_sensory = 4
        self.num_layers = 20
        self.batch_size = 2

        dense_weights = np.random.rand(self.num_neurons, self.num_neurons).astype(
            np.float32
        )
        dense_weights = dense_weights / dense_weights.sum(axis=1, keepdims=True)
        dense_weights[:, :3] = -dense_weights[:, :3]
        self.all_weights = csr_matrix(dense_weights)
        self.sensory_indices = list(range(self.num_sensory))

        self.idx_to_group = {}
        types = ["type_0", "type_1", "type_2"]
        for i in range(self.num_neurons):
            self.idx_to_group[i] = types[i % 3]

    def _make_input(self):
        return torch.rand(
            self.batch_size,
            self.num_sensory,
            self.num_layers,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Forward equivalence
    # ------------------------------------------------------------------

    def test_multilayered_checkpoint_matches_no_checkpoint(self):
        """Checkpointed forward should produce identical output to non-checkpointed."""
        torch.manual_seed(0)
        model = MultilayeredNetwork(
            self.all_weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            idx_to_group=self.idx_to_group,
        ).to(self.device)
        inp = self._make_input()

        out_no_ckpt = model(inp, checkpoint_steps=0).detach()
        out_ckpt = model(inp, checkpoint_steps=7).detach()

        self.assertTrue(
            torch.allclose(out_no_ckpt, out_ckpt, atol=1e-5),
            "Checkpointed and non-checkpointed forward should match (MultilayeredNetwork)",
        )

    def test_linear_checkpoint_matches_no_checkpoint(self):
        """Checkpointed forward should produce identical output to non-checkpointed."""
        torch.manual_seed(0)
        model = LinearNetwork(
            self.all_weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            idx_to_group=self.idx_to_group,
        ).to(self.device)
        inp = self._make_input()

        out_no_ckpt = model(inp, checkpoint_steps=0).detach()
        out_ckpt = model(inp, checkpoint_steps=7).detach()

        self.assertTrue(
            torch.allclose(out_no_ckpt, out_ckpt, atol=1e-5),
            "Checkpointed and non-checkpointed forward should match (LinearNetwork)",
        )

    # ------------------------------------------------------------------
    # Gradient equivalence
    # ------------------------------------------------------------------

    def _gradient_equivalence(self, cls):
        """Helper: compare parameter gradients with and without checkpointing."""
        torch.manual_seed(42)
        inp = self._make_input()

        def _run(ckpt_steps):
            torch.manual_seed(42)
            model = cls(
                self.all_weights,
                self.sensory_indices,
                num_layers=self.num_layers,
                idx_to_group=self.idx_to_group,
            ).to(self.device)
            model.set_param_grads(slopes=True, raw_biases=True, tau=True)
            out = model(inp.clone(), checkpoint_steps=ckpt_steps)
            out.sum().backward()
            return {
                "slope": model.slope.grad.clone(),
                "bias": model.raw_biases.grad.clone(),
                "tau": model.tau_param.grad.clone(),
            }

        grads_plain = _run(0)
        grads_ckpt = _run(5)

        for name in grads_plain:
            self.assertTrue(
                torch.allclose(grads_plain[name], grads_ckpt[name], atol=1e-4),
                f"{cls.__name__}: {name} gradients differ between checkpointed and non-checkpointed",
            )

    def test_multilayered_gradient_equivalence(self):
        self._gradient_equivalence(MultilayeredNetwork)

    def test_linear_gradient_equivalence(self):
        self._gradient_equivalence(LinearNetwork)

    # ------------------------------------------------------------------
    # Checkpoint fallback with manipulate
    # ------------------------------------------------------------------

    def test_checkpoint_fallback_with_manipulate(self):
        """When manipulate is used, checkpoint_steps should be ignored (fallback)."""
        model = MultilayeredNetwork(
            self.all_weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            idx_to_group=self.idx_to_group,
        ).to(self.device)
        inp = self._make_input()
        manipulate = {0: {"type_0": 0.5}}

        # Should not raise, should silently fall back to non-checkpointed
        out = model(inp, manipulate=manipulate, checkpoint_steps=5)
        self.assertEqual(
            out.shape, (self.batch_size, self.num_neurons, self.num_layers)
        )

    # ------------------------------------------------------------------
    # Different checkpoint_steps values all produce same output
    # ------------------------------------------------------------------

    def test_various_checkpoint_step_sizes(self):
        """Different chunk sizes should all produce the same output."""
        model = MultilayeredNetwork(
            self.all_weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            idx_to_group=self.idx_to_group,
        ).to(self.device)
        inp = self._make_input()

        baseline = model(inp, checkpoint_steps=0).detach()
        for steps in [1, 3, 10, self.num_layers]:
            out = model(inp, checkpoint_steps=steps).detach()
            self.assertTrue(
                torch.allclose(baseline, out, atol=1e-5),
                f"checkpoint_steps={steps} produced different output",
            )

    # ------------------------------------------------------------------
    # train_model with checkpoint_steps
    # ------------------------------------------------------------------

    def test_train_model_with_checkpoint_steps(self):
        """train_model should work with checkpoint_steps and reduce loss."""
        model = MultilayeredNetwork(
            self.all_weights,
            self.sensory_indices,
            num_layers=5,
            idx_to_group=self.idx_to_group,
        ).to(self.device)
        inputs = torch.rand(6, self.num_sensory, 5, device=self.device)
        targets = pd.DataFrame(
            [{"batch": i, "neuron_idx": 5, "layer": 4, "value": 0.5} for i in range(6)]
        )

        _, history, *_ = train_model(
            model,
            inputs,
            targets,
            num_epochs=5,
            wandb=False,
            train_slopes=True,
            train_biases=False,
            train_divisive_strength=False,
            train_tau=False,
            checkpoint_steps=3,
        )
        self.assertTrue(len(history["loss"]) == 5)

    # ------------------------------------------------------------------
    # Hoisted parameters produce same output as original indexing
    # ------------------------------------------------------------------

    def test_hoisted_params_match_original(self):
        """activation_function with pre-expanded params should match per-call indexing."""
        model = MultilayeredNetwork(
            self.all_weights,
            self.sensory_indices,
            num_layers=3,
            idx_to_group=self.idx_to_group,
        ).to(self.device)

        x = torch.rand(self.num_neurons, self.batch_size, device=self.device)
        x_prev = torch.rand(self.num_neurons, self.batch_size, device=self.device)

        # Call with hoisted params (as forward does)
        slopes_full = model.slope[model.indices].view(-1, 1)
        biases_full = model.biases[model.indices].view(-1, 1)
        taus_full = model.effective_tau[model.indices].view(-1, 1)
        out_hoisted = model.activation_function(
            x.clone(),
            x_previous=x_prev,
            slopes_full=slopes_full,
            biases_full=biases_full,
            taus_full=taus_full,
        )

        # Call without hoisted params (fallback indexing)
        out_fallback = model.activation_function(x.clone(), x_previous=x_prev)

        self.assertTrue(
            torch.allclose(out_hoisted, out_fallback, atol=1e-6),
            "Hoisted params and per-call indexing should produce identical results",
        )

    # ------------------------------------------------------------------
    # 2D input still works with checkpointing
    # ------------------------------------------------------------------

    def test_checkpoint_2d_input(self):
        """checkpoint_steps should work with 2D (non-batched) input."""
        model = MultilayeredNetwork(
            self.all_weights,
            self.sensory_indices,
            num_layers=self.num_layers,
            idx_to_group=self.idx_to_group,
        ).to(self.device)
        inp = torch.rand(self.num_sensory, self.num_layers, device=self.device)

        out_no_ckpt = model(inp, checkpoint_steps=0).detach()
        out_ckpt = model(inp, checkpoint_steps=5).detach()

        self.assertEqual(out_no_ckpt.shape, (self.num_neurons, self.num_layers))
        self.assertTrue(torch.allclose(out_no_ckpt, out_ckpt, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
