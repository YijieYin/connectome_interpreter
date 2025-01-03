import unittest
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from connectome_interpreter.activation_maximisation import (
    MultilayeredNetwork,
    activation_maximisation,
    activations_to_df,
    activations_to_df_batched,
    TargetActivation,
    get_neuron_activation
)


class TestMultilayeredNetwork(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.num_neurons = 10
        self.num_sensory = 4
        self.num_layers = 3
        self.batch_size = 2

        self.all_weights = torch.rand(
            self.num_neurons, self.num_neurons).to(self.device)
        self.all_weights = self.all_weights / \
            self.all_weights.sum(dim=1, keepdim=True)
        self.all_weights[:, :3] = -self.all_weights[:, :3]
        self.sensory_indices = list(range(self.num_sensory))

        self.model = MultilayeredNetwork(
            self.all_weights,
            self.sensory_indices,
            num_layers=self.num_layers
        ).to(self.device)

    def test_initialization(self):
        self.assertEqual(self.model.num_layers, self.num_layers)
        self.assertEqual(len(self.model.sensory_indices), self.num_sensory)
        self.assertTrue(torch.equal(self.model.all_weights, self.all_weights))

    def test_forward_pass_2d(self):
        input_tensor = torch.rand(
            self.num_sensory, self.num_layers).to(self.device)
        output = self.model(input_tensor)

        expected_shape = (self.num_neurons, self.num_layers)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.all(output >= -1) and torch.all(output <= 1))

    def test_forward_pass_3d(self):
        input_tensor = torch.rand(
            self.batch_size, self.num_sensory, self.num_layers).to(self.device)
        output = self.model(input_tensor)

        expected_shape = (self.batch_size, self.num_neurons, self.num_layers)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.all(output >= -1) and torch.all(output <= 1))


class TestTargetActivation(unittest.TestCase):
    def setUp(self):
        self.dict_targets = {
            0: {1: 0.5, 2: 0.8},
            1: {0: 0.3}
        }
        self.df_targets = pd.DataFrame([
            {'batch': 0, 'layer': 0, 'neuron': 1, 'value': 0.5},
            {'batch': 0, 'layer': 0, 'neuron': 2, 'value': 0.8},
            {'batch': 1, 'layer': 1, 'neuron': 0, 'value': 0.3}
        ])

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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultilayeredNetwork(
            torch.rand(10, 10).to(self.device),
            sensory_indices=[0, 1, 2, 3],
            num_layers=3
        ).to(self.device)

        self.targets = TargetActivation({
            0: {0: 0.5, 1: 0.8},
            1: {2: 0.3}
        }, batch_size=2)

    def test_basic_optimization(self):
        result = activation_maximisation(
            self.model,
            self.targets,
            num_iterations=10,
            in_reg_lambda=1e-3,
            out_reg_lambda=1e-3,
            wandb=False,
            device=self.device
        )

        input_tensor, output, act_losses, *_ = result
        expected_shape = (2, 4, 3)  # (batch_size, num_sensory, num_layers)
        self.assertEqual(input_tensor.shape, expected_shape)
        self.assertTrue(act_losses[-1] <= act_losses[0])

    def test_custom_regularization(self):
        custom_reg = {'in': lambda x: torch.sum(torch.abs(x)),
                      'out': lambda x: torch.sum(torch.abs(x))}

        result = activation_maximisation(
            self.model,
            self.targets,
            custom_reg_functions=custom_reg,
            num_iterations=10,
            wandb=False,
            device=self.device
        )

        _, _, _, out_reg_losses, in_reg_losses, _ = result
        self.assertTrue(len(in_reg_losses) > 0 and len(out_reg_losses) > 0)


class TestActivationsToDF(unittest.TestCase):
    def setUp(self):
        self.weights = np.array([
            [0.5, 0.3, 0.0],
            [0.0, 0.4, 0.2],
            [0.0, 0.0, 0.6]
        ])
        self.input_act = np.array([[0.8, 0.6], [0.7, 0.5]])
        self.output_act = np.array([[0.8, 0.6], [0.7, 0.5], [0.6, 0.4]])
        self.sensory_indices = [0, 1]

    def test_basic_functionality(self):
        paths = activations_to_df(
            self.weights,
            self.input_act,
            self.output_act,
            self.sensory_indices
        )

        expected_columns = ['pre', 'post', 'weight',
                            'layer', 'pre_activation', 'post_activation']
        self.assertTrue(all(col in paths.columns for col in expected_columns))
        self.assertEqual(paths['layer'].nunique(), self.output_act.shape[1])

    def test_sparse_input(self):
        sparse_weights = csr_matrix(self.weights)

        paths_dense = activations_to_df(
            self.weights, self.input_act, self.output_act, self.sensory_indices)
        paths_sparse = activations_to_df(
            sparse_weights, self.input_act, self.output_act, self.sensory_indices)

        pd.testing.assert_frame_equal(paths_dense, paths_sparse)


class TestActivationsToDFBatched(unittest.TestCase):
    def setUp(self):
        self.weights = np.array([
            [0.5, 0.3, 0.0],
            [0.0, 0.4, 0.2],
            [0.0, 0.0, 0.6]
        ])
        self.batched_input = np.array([
            [[0.8, 0.6], [0.7, 0.5]],
            [[0.6, 0.4], [0.5, 0.3]]
        ])
        self.batched_output = np.array([
            [[0.8, 0.6], [0.7, 0.5], [0.6, 0.4]],
            [[0.6, 0.4], [0.5, 0.3], [0.4, 0.2]]
        ])
        self.sensory_indices = [0, 1]

    def test_batched_processing(self):
        paths = activations_to_df_batched(
            self.weights,
            self.batched_input,
            self.batched_output,
            self.sensory_indices
        )

        self.assertTrue('batch' in paths.columns)
        self.assertEqual(paths['batch'].nunique(), self.batched_input.shape[0])


class TestGetNeuronActivation(unittest.TestCase):
    def setUp(self):
        # Shape: (batch_size=3, num_neurons=4, num_timesteps=5)
        self.output = torch.rand(3, 4, 5)
        self.batch_names = ['batch1', 'batch2', 'batch3']
        self.neuron_indices = [10, 11, 12, 13]
        self.idx_to_group = {10: 'group1',
                             11: 'group1', 12: 'group2', 13: 'group2'}

    def test_valid_input_with_groups(self):
        df = get_neuron_activation(
            self.output,
            self.batch_names,
            self.neuron_indices,
            self.idx_to_group
        )
        self.assertEqual(df.shape, (12, 8))  # 3 batches * 4 neurons, 8 columns
        self.assertListEqual(df['batch_name'].tolist()[:4], ['batch1'] * 4)
        self.assertListEqual(df['idx'].tolist()[:4], [10, 11, 12, 13])
        self.assertListEqual(df['group'].tolist()[:4], [
                             'group1', 'group1', 'group2', 'group2'])

    def test_valid_input_without_groups(self):
        df = get_neuron_activation(
            self.output,
            self.batch_names,
            self.neuron_indices
        )
        self.assertEqual(df.shape, (12, 7))  # No 'group' column
        self.assertNotIn('group', df.columns)

    def test_mismatched_batch_names_length(self):
        with self.assertRaises(ValueError) as context:
            get_neuron_activation(
                self.output,
                ['batch1', 'batch2'],  # Incorrect length
                self.neuron_indices
            )
        self.assertIn('Length of batch_names has to be the same as output.shape[0]', str(
            context.exception))

    def test_mismatched_neuron_indices_length(self):
        with self.assertRaises(ValueError) as context:
            get_neuron_activation(
                self.output,
                self.batch_names,
                [10, 11]  # Incorrect length
            )
        self.assertIn('Length of neuron_indices has to be the same as output.shape[1]', str(
            context.exception))

    def test_idx_to_group_key_error(self):
        with self.assertRaises(KeyError):
            get_neuron_activation(
                self.output,
                self.batch_names,
                [14, 15, 16, 17],  # Indices not in idx_to_group
                self.idx_to_group
            )


if __name__ == '__main__':
    unittest.main()