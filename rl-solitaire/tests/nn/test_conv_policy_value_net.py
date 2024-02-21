import unittest
import torch

from nn.network_config import NetConfig
from nn.policy_value.conv import ConvPolicyValueNet
from env.env import N_STATE_CHANNELS, N_ACTIONS

BASE_POLICY_VALUE_CONFIG_FILE_PATH = './base_policy_value_config.yaml'
CONV_POLICY_VALUE_CONFIG_FILE_PATH = './conv_policy_value_config.yaml'


class TestConvPolicyValueNet(unittest.TestCase):
    def setUp(self) -> None:
        self.conv_net_config = NetConfig(config_path=CONV_POLICY_VALUE_CONFIG_FILE_PATH)

    def test_convnet_attributes(self):
        net = ConvPolicyValueNet(config=self.conv_net_config)
        self.assertTrue(hasattr(net, "state_embeddings"))
        self.assertTrue(hasattr(net, "policy_head"))
        self.assertTrue(hasattr(net, "value_head"))

        # state embeddings
        self.assertTrue(hasattr(net.state_embeddings, "input_conv"))
        self.assertTrue(hasattr(net.state_embeddings, "residual_blocks"))

        self.assertEqual(net.state_embeddings.input_conv.in_channels, N_STATE_CHANNELS)
        self.assertEqual(net.state_embeddings.input_conv.out_channels,
                         self.conv_net_config.architecture_config["embeddings"]["hidden_dim"])

        residual_blocks = net.state_embeddings.residual_blocks
        self.test_residual_blocks(residual_blocks, self.conv_net_config.architecture_config["embeddings"],
                                  input_dim_key="hidden_dim", hidden_dim_key="residual_hidden_dim")

        # policy head
        self.assertTrue(hasattr(net.policy_head, "residual_blocks"))
        residual_blocks = net.policy_head.residual_blocks
        self.test_residual_blocks(residual_blocks, self.conv_net_config.architecture_config["policy_head"],
                                  input_dim_key="input_dim", hidden_dim_key="hidden_dim")
        self.assertTrue(hasattr(net.policy_head, "output_linear"))

        # value head
        self.assertTrue(hasattr(net.value_head, "residual_blocks"))
        residual_blocks = net.value_head.residual_blocks
        self.test_residual_blocks(residual_blocks, self.conv_net_config.architecture_config["value_head"],
                                  input_dim_key="input_dim", hidden_dim_key="hidden_dim")
        self.assertTrue(hasattr(net.value_head, "output_linear"))

    def test_residual_blocks(self, residual_blocks, config_dict, input_dim_key, hidden_dim_key):
        for i in range(1, config_dict["n_residual_blocks"] + 1):
            self.assertTrue(hasattr(residual_blocks, f"batchnorm{i}"))
            self.assertTrue(hasattr(residual_blocks, f"residual{i}"))

            residual_block = getattr(residual_blocks, f"residual{i}")
            self.assertTrue(hasattr(residual_block, "layers"))
            residual_block_layers = residual_block.layers
            self.assertTrue(hasattr(residual_block_layers, "conv1"))

            self.assertEqual(residual_block_layers.conv1.in_channels,
                             config_dict[input_dim_key])
            self.assertEqual(residual_block_layers.conv1.out_channels,
                             config_dict[hidden_dim_key])

            for j in range(1, residual_block.n_layers - 1):
                self.assertTrue(hasattr(residual_block_layers, f"conv{j + 1}"))
                conv_layer = getattr(residual_block_layers, f"conv{j + 1}")
                self.assertEqual(conv_layer.in_channels,
                                 config_dict[input_dim_key])
                self.assertEqual(conv_layer.out_channels,
                                 config_dict[hidden_dim_key])

            n_layers = residual_block.n_layers
            self.assertTrue(hasattr(residual_block_layers, f"conv{n_layers}"))
            conv_layer = getattr(residual_block_layers, f"conv{n_layers}")
            self.assertEqual(conv_layer.in_channels,
                             config_dict[hidden_dim_key])
            self.assertEqual(conv_layer.out_channels,
                             config_dict[input_dim_key])

    def test_convnet_output(self):
        net = ConvPolicyValueNet(config=self.conv_net_config)
        batch_size = 64
        x = torch.randn(batch_size, 7, 7, 3)
        with torch.no_grad():
            policies, values = net(x)
        self.assertSequenceEqual(policies.shape, (batch_size, N_ACTIONS))
        self.assertSequenceEqual(values.shape, (batch_size, 1))

    def test_convnet_inner_activations(self):
        net = ConvPolicyValueNet(config=self.conv_net_config)
        hidden_dim = self.conv_net_config.architecture_config["embeddings"]["hidden_dim"]
        # residual_hidden_dim = self.conv_net_config.architecture_config["embeddings"]["residual_hidden_dim"]
        batch_size = 64
        x = torch.randn(batch_size, 7, 7, 3)
        x = net._reshape_2d_input(x)
        self.assertSequenceEqual(x.shape, (batch_size, 3, 7, 7))
        with torch.no_grad():
            h = net.state_embeddings.input_conv(x)
            self.assertSequenceEqual(h.shape, (batch_size, hidden_dim, 7, 7))

            h = net.state_embeddings.residual_blocks.batchnorm1(h)
            self.assertSequenceEqual(h.shape, (batch_size, hidden_dim, 7, 7))

            h = net.state_embeddings.residual_blocks.residual1(h)
            self.assertSequenceEqual(h.shape, (batch_size, hidden_dim, 7, 7))

            h_state = net.state_embeddings(x)
            self.assertSequenceEqual(h_state.shape, (batch_size, hidden_dim, 7, 7))

            h = net.policy_head.residual_blocks.batchnorm1(h_state)
            self.assertSequenceEqual(h.shape, (batch_size, hidden_dim, 7, 7))

            h = net.policy_head.residual_blocks.residual1(h)
            self.assertSequenceEqual(h.shape, (batch_size, hidden_dim, 7, 7))

            h = net.value_head.residual_blocks.batchnorm1(h_state)
            self.assertSequenceEqual(h.shape, (batch_size, hidden_dim, 7, 7))

            h = net.value_head.residual_blocks.residual1(h)
            self.assertSequenceEqual(h.shape, (batch_size, hidden_dim, 7, 7))


if __name__ == '__main__':
    unittest.main()
