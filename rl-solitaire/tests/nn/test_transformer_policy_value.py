import unittest

import torch

from env.env import N_STATE_CHANNELS, N_ACTIONS
from nn.network_config import NetConfig
from nn.policy_value.transformer import TransformerPolicyValueNet

BASE_POLICY_VALUE_CONFIG_FILE_PATH = './base_policy_value_config.yaml'
POLICY_VALUE_NET_CONFIG_FILE_PATH = './transformer_policy_value_config.yaml'


class TestTransformerPolicyValueNet(unittest.TestCase):
    def setUp(self) -> None:
        self.net_config = NetConfig(config_path=BASE_POLICY_VALUE_CONFIG_FILE_PATH)

    def test_transformer_attributes(self):
        net_config = NetConfig(config_path=POLICY_VALUE_NET_CONFIG_FILE_PATH)
        net = TransformerPolicyValueNet(config=net_config)
        self.assertTrue(hasattr(net, "state_embeddings"))
        self.assertTrue(hasattr(net, "policy_head"))
        self.assertTrue(hasattr(net, "value_head"))

        self.assertEqual(type(net.state_embeddings), torch.nn.Module)
        self.assertEqual(type(net.policy_head), torch.nn.Module)
        self.assertEqual(type(net.value_head), torch.nn.Module)

        self.assertTrue(hasattr(net.state_embeddings, "input_embedding"))
        self.assertTrue(hasattr(net.state_embeddings, "positional_encoder"))
        self.assertTrue(hasattr(net.state_embeddings, "transformer_encoder"))
        self.assertTrue(hasattr(net.state_embeddings, "transformer_encoder_layer"))

        self.assertTrue(hasattr(net.policy_head, "input_linear"))
        self.assertTrue(hasattr(net.policy_head, "transformer_encoder"))
        self.assertTrue(hasattr(net.policy_head, "transformer_encoder_layer"))
        self.assertTrue(hasattr(net.policy_head, "output_linear"))

        self.assertTrue(hasattr(net.value_head, "input_linear"))
        self.assertTrue(hasattr(net.value_head, "transformer_encoder"))
        self.assertTrue(hasattr(net.value_head, "transformer_encoder_layer"))
        self.assertTrue(hasattr(net.value_head, "output_linear"))

    def test_src_mask(self):
        net_config = NetConfig(config_path=POLICY_VALUE_NET_CONFIG_FILE_PATH)
        net = TransformerPolicyValueNet(config=net_config)

        self.assertTrue(hasattr(net, "src_mask"))
        self.assertEqual(type(net.src_mask), torch.Tensor)
        self.assertSequenceEqual(net.src_mask.shape, (49, 49))

    def test_state_embeddings_logic(self):
        net_config = NetConfig(config_path=POLICY_VALUE_NET_CONFIG_FILE_PATH)
        net = TransformerPolicyValueNet(config=net_config)
        batch_size = 64
        state_embeddings_config_dict = net_config.architecture_config["embeddings"]

        with torch.no_grad():
            x = torch.randn(size=(batch_size, 7, 7, N_STATE_CHANNELS))

            # get_state_embeddings_logic
            x = net.reformat_input(x)
            self.assertSequenceEqual(x.shape, (49, batch_size, N_STATE_CHANNELS))
            x = net.state_embeddings.input_embedding(x)
            self.assertSequenceEqual(x.shape, (49, batch_size, state_embeddings_config_dict["hidden_dim"]))
            x = net.state_embeddings.positional_encoder(x)  # returns x + positional_encodings
            self.assertSequenceEqual(x.shape, (49, batch_size, state_embeddings_config_dict["hidden_dim"]))
            x = net.state_embeddings.transformer_encoder(src=x, mask=net.src_mask, is_causal=False)
            self.assertSequenceEqual(x.shape, (49, batch_size, state_embeddings_config_dict["hidden_dim"]))

    def test_policy_head_logic(self):
        net_config = NetConfig(config_path=POLICY_VALUE_NET_CONFIG_FILE_PATH)
        net = TransformerPolicyValueNet(config=net_config)
        batch_size = 64
        policy_head_config_dict = net_config.architecture_config["policy_head"]

        with torch.no_grad():
            x = torch.randn(size=(batch_size, 7, 7, N_STATE_CHANNELS))

            state_embeddings = net.get_state_embeddings(x)
            self.assertSequenceEqual(state_embeddings.shape, (49, batch_size,
                                                              policy_head_config_dict["input_dim"]))

            x = net.policy_head.input_linear(state_embeddings)
            self.assertSequenceEqual(x.shape, (49, batch_size, policy_head_config_dict["hidden_dim"]))
            x = net.policy_head.transformer_encoder(src=x, mask=net.src_mask, is_causal=False)
            self.assertSequenceEqual(x.shape, (49, batch_size, policy_head_config_dict["hidden_dim"]))
            x = net.policy_head.output_linear(x)  # shape (49, N, 4)
            self.assertSequenceEqual(x.shape, (49, batch_size, 4))
            x = x[~net.src_mask[0, :], :, :].reshape(x.shape[1], -1)  # shape (N, 33*4) = (N, 132)
            self.assertSequenceEqual(x.shape, (batch_size, N_ACTIONS))

            x = net.get_policy_from_state_embeddings(state_embeddings)
            self.assertSequenceEqual(x.shape, (batch_size, N_ACTIONS))

    def test_value_head_logic(self):
        net_config = NetConfig(config_path=POLICY_VALUE_NET_CONFIG_FILE_PATH)
        net = TransformerPolicyValueNet(config=net_config)
        batch_size = 64
        value_head_config_dict = net_config.architecture_config["value_head"]

        with torch.no_grad():
            x = torch.randn(size=(batch_size, 7, 7, N_STATE_CHANNELS))

            state_embeddings = net.get_state_embeddings(x)
            self.assertSequenceEqual(state_embeddings.shape,
                                     (49, batch_size, value_head_config_dict["input_dim"]))

            x = net.value_head.input_linear(state_embeddings)
            self.assertSequenceEqual(x.shape, (49, batch_size, value_head_config_dict["hidden_dim"]))
            x = net.value_head.transformer_encoder(src=x, mask=net.src_mask, is_causal=False)
            self.assertSequenceEqual(x.shape, (49, batch_size, value_head_config_dict["hidden_dim"]))
            x = x[~net.src_mask[0, :], :, :].reshape(x.shape[1], -1)  # shape (N, 33 * value_head_hidden_dim)
            self.assertSequenceEqual(x.shape, (batch_size, 33*value_head_config_dict["hidden_dim"]))
            x = net.value_head.output_linear(x)
            self.assertSequenceEqual(x.shape, (batch_size, 1))

    def test_transformer_outputs(self):
        net_config = NetConfig(config_path=POLICY_VALUE_NET_CONFIG_FILE_PATH)
        net = TransformerPolicyValueNet(config=net_config)
        batch_size = 64

        with torch.no_grad():
            x = torch.randn(size=(batch_size, 7, 7, N_STATE_CHANNELS))
            policies, values = net(x)
            self.assertSequenceEqual(policies.shape, (batch_size, N_ACTIONS))
            self.assertSequenceEqual(values.shape, (batch_size, 1))

    def test_attention_weights_with_src_mask(self):
        net_config = NetConfig(config_path=POLICY_VALUE_NET_CONFIG_FILE_PATH)
        net = TransformerPolicyValueNet(config=net_config)
        batch_size = 64

        with torch.no_grad():
            x_ = torch.randn(size=(batch_size, 7, 7, N_STATE_CHANNELS))
            x = net.reformat_input(x_)
            x = net.state_embeddings.input_embedding(x)
            x = net.state_embeddings.positional_encoder(x)  # returns x + positional_encodings
            for layer in net.state_embeddings.transformer_encoder.layers:
                attn_output, attn_weights = layer.self_attn.forward(x, x, x,
                                                                    attn_mask=net.src_mask,
                                                                    key_padding_mask=None,
                                                                    need_weights=True,
                                                                    is_causal=True)
                self._check_attn_weights(attn_weights)
                x = layer(x)

            x = net.get_state_embeddings(x_)
            h = x
            for layer in net.policy_head.transformer_encoder.layers:
                attn_output, attn_weights = layer.self_attn.forward(h, h, h,
                                                                    attn_mask=net.src_mask,
                                                                    key_padding_mask=None,
                                                                    need_weights=True,
                                                                    is_causal=True)
                self._check_attn_weights(attn_weights)
                h = layer(h)

            h = x
            for layer in net.value_head.transformer_encoder.layers:
                attn_output, attn_weights = layer.self_attn.forward(h, h, h,
                                                                    attn_mask=net.src_mask,
                                                                    key_padding_mask=None,
                                                                    need_weights=True,
                                                                    is_causal=True)
                self._check_attn_weights(attn_weights)
                h = layer(h)

    def _check_attn_weights(self, attn_weights):
        attn_weights.reshape(attn_weights.shape[0], attn_weights.shape[1], 7, 7)
        zero_indices = [(0, 0), (0, 1), (0, 5), (0, 6)] + \
                       [(1, 0), (1, 1), (1, 5), (1, 6)] + \
                       [(5, 0), (5, 1), (5, 5), (5, 6)] + \
                       [(6, 0), (6, 1), (6, 5), (6, 6)]
        zero_indices = set(zero_indices)
        for i in range(7):
            for j in range(7):
                if (i, j) in zero_indices:
                    self.assertTrue((attn_weights[:, i, j] == 0).all())
                else:
                    # self.assertTrue((attn_weights[:, i, j] > 1e-15).all())
                    pass

    def test_get_policy_value(self):
        net_config = NetConfig(config_path=POLICY_VALUE_NET_CONFIG_FILE_PATH)
        net = TransformerPolicyValueNet(config=net_config)
        batch_size = 64

        with torch.no_grad():
            x = torch.randn(size=(batch_size, 7, 7, N_STATE_CHANNELS))
            policies = net.get_policy(x)
            values = net.get_value(x)
            self.assertSequenceEqual(policies.shape, (batch_size, N_ACTIONS))
            self.assertSequenceEqual(values.shape, (batch_size, 1))


if __name__ == '__main__':
    unittest.main()
