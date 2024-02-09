import unittest
import torch

from nn.network_config import NetConfig
from nn.policy_value.skeleton import BasePolicyValueNet
from nn.policy_value.fully_connected import FCPolicyValueNet
from env.env import N_STATE_CHANNELS, N_ACTIONS

BASE_POLICY_VALUE_CONFIG_FILE_PATH = './base_policy_value_config.yaml'
FC_POLICY_VALUE_CONFIG_FILE_PATH = './fc_policy_value_config.yaml'


class TestPolicyValueNet(unittest.TestCase):
    def setUp(self) -> None:
        self.net_config = NetConfig(config_path=BASE_POLICY_VALUE_CONFIG_FILE_PATH)

    def test_skeleton_attributes(self):
        net = BasePolicyValueNet(config=self.net_config)
        self.assertTrue(net.regularization)
        self.assertEqual(net.regularization_type, "entropy")
        self.assertEqual(net.regularization_coef, 0.0005)
        self.assertTrue(type(net.regularization_loss) == torch.nn.KLDivLoss)

        self.assertEqual(net.actor_coef, 0.8)
        self.assertEqual(net.critic_coef, 0.1)

        self.assertTrue(type(net.actor_loss) == torch.nn.CrossEntropyLoss)
        self.assertTrue(type(net.critic_loss) == torch.nn.MSELoss)

    def test_fc_attributes(self):
        net_config = NetConfig(config_path=FC_POLICY_VALUE_CONFIG_FILE_PATH)
        net = FCPolicyValueNet(config=net_config)
        self.assertTrue(hasattr(net, "state_embeddings"))
        self.assertTrue(hasattr(net, "policy_head"))
        self.assertTrue(hasattr(net, "value_head"))

        n_layers = 2
        for i in range(1, n_layers + 1):
            self.assertTrue(hasattr(net.state_embeddings, f"linear{i}"))
            self.assertTrue(hasattr(net.state_embeddings, f"{net.activation}{i}"))
            if i < n_layers:
                self.assertTrue(hasattr(net.policy_head, f"linear{i}"))
                self.assertTrue(hasattr(net.value_head, f"linear{i}"))
                self.assertTrue(hasattr(net.policy_head, f"{net.activation}{i}"))
                self.assertTrue(hasattr(net.value_head, f"{net.activation}{i}"))
            else:
                self.assertTrue(hasattr(net.policy_head, f"output"))
                self.assertTrue(hasattr(net.value_head, f"output"))

    def test_fc_outputs(self):
        net_config = NetConfig(config_path=FC_POLICY_VALUE_CONFIG_FILE_PATH)
        net = FCPolicyValueNet(config=net_config)
        batch_size = 64

        with torch.no_grad():
            x = torch.randn(size=(batch_size, 7, 7, N_STATE_CHANNELS))
            policies, values = net(x)
            self.assertSequenceEqual(policies.shape, (batch_size, N_ACTIONS))
            self.assertSequenceEqual(values.shape, (batch_size, 1))


if __name__ == '__main__':
    unittest.main()
