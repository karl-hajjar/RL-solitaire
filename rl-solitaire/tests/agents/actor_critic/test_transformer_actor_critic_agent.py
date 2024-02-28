import unittest
import os
import numpy as np
import torch

from nn.network_config import NetConfig
from nn.policy_value.transformer import TransformerPolicyValueNet
from agents.actor_critic.actor_critic_agent import ActorCriticAgent
from env.env import Env, N_STATE_CHANNELS, N_ACTIONS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
FC_POLICY_VALUE_CONFIG_FILE_PATH = os.path.join(ROOT, "nn/policy_value/transformer_policy_value_config.yaml")


class TestTransformerActorCriticAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.net_config = NetConfig(config_path=FC_POLICY_VALUE_CONFIG_FILE_PATH)
        self.network = TransformerPolicyValueNet(config=self.net_config)
        self.agent = ActorCriticAgent(network=self.network, name="FCActorCriticAgent")

    def test_collect_data(self):
        env = Env()
        T = 10
        with torch.no_grad():
            data = self.agent.collect_data(env, T=T)
        for key in ["states", "actions", "action_masks", "advantages", "value_targets"]:
            self.assertIn(key, data.keys())
            self.assertTrue(type(data[key]) == np.ndarray)

        T = len(data["states"])
        self.assertSequenceEqual(data["states"].shape, (T, 7, 7, N_STATE_CHANNELS))
        self.assertSequenceEqual(data["actions"].shape, (T,))
        self.assertSequenceEqual(data["action_masks"].shape, (T, N_ACTIONS))
        self.assertSequenceEqual(data["advantages"].shape, (T, 1))
        self.assertSequenceEqual(data["value_targets"].shape, (T, 1))

        self.assertTrue(data["states"].dtype == np.float32)
        self.assertTrue(data["actions"].dtype == int)
        self.assertTrue(data["action_masks"].dtype == np.float32)
        self.assertTrue(data["advantages"].dtype == np.float32)
        self.assertTrue(data["value_targets"].dtype == np.float32)


if __name__ == '__main__':
    unittest.main()
