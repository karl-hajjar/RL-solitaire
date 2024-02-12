import unittest
import os
import numpy as np
import torch

from nn.network_config import NetConfig
from nn.policy_value.fully_connected import FCPolicyValueNet
from agents.actor_critic.actor_critic import ActorCriticAgent
from env.env import Env

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
FC_POLICY_VALUE_CONFIG_FILE_PATH = os.path.join(ROOT, "nn/policy_value/fc_policy_value_config.yaml")


class TestFcActorCriticAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.net_config = NetConfig(config_path=FC_POLICY_VALUE_CONFIG_FILE_PATH)
        self.network = FCPolicyValueNet(config=self.net_config)
        self.agent = ActorCriticAgent(network=self.network, name="FCActorCriticAgent")

    def test_collect_data(self):
        env = Env()
        T = 1
        with torch.no_grad():
            data = self.agent.collect_data(env, T=T)
        for key in ["states", "actions", "action_masks", "advantages", "value_targets"]:
            self.assertIn(key, data.keys())
            self.assertTrue(type(data[key]) == np.ndarray)


if __name__ == '__main__':
    unittest.main()
