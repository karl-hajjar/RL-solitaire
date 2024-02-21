import unittest
import os
import numpy as np
import torch
import pickle

from nn.network_config import NetConfig
from nn.policy_value.fully_connected import FCPolicyValueNet
from agents.actor_critic.actor_critic_agent import ActorCriticAgent
from agents.actor_critic.actor_critic_trainer import ActorCriticTrainer
from env.env import Env, N_STATE_CHANNELS, N_ACTIONS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
FC_POLICY_VALUE_CONFIG_FILE_PATH = os.path.join(ROOT, "nn/policy_value/fc_policy_value_config.yaml")
RESOURCES_DIR = os.path.join(ROOT, "tests/agents/actor_critic/resources")


class TestActorCriticTrainer(unittest.TestCase):
    def setUp(self) -> None:
        self.net_config = NetConfig(config_path=FC_POLICY_VALUE_CONFIG_FILE_PATH)
        self.log_dir = RESOURCES_DIR
        self.checkpoints_dir = os.path.join(RESOURCES_DIR, "checkpoints")
        self.agent_results_filepath = os.path.join(RESOURCES_DIR, "results.pickle")

    def test_trainer_init(self):
        network = FCPolicyValueNet(config=self.net_config)
        agent = ActorCriticAgent(network=network, name="FCActorCriticAgent")
        env = Env()
        trainer = ActorCriticTrainer(env, agent, n_iter=2, n_games_train=5,
                                     agent_results_filepath=self.agent_results_filepath, log_every=1,
                                     log_dir=self.log_dir, checkpoints_dir=self.checkpoints_dir)

        self.assertEqual(trainer.trainer.max_epochs, 1)
        self.assertEqual(trainer.trainer.max_steps, -1)

        n_optim_steps = 50
        trainer = ActorCriticTrainer(env, agent, n_iter=2, n_games_train=5, n_optim_steps=n_optim_steps,
                                     agent_results_filepath=self.agent_results_filepath, log_every=1,
                                     log_dir=self.log_dir, checkpoints_dir=self.checkpoints_dir)

        self.assertIsNone(trainer.trainer.max_epochs)
        self.assertEqual(trainer.trainer.max_steps, n_optim_steps)

    def test_collect_data(self):
        network = FCPolicyValueNet(config=self.net_config)
        agent = ActorCriticAgent(network=network, name="FCActorCriticAgent")
        env = Env()
        trainer = ActorCriticTrainer(env, agent, n_iter=2, n_games_train=5,
                                     agent_results_filepath=self.agent_results_filepath, log_every=1,
                                     log_dir=self.log_dir, checkpoints_dir=self.checkpoints_dir)
        with torch.no_grad():
            data = trainer.collect_data()

        for key in ["states", "actions", "action_masks", "advantages", "value_targets"]:
            self.assertIn(key, data.keys())
            self.assertTrue(type(data[key]) == np.ndarray)

        T = len(data["states"])
        self.assertGreater(T, trainer.n_games_train)
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

    def test_reformat_data(self):
        network = FCPolicyValueNet(config=self.net_config)
        agent = ActorCriticAgent(network=network, name="FCActorCriticAgent")
        env = Env()
        trainer = ActorCriticTrainer(env, agent, n_iter=2, n_games_train=5,
                                     agent_results_filepath=self.agent_results_filepath, log_every=1,
                                     log_dir=self.log_dir, checkpoints_dir=self.checkpoints_dir)

        with torch.no_grad():
            data = trainer.collect_data()
            T_ = len(data["states"])
            data = trainer.reformat_data(data)

            for key in ["states", "actions", "action_masks", "advantages", "value_targets"]:
                self.assertIn(key, data.keys())
                self.assertTrue(type(data[key]) == torch.Tensor)

            T = len(data["states"])
            self.assertEqual(T_, T)
            self.assertGreater(T, trainer.n_games_train)
            self.assertSequenceEqual(data["states"].shape, (T, 7, 7, N_STATE_CHANNELS))
            self.assertSequenceEqual(data["actions"].shape, (T,))
            self.assertSequenceEqual(data["action_masks"].shape, (T, N_ACTIONS))
            self.assertSequenceEqual(data["advantages"].shape, (T, 1))
            self.assertSequenceEqual(data["value_targets"].shape, (T, 1))

            self.assertTrue(data["states"].dtype == torch.float32)
            self.assertTrue(data["actions"].dtype == torch.int64)
            self.assertTrue(data["action_masks"].dtype == torch.float32)
            self.assertTrue(data["advantages"].dtype == torch.float32)
            self.assertTrue(data["value_targets"].dtype == torch.float32)

    def test_prepare_dataset(self):
        network = FCPolicyValueNet(config=self.net_config)
        agent = ActorCriticAgent(network=network, name="FCActorCriticAgent")
        env = Env()
        trainer = ActorCriticTrainer(env, agent, n_iter=2, n_games_train=5,
                                     agent_results_filepath=self.agent_results_filepath, log_every=1,
                                     log_dir=self.log_dir, checkpoints_dir=self.checkpoints_dir)

        with torch.no_grad():
            data = trainer.collect_data()
            T_ = len(data["states"])
            dataset = trainer.prepare_dataset(data)

            for key in ["states", "action_indices", "action_masks", "advantages", "value_targets"]:
                self.assertTrue(hasattr(dataset, key))
                self.assertTrue(type(getattr(dataset, key)) == torch.Tensor)

            T = len(dataset)
            self.assertEqual(T_, T)
            self.assertGreater(T, trainer.n_games_train)
            self.assertSequenceEqual(dataset.states.shape, (T, 7, 7, N_STATE_CHANNELS))
            self.assertSequenceEqual(dataset.action_indices.shape, (T,))
            self.assertSequenceEqual(dataset.action_masks.shape, (T, N_ACTIONS))
            self.assertSequenceEqual(dataset.advantages.shape, (T, 1))
            self.assertSequenceEqual(dataset.value_targets.shape, (T, 1))

            self.assertTrue(dataset.states.dtype == torch.float32)
            self.assertTrue(dataset.action_indices.dtype == torch.int16)
            self.assertTrue(dataset.action_masks.dtype == torch.float32)
            self.assertTrue(dataset.advantages.dtype == torch.float32)
            self.assertTrue(dataset.value_targets.dtype == torch.float32)

            i = T // 2
            item = dataset[i]
            self.assertSequenceEqual(item[0].shape, (7, 7, N_STATE_CHANNELS))
            self.assertSequenceEqual(item[1].shape, ())
            self.assertSequenceEqual(item[2].shape, (N_ACTIONS,))
            self.assertSequenceEqual(item[3].shape, (1,))
            self.assertSequenceEqual(item[4].shape, (1,))

    def test_prepare_dataloader(self):
        network = FCPolicyValueNet(config=self.net_config)
        agent = ActorCriticAgent(network=network, name="FCActorCriticAgent")
        env = Env()
        batch_size = 16
        trainer = ActorCriticTrainer(env, agent, n_iter=2, n_games_train=5, batch_size=batch_size,
                                     agent_results_filepath=self.agent_results_filepath, log_every=1,
                                     log_dir=self.log_dir, checkpoints_dir=self.checkpoints_dir)

        with torch.no_grad():
            data = trainer.collect_data()
            T_ = len(data["states"])
            dataset = trainer.prepare_dataset(data)
            dataloader = trainer.prepare_dataloader(dataset)
            T = len(dataset)
            self.assertEqual(T_, T)

            N, r = divmod(T, batch_size)
            for i, batch in enumerate(dataloader):
                # self.states, self.action_indices, self.action_masks, self.advantages, self.value_targets
                states, action_indices, action_masks, advantages, value_targets = batch

                self.assertTrue(type(states) == torch.Tensor)
                self.assertTrue(type(action_indices) == torch.Tensor)
                self.assertTrue(type(action_masks) == torch.Tensor)
                self.assertTrue(type(advantages) == torch.Tensor)
                self.assertTrue(type(value_targets) == torch.Tensor)

                if i < N:
                    self.assertSequenceEqual(states.shape, (batch_size, 7, 7, N_STATE_CHANNELS))
                    self.assertSequenceEqual(action_indices.shape, (batch_size,))
                    self.assertSequenceEqual(action_masks.shape, (batch_size, N_ACTIONS))
                    self.assertSequenceEqual(advantages.shape, (batch_size, 1))
                    self.assertSequenceEqual(value_targets.shape, (batch_size, 1))
                else:
                    self.assertSequenceEqual(states.shape, (r, 7, 7, N_STATE_CHANNELS))
                    self.assertSequenceEqual(action_indices.shape, (r,))
                    self.assertSequenceEqual(action_masks.shape, (r, N_ACTIONS))
                    self.assertSequenceEqual(advantages.shape, (r, 1))
                    self.assertSequenceEqual(value_targets.shape, (r, 1))

                self.assertTrue(states.dtype == torch.float32)
                self.assertTrue(action_indices.dtype == torch.int32)
                self.assertTrue(action_masks.dtype == torch.float32)
                self.assertTrue(advantages.dtype == torch.float32)
                self.assertTrue(value_targets.dtype == torch.float32)

    def test_train(self):
        network = FCPolicyValueNet(config=self.net_config)
        agent = ActorCriticAgent(network=network, name="FCActorCriticAgent")
        env = Env()
        n_iter = 60
        n_games_train = 20
        n_games_eval = 50
        trainer = ActorCriticTrainer(env, agent, n_iter=n_iter, n_games_train=n_games_train, n_games_eval=n_games_eval,
                                     agent_results_filepath=self.agent_results_filepath, log_every=1,
                                     log_dir=self.log_dir, checkpoints_dir=self.checkpoints_dir)

        trainer.train()
        results = []
        with open(self.agent_results_filepath, 'rb') as file:
            while True:
                try:
                    results.append(pickle.load(file))
                except EOFError:
                    break
        self.assertEqual(len(results), n_iter)
        for i in range(n_iter):
            self.assertIn("rewards", results[i].keys())
            self.assertIn("pegs_left", results[i].keys())
            self.assertTrue(type(results[i]["rewards"]) == list)
            self.assertTrue(type(results[i]["pegs_left"]) == list)
            self.assertEqual(len(results[i]["rewards"]), trainer.n_games_eval)
            self.assertEqual(len(results[i]["pegs_left"]), trainer.n_games_eval)


if __name__ == '__main__':
    unittest.main()
