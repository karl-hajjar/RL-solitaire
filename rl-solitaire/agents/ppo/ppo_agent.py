import numpy as np

from ..base_agent import BaseAgent
import torch


class PPOAgent(BaseAgent):
    """PPOAgent implements a class of agents using Proximal Policy Optimization (PPO) methods."""

    def __init__(self, network: torch.nn.Module, name="ActorCriticAgent", discount=1.0):
        super().__init__(name, discount)
        self.network = network

    def set_evaluation_mode(self):
        self.network.eval()

    def get_value(self, state: np.ndarray) -> np.ndarray:
        """
        Returns the values estimated using the agent's NN on a batch of states.
        :param state: np.ndarray of size (batch_size, state_shape)
        :return: an np.ndarray of shape of size (batch_size,) containing the value of each state in the batch.
        """
        return self.network.get_value(torch.from_numpy(state)).numpy()

    def get_policy(self, states: np.ndarray) -> np.ndarray:
        """
        Returns the policies obtained from agent's NN on a batch of states.
        :param states: np.ndarray of size (batch_size, state_shape)
        :return: an np.ndarray of shape of size (batch_size, N_ACTIONS) containing the policy for each state in the
        batch.
        """
        return self.network.get_policy(torch.from_numpy(states)).numpy()

    def _format_data(self, states, actions, action_masks, rewards, next_state, end) -> dict[str, np.ndarray]:
        t = len(states)
        if end:
            value = 0.
        else:
            value = self.get_value(next_state[np.newaxis, :])[0, 0]

        # evaluate state values of all states encountered in a batch to save time
        states = np.array(states)
        bootstrapped_state_values = self.get_value(states).reshape(-1)

        # get old policy for each all states in a batch to save time
        old_policies = self.get_policy(states)
        actions = np.array(actions)

        assert (t == len(rewards) == len(actions) == len(bootstrapped_state_values))
        reversed_value_targets = []
        reversed_advantages = []
        for s in range(t):
            value = rewards[t - s - 1] + self.discount * value
            reversed_value_targets.append(value)
            reversed_advantages.append(value - bootstrapped_state_values[t - s - 1])

        return {"states": states,
                "actions": actions,
                "action_probas": old_policies[np.arange(t), actions].reshape(-1, 1),
                "action_masks": np.array(action_masks).astype(np.float32),
                "advantages": np.array(reversed_advantages[::-1]).reshape(-1, 1).astype(np.float32),
                "value_targets": np.array(reversed_value_targets[::-1]).reshape(-1, 1).astype(np.float32)}
