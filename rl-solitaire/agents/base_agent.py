from time import sleep
import matplotlib.pyplot as plt
import numpy as np

from .agent_config import AgentConfig
from .utils import mask_infeasible_actions

MIN_ACTION_PROBA = 1.0e-7


class BaseAgent:
    """Agent is the base class for implementing agents to play the game of solitaire"""

    def __init__(self, name="BaseAgent", discount=1.0):
        self._name = name
        self.discount = discount

    @property
    def name(self):
        return self._name

    def play(self, env, render=False):
        """
        A method to interact with the environment until a terminal state is reached.
        :param env: the environment to interact with.
        :param render: whether to render the different states the environment goes through.
        :return:
        """
        G = 0.0
        end = False
        discount = self.discount

        if render:  # render the state of the board at the beginning of the game
            env.init_fig()
            env.render()
            sleep(1.5)

        while not end:
            action = self.select_action(env.state, env.feasible_actions)
            if render:
                env.render(action=action, show_action=True)  # render a first time displaying the action selected
                sleep(0.8)
            reward, _, end = env.step(action)
            G += discount * reward
            discount = discount * self.discount
            if render:
                env.render()  # render a second time the state of the board after the action is played
                sleep(0.6)

        if render:
            env.render()
            sleep(2)
            plt.close()

        return G, env.n_pegs

    def select_action(self, state, feasible_actions, greedy=False):
        policy = self.get_policy(state)
        # add small epsilon to make sure one of the feasible actions is picked (avoid issues with numerical errors)
        policy[policy < MIN_ACTION_PROBA] = MIN_ACTION_PROBA
        policy = mask_infeasible_actions(policy, feasible_actions)  # action probas are re-normalized by default
        if greedy:
            action_index = np.argmax(policy)
        else:
            action_index = np.random.choice(range(len(policy)), p=policy)
        return action_index

    def collect_data(self, env, T):
        if T <= 0:
            raise ValueError(f"T must be >= 1 but was {T}")
        t = 0
        end = False
        states = []
        actions = []
        rewards = []
        next_state = env.state

        while t < T and not end:
            state = env.state
            states.append(state)
            action_index = self.select_action(state, env.feasible_actions)
            action = env.convert_action_id_to_action(action_index)
            reward, next_state, end = env.step(action)
            actions.append(action_index)
            rewards.append(reward)
            t += 1

        return self._format_data(states, actions, rewards, next_state, end), end

    def _format_data(self, states, actions, rewards, next_state, end):
        t = len(states)
        assert (t == len(actions) == len(rewards))
        data = [{"state": states[s],
                 "action": actions[s],
                 "reward": rewards[s]} for s in range(t)]
        return data

    def get_policy(self, state: np.array):
        pass

    def get_value(self, state: np.array):
        pass
