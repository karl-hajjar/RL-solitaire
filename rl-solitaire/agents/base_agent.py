from time import sleep
import matplotlib.pyplot as plt
import numpy as np

from .utils import mask_infeasible_actions

MIN_ACTION_PROBA = 1.0e-7


class BaseAgent:
    """Agent is the base class for implementing agents to play the game of solitaire"""

    def __init__(self, name="BaseAgent", discount=1.0):
        self._name = name
        self.discount = discount

    @property
    def name(self) -> str:
        return self._name

    def play(self, env, render=False) -> (float, int):
        """
        A method to interact with the environment until a terminal state is reached.
        :param env: the environment to interact with.
        :param render: whether to render the different states the environment goes through.
        :return: The final (discounted) cumulative reward and the number of pegs left.
        """
        total_return = 0.0
        end = False
        discount = self.discount

        if render:  # render the state of the board at the beginning of the game
            env.init_fig()
            env.render()
            sleep(1.5)

        while not end:
            action_index = self.select_action(env.state, env.feasible_actions)
            action = env.convert_action_id_to_action(action_index)
            if render:
                env.render(action=action, show_action=True)  # render a first time displaying the action selected
                sleep(0.8)
            reward, _, end = env.step(action)
            total_return += discount * reward
            discount = discount * self.discount
            if render:
                env.render()  # render a second time the state of the board after the action is played
                sleep(0.6)

        if render:
            env.render()
            sleep(2)
            plt.close()

        return total_return, env.n_pegs

    def select_action(self, state: np.array, feasible_actions: np.array, greedy=False) -> int:
        """
        Returns the selected action given the state and the feasible actions in this state.
        :param state: 3d np.array of shape (7, 7, N_STATE_CHANNELS).
        :param feasible_actions: 2d np.array of shape (len(GRID), len(MOVES)).
        :param greedy: a boolean determining if action is sampled wrt to policy probabilities or selected greedily as
        the argmax of the policy.
        :return: an int representing the index in (0, ..., N_ACTIONS-1) of the selected action.
        """
        policy = self.get_policy(state)
        # add small epsilon to make sure one of the feasible actions is picked (avoid issues with numerical errors)
        policy[policy < MIN_ACTION_PROBA] = MIN_ACTION_PROBA
        policy = mask_infeasible_actions(policy, feasible_actions)  # action probas are re-normalized by default
        if greedy:
            action_index = np.argmax(policy)
        else:
            action_index = np.random.choice(range(len(policy)), p=policy)
        return action_index

    def collect_data(self, env, T) -> (list, bool):
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

    def evaluate(self, env, n_games=1) -> (list[float], list[float]):
        rewards = []
        pegs_left = []
        for _ in range(n_games):
            env.reset()
            reward, n_pegs_left = self.play(env)
            rewards.append(reward)
            pegs_left.append(n_pegs_left)

        return rewards, pegs_left
