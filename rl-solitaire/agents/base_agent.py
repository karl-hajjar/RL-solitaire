from time import sleep
import matplotlib.pyplot as plt
import numpy as np

from agents.utils import mask_infeasible_actions

MIN_ACTION_PROBA = 1.0e-7


class BaseAgent:
    """Agent is the base class for implementing agents to play the game of solitaire"""

    def __init__(self, name="BaseAgent", discount=1.0):
        self._name = name
        self.discount = discount

    @property
    def name(self) -> str:
        return self._name

    def set_evaluation_mode(self):
        pass

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

    def select_action(self, state: np.ndarray, feasible_actions: np.ndarray, greedy=False) -> int:
        """
        Returns the selected action given the state and the feasible actions in this state.
        :param state: 3d np.ndarray of shape (7, 7, N_STATE_CHANNELS).
        :param feasible_actions: 2d np.ndarray of shape (len(GRID), len(MOVES)).
        :param greedy: a boolean determining if action is sampled wrt to policy probabilities or selected greedily as
        the argmax of the policy.
        :return: an int representing the index in (0, ..., N_ACTIONS-1) of the selected action.
        """
        # Reshape state as get_policy expects a batch of states, and then return only the first policy of the batch
        policy = self.get_policy(state[np.newaxis, :])[0]
        # add small epsilon to make sure one of the feasible actions is picked (avoid issues with numerical errors)
        policy[policy < MIN_ACTION_PROBA] = MIN_ACTION_PROBA
        # feasible action probas are re-normalized by default
        policy = mask_infeasible_actions(policy, feasible_actions.reshape(-1))
        if greedy:
            action_index = np.argmax(policy)
        else:
            action_index = np.random.choice(range(len(policy)), p=policy)
        return action_index

    def collect_data(self, env, T=None) -> dict[str, np.ndarray]:
        states, actions, action_masks, rewards, next_state, end = self.collect_data_(env, T=None)
        return self._format_data(states, actions, action_masks, rewards, next_state, end)

    def collect_data_(self, env, T=None) -> (list[np.ndarray], list[int], list[float], list[float], np.ndarray, bool):
        if T is None:
            T = env.N_MAX_STEPS
        elif T <= 0:
            raise ValueError(f"If T is not None, T must be >= 1 but was {T}")
        t = 0
        end = False
        states = []
        actions = []
        action_masks = []
        rewards = []
        next_state = env.state

        while t < T and not end:
            state = env.state
            states.append(state)

            feasible_actions = env.feasible_actions
            action_mask = mask_infeasible_actions(np.ones_like(feasible_actions.reshape(-1)),
                                                  feasible_actions.reshape(-1),
                                                  normalize=False).astype(float)
            action_masks.append(action_mask)

            action_index = self.select_action(state, feasible_actions)
            action = env.convert_action_id_to_action(action_index)

            reward, next_state, end = env.step(action)
            actions.append(action_index)
            rewards.append(reward)
            t += 1

        return states, actions, action_masks, rewards, next_state, end

    def _format_data(self, states, actions, action_masks, rewards, next_state, end) -> dict[str, np.ndarray]:
        assert (len(states) == len(actions) == len(rewards))
        return {"states": np.array(states),
                "actions": np.array(actions).astype(np.float32),
                "rewards": np.array(rewards).astype(np.float32)}

    def get_policy(self, states: np.ndarray) -> np.ndarray:
        """
        Returns the policies for a given batch of states.
        :param states: np.ndarray of shape (batch_size, state_shape)
        :return: np.ndarray of size (batch_size, N_ACTIONS) containing the policies of each state in the batch.
        """
        return np.array([])

    def evaluate(self, env, n_games=1) -> (list[float], list[float]):
        rewards = []
        pegs_left = []
        for _ in range(n_games):
            env.reset()
            reward, n_pegs_left = self.play(env)
            rewards.append(reward)
            pegs_left.append(n_pegs_left)

        return rewards, pegs_left
