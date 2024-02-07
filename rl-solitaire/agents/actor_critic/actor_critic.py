import numpy as np
from copy import deepcopy
from multiprocessing.dummy import Pool as ThreadPool

from ..base_agent import BaseAgent
import torch


class ActorCriticAgent(BaseAgent):
    """ActorCriticAgent implements a class of agents using the actor-critic method."""

    def __init__(self, network: torch.nn.Module, name="ActorCriticAgent", discount=1.0):
        super().__init__(name, discount)
        self.network = network

    def get_value(self, state: np.array) -> np.array:
        """
        Returns the values estimated using the agent's NN on a batch of states.
        :param state: np.array of size (batch_size, state_shape)
        :return: an np.array of shape of size (batch_size,) containing the value of each state in the batch.
        """
        return self.network.get_value(torch.from_numpy(state)).numpy()

    def get_policy(self, state: np.array) -> np.array:
        """
        Returns the policies obtained from agent's NN on a batch of states.
        :param state: np.array of size (batch_size, state_shape)
        :return: an np.array of shape of size (batch_size, N_ACTIONS) containing the policy for each state in the batch.
        """
        return self.network.get_policy(torch.from_numpy(state)).numpy()

    def _format_data(self, states, actions, rewards, next_state, end):
        t = len(states)

        if end:
            value = 0.
        else:
            value = self.get_value(next_state[np.newaxis, :])[0, 0]

        # evaluate state values of all states encountered in a batch to save time
        state_values = self.get_value(np.array(states)).reshape(-1)

        assert (t == len(rewards) == len(actions) == len(state_values))

        data = []
        for s in range(t):
            value = rewards[t - s - 1] + self.discount * value
            advantage = value - state_values[t - s - 1]
            data = [dict({"state": states[t - s - 1],
                          "advantage": advantage,
                          "action": actions[t - s - 1],
                          "critic_target": value})] + data

        return data, end

    def train(self, env, n_games, data_buffer, batch_size, n_workers, display_every, T_update_net):
        envs = np.array([deepcopy(env) for _ in range(n_games)])
        ended = np.array([False for _ in range(n_games)])

        ## USE A BUFFER OF COLLECTED DATA TO TRAIN THE NETWORK (or wait until the end of a game to use the values)

        pool = ThreadPool(n_workers)
        tb_logs = []
        cmpt = 0
        while np.sum(ended) < n_games:
            # collect data from workers using same network stored only once in the base agent
            results = pool.map(lambda x: self.collect_data(x, T_update_net), envs[~ended])
            d, ended_new = zip(*results)
            # add data to buffer
            data_buffer.add_list([el for l in d for el in l])

            # sample data from the buffer
            batch = data_buffer.sample(n_samples=batch_size)
            # prepare data to feed to tensorflow
            data = dict({})
            for key in ["advantage", "critic_target"]:
                data[key] = np.array([dp[key] for dp in batch]).reshape(-1,
                                                                        1)  # reshape to get proper shape for tensorflow input
            data["state"] = np.array([dp["state"] for dp in batch]).reshape(-1, 7, 7, self.state_channels)
            action_mask = np.zeros((len(batch), N_ACTIONS), dtype=np.float32)
            for i, dp in enumerate(batch):
                index = dp["action"]
                action_mask[i, index] = 1.0
            data["action_mask"] = action_mask
            # update network with the data produced
            summaries, critic_loss, actor_loss, l2_loss, loss = self.net.optimize(data)

            # write obtained summaries to file, so they can be displayed in TensorBoard
            self.net.summary_writer.add_summary(summaries, self.net.steps)
            self.net.summary_writer.flush()

            # display info on optimization step
            if cmpt % display_every == 0:
                print('Losses at step ', cmpt)
                print('loss : {:.3f} | actor loss : {:.5f} | critic loss : {:.6f} | reg loss : {:.3f}'.format(loss,
                                                                                                              actor_loss,
                                                                                                              critic_loss,
                                                                                                              l2_loss))

            # update values of cmpt and ended
            cmpt += 1
            ended[~ended] = ended_new

        pool.close()
        pool.join()

    def evaluate(self, env, n_games, n_workers):
        # play n_games and store the final reward and number of pegs left for each game
        envs = [deepcopy(env) for _ in range(n_games)]
        pool = ThreadPool(n_workers)
        results = pool.map(self.play, envs)
        rewards, pegs_left = zip(*results)
        pool.close()
        pool.join()
        return dict({"rewards": rewards, "pegs_left": pegs_left})
