import numpy as np
from copy import deepcopy

from ..base_agent import BaseAgent


class ActorCriticAgent(BaseAgent):
    """ActorCriticAgent implements a class of agents using the actor-critic method"""

    def __init__(self, agent_config, net_config, checkpoint_dir, tensorboard_log_dir, render=False, restore=False):
        super().__init__(agent_config['name'], agent_config['gamma'], render)
        self.net = Net(net_config)
        self.net.build()
        if restore:
            latest_checkpoint = get_latest_checkpoint(os.path.join(checkpoint_dir, "checkpoint"))
            self.net.restore(os.path.join(checkpoint_dir, "checkpoint_{}".format(latest_checkpoint)))
            # saver to save (and later restore) model checkpoints
            self.net.saver = tf.train.Saver(max_to_keep=500)
        else:
            self.net.saver = tf.train.Saver(max_to_keep=500)
            self.net.initialize(checkpoint_dir)
        self.net.summary_writer = tf.summary.FileWriter(tensorboard_log_dir, self.net.sess.graph)
        self.state_channels = net_config["state_channels"]

    def collect_data(self, env, T):
        t = 0
        end = False
        data = []
        states = []
        actions = []
        rewards = []

        while t < T and not end:
            state = env.state
            states.append(state)
            action_index = self.select_action(state, env.feasible_actions)
            action = divmod(action_index, 4)
            reward, next_state, end = env.step(action)
            actions.append(action_index)
            rewards.append(reward)
            t += 1

        if end:
            R = 0.
        else:
            R = self.net.get_value(next_state.reshape(-1, 7, 7, self.state_channels)).reshape(-1)

        # evaluate state values of all states encountered in a batch to save time
        state_values = self.net.get_value(np.array(states).reshape(-1, 7, 7, self.state_channels)).reshape(-1)

        assert (len(states) == len(rewards) == len(actions) == len(state_values) == t)

        for s in range(t):
            R = rewards[t - s - 1] + self.gamma * R
            advantage = R - state_values[t - s - 1]
            data = [dict({"state": states[t - s - 1],
                          "advantage": advantage,
                          "action": actions[t - s - 1],
                          "critic_target": R})] + data

        return data, end

    def select_action(self, state, feasible_actions, greedy=False):
        policy = self.net.get_policy(state.reshape(1, 7, 7, self.state_channels))
        # add small epsilon to make sure one of the feasible actions is picked
        policy[policy < 1.0e-7] = 1.0e-7
        policy = mask_out(policy, feasible_actions, GRID)
        policy = policy / np.sum(policy)  # renormalize
        if greedy:
            # max_indices = np.argwhere(policy == np.max(policy))
            # ind = max_indices[np.random.randint(0,len(max_indices))][0]
            ind = np.argmax(policy)
        else:
            ind = np.random.choice(range(len(policy)), p=policy)
        # pos_id, move_id = divmod(ind,4)
        # return pos_id, move_id
        return ind

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

    def play(self, env, greedy=False):
        # play game with agent until the end
        G = 0.0
        discount = 1.0
        end = False

        if self.render:  # render the state of the board at the begining of the game
            env.init_fig()
            env.render()
            sleep(1.5)

        while not end:
            action_index = self.select_action(env.state, env.feasible_actions, greedy=greedy)
            action = divmod(action_index, 4)
            if self.render:
                env.render(action=action, show_action=True)  # render a first time displaying the action selected
                sleep(0.8)
            reward, _, end = env.step(action)
            G += discount * reward
            discount = discount * self.gamma
            if self.render:
                env.render()  # render a second time the state of the board after the action is played
                sleep(0.6)

        if self.render:
            env.render()
            sleep(2)
            plt.close()

        return (G, env.n_pegs)

    def evaluate(self, env, n_games, n_workers):
        # play n_games and store the final reward and number of pegs left for each game
        envs = [deepcopy(env) for _ in range(n_games)]
        pool = ThreadPool(n_workers)
        results = pool.map(self.play, envs)
        rewards, pegs_left = zip(*results)
        pool.close()
        pool.join()
        return dict({"rewards": rewards, "pegs_left": pegs_left})
