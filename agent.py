from __future__ import print_function
from __future__ import division
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import tensorflow as tf
from multiprocessing.dummy import Pool as ThreadPool
from copy import deepcopy
import os

from env.env import GRID, ACTION_NAMES, POS_TO_INDEX
from network.network import Net
from util import mask_out, get_latest_checkpoint, rotate_state_action

class Agent(object):
	"""Agent is the base class for implementing agents to play the game of solitaire"""

	def __init__(self):
		pass

	def play(self, env):
		end = False
		while not end:
			action = self.select_action(state, feasible_actions)
			reward, state, end = env.step(action)

	def select_action(self, state, feasible_actions):
		pass 



class RandomAgent(Agent):
	"""RandomAgent is class of agents which select actions randomly"""

	def __init__(self, name="Random Agent", seed=None, render=False):
		'''
		Instanciates an object of the class RandomAgent by initializing the seed, defining the name of the agent, and setting
		the render parameter.

		Parameters
		----------
		name : string (default "Random Agent")
			The name of the agent.
		seed : int or None (default None)
			The seed to use in numpy.random. If None, the seed is set using the current time by default.
		render : bool (default False)
			Whether or not to display a visual representation of the game as the agent plays.

		Attributes
		----------
		name : string
			The name of the agent.
		render : bool
			Whether or not to display a visual representation of the game as the agent plays.
		'''
		super().__init__()
		if seed is not None:
			np.random.seed(seed)
		self.name = name
		self.render = render


	def play(self, env):
		'''
		Plays a game given the environment `env` until the end, selecting moves at random.

		Parameters
		----------
		env : Env
			The environment with which the agent will interact.
		'''
		end = False
		if self.render: # render the state of the board at the begining of the game 
			env.init_fig()
			env.render()
			sleep(1.5)
		while not end:
			action = self.select_action(env.feasible_actions)
			if self.render:
				env.render(action=action, show_action=True) # render a first time displaying the action selected
				sleep(0.8)
			reward, state, end = env.step(action)
			if self.render: 
				env.render() # render a second time the state of the board after the action is played
				sleep(0.6)
		if self.render:
			env.render()
			sleep(2)
			plt.close()

		return env.n_pegs


	def select_action(self, feasible_actions):
		'''
		Selects an action at random from the legal actions in the current state of the env, which are given by `feasible_actions`.

		Parameters
		----------
		feasible_actions : 2d-array of bools
			An array indicating for each position on the board, whether each action is legal (True) or not (False).

		Returns
		-------
		out : tuple of ints (pos_id, move_id)
			a tuple representing the action selected : which peg to pick up, and where to move it. 
		'''
		actions = np.argwhere(feasible_actions)
		return actions[np.random.randint(0,len(actions))]



class ActorCriticAgent(Agent):
	"""ActorCriticAgent implements a class of agents using the actor-critic method"""

	def __init__(self, net_config, checkpoint_dir, tensorboard_log_dir, name="Actor-Critic Agent", seed=None, render=False, restore=False):
		super().__init__()
		self.net = Net(net_config)
		self.net.build()
		if restore:
			latest_checkpoint = get_latest_checkpoint(os.path.join(checkpoint_dir, "checkpoint"))
			self.net.restore(os.path.join(checkpoint_dir, "checkpoint_{}".format(latest_checkpoint)))
			# saver to save (and later restore) model checkpoints
			self.saver = tf.train.Saver(max_to_keep=500-latest_checkpoint)
		else:
			self.net.initialize(checkpoint_dir)
			self.saver = tf.train.Saver(max_to_keep=500)
		self.summary_writer = tf.summary.FileWriter(tensorboard_log_dir, self.net.sess.graph)
		self.state_channels = net_config["state_channels"]
		self.gamma = net_config["gamma"]

		if seed is not None:
			np.random.seed(seed)
		self.name = name
		self.render = render


	def collect_data(self, env):
		# return state, advantage, action, critic_target
		state = env.state
		action = self.select_action(state, env.feasible_actions)
		reward, next_state, end = env.step(action)
		state_value, next_state_value = self.net.get_value(np.array([state, next_state]).reshape(-1,7,7,self.state_channels)).reshape(-1) # evaluate state values in a batch to save time
		if end:
			next_state_value = 0
		critic_target = reward + self.gamma * next_state_value 
		advantage = critic_target - state_value 
		x,y = GRID[action[0]]
		action = np.array([3-y, x+3, action[1]])
		state, action = rotate_state_action(state, action)
		data = dict({"state" : state, 
					 "advantage" : advantage, 
					 "action" : action,
					 "critic_target" : critic_target})
		return data, end


	def select_action(self, state, feasible_actions, greedy=False):
		policy = mask_out(self.net.get_policy(state.reshape(1,7,7,self.state_channels)), feasible_actions, GRID)
		policy = policy / np.sum(policy) # renormalize
		if greedy:
			max_indices = np.argwhere(policy == np.max(policy))
			x,y,i = max_indices[np.random.randint(0,len(max_indices))]
			return POS_TO_INDEX[(x,y)], i
		else:
			index = np.random.choice(range(policy.size), p=policy.ravel())
			x, ind = divmod(index, (policy.shape[1] * policy.shape[2]))
			y, i = divmod(ind, policy.shape[2])
			return POS_TO_INDEX[(x,y)], i


	def train(self, env, n_games, n_workers, display_every):
			envs = np.array([deepcopy(env) for _ in range(n_games)])
			ended = np.array([False for _ in range(n_games)])

			pool = ThreadPool(n_workers)
			tb_logs = []
			cmpt = 0
			while np.sum(ended) < n_games:
				# collect data from workers using same network stored only once in the base agent
				results = pool.map(self.collect_data, envs[~ended])
				d, ended_new = zip(*results)
				# prepare data to feed to tensorflow
				data = dict({})
				for key in ["advantage", "critic_target"]:
					data[key] = np.array([dp[key] for dp in d]).reshape(-1,1) # reshape to get proper shape for tensorflow input
				data["state"] = np.array([dp["state"] for dp in d]).reshape(-1,7,7,self.state_channels)
				action_mask = np.zeros((len(d), 7, 7, 4), dtype=bool)
				for i, dp in enumerate(d):
					j,k,l = dp["action"]
					action_mask[i,j,k,l] = True
				data["action_mask"] = action_mask
				# update network with the data produced
				summaries, critic_loss, actor_loss, l2_loss, loss = self.net.optimize(data)


				# write obtained summaries to file, so they can be displayed in TensorBoard
				self.net.summary_writer.add_summary(summaries, self.net.steps)
				self.net.summary_writer.flush()

				# display info on optimization step
				if cmpt % display_every == 0:
					print('Losses at step ', cmpt)
					print('loss : {:.3f} | actor loss : {:.3f} | critic loss : {:.3f} | reg loss : {:.3f}'.format(loss,
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
		end = False
		while not end:
			action = self.select_action(env.state, env.feasible_actions, greedy=greedy)
			reward, _, end = env.step(action)
		return (reward, env.n_pegs)


	def evaluate(self, env, n_games, n_workers):
		# play n_games and store the final reward and number of pegs left for each game
		envs = [deepcopy(env) for _ in range(n_games)]
		pool = ThreadPool(n_workers)
		results = pool.map(self.play, envs)
		rewards, pegs_left = zip(*results)
		pool.close()
		pool.join()
		return dict({"rewards" : rewards, "pegs_left" : pegs_left})


		