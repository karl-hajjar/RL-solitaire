import numpy as np
from time import time, sleep
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm

from env.env import GRID, ACTION_NAMES
from network.network import Net

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
			The seed to use in numpy.random. If None, the seed is set using the current time.
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
		if seed is None:
			np.random.seed(int(time()))
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

	def __init__(self, env, net_config, name="Actor-Critic Agent", seed=None, render=False):
		super().__init__()
		self.env = env
		self.net = Net(net_config)
		self.net.build()
		self.net.initialize()

		if seed is None:
			np.random.seed(int(time()))
		self.name = name
		self.render = render


	def collect_data(self, env):
		# return state, advantage, critic_target
		state = env.state
		action = self.select_action(state, env.feasible_actions)
		reward, next_state, end = env.step(action)
		state_value, next_state_value = net.get_value([state, next_state]) # evaluate state values in a batch to save time
		critic_target = reward + next_state_value 
		advantage = critic_target - state_value 
		action_prob = self.net.get_policy(state)[action]
		data = [state, advantage, action_prob, critic_target]
		return data, end


	def select_action(self, state, feasible_actions, greedy=False):
		policy = self.net.get_policy(state)
		policy[~feasible_actions] = 0 # mask out infeasible actions
		policy /= np.sum(policy) # renormalize
		if greedy:
			max_indices = np.argwhere(policy == np.max(policy))
			return max_indices[np.random.randint(0,len(max_indices))]
		else:
			index = np.random.choice(range(policy.size), p=policy.ravel())
			return divmod(index, policy.shape[1])


	def train(self, env, n_iter, n_workers):
		for it in tqdm(range(n_iter), desc="parallel gameplay iterations"):

			envs = [env for _ in range(n_workers)]
			ended = [False for _ in range(n_workers)]

			pool = ThreadPool(n_workers) 
			while np.sum(ended) < n_workers:
				# collect data from workers using same network stored only once in the base agent
				results = pool.map(self.collect_data, envs[~ended])
				data, ended_new = zip(*results)
				ended[~ended] = ended_new
				self.net.optimize(data)

			pool.close()
			pool.join()



		