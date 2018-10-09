import numpy as np
from env.env import GRID, ACTION_NAMES
from time import time, sleep
import matplotlib.pyplot as plt

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
		if self.render:
			env.init_fig()
			env.render()
			sleep(1.5)
		while not end:
			action = self.select_action(env.get_feasible_actions())
			if self.render:
				env.render(action=action, show_action=True)
				sleep(0.8)
			reward, state, end = env.step(action)
			if self.render:
				env.render()
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

