import numpy as np
from env.env import GRID, ACTION_NAMES
from time import time, sleep
#import matplotlib.pyplot as plt

class Agent(object):
	"""docstring for Agent"""

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
	"""docstring for RandomAgent"""

	def __init__(self, name="Random Agent", seed=None, render=False, show_available_moves=False, verbose=False):
		super().__init__()
		if seed is None:
			np.random.seed(int(time()))
		self.name = name
		self.render = render
		if not render:
			self.show_available_moves = False
		else:
			self.show_available_moves = show_available_moves
		self.verbose = verbose


	def play(self, env):
		end = False
		if self.render:
			env.render(self.show_available_moves)
		cpt = 1
		while not end:
			action = self.select_action(env.get_feasible_actions())
			if self.verbose:
				print('\n\t\t\tMove {} : peg at position {} selected to move {}\n'.format(cpt, GRID[action[0]], ACTION_NAMES[action[1]]))
				cpt += 1
			reward, state, end = env.step(action)
			if self.render:
				env.render(self.show_available_moves)
				sleep(2)
				#plt.gcf().clear()


	def select_action(self, feasible_actions):
		actions = np.argwhere(feasible_actions)
		return actions[np.random.randint(0,len(actions))]

