from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from .rendering import *
from .border_constraints import compute_out_of_border_actions

GRID = [(i,j) for j in [-3,-2] for i in [-1,0,1]] + \
	   [(i,j) for j in [-1,0,1] for i in np.arange(-3,4)] + \
	   [(i,j) for j in [2,3] for i in [-1,0,1]]

# positions (x,y) in the grid are related to indexes (i,j) of an array 7x7 by
# i = 3 - y
# j = x + 3
POS_TO_INDEX = dict({})
for ind, (x,y) in enumerate(GRID):
	POS_TO_INDEX[(3-y,x+3)] = ind


# actions will be  ordered as follows : we take positions as they are ordered in GRID and list in order "up", "down", "right", "left" 
# the 4 possible actions 
N_PEGS = len(GRID) - 1 # = 32 center point in the grid does not contain any peg
N_ACTIONS = len(GRID) * 4
ACTION_NAMES = ["up", "down", "right", "left"]
MOVES = [(0,1), (0,-1), (1,0), (-1,0)]
OUT_OF_BORDER_ACTIONS = compute_out_of_border_actions(GRID)

class Env(object):
	"""A class implementing the solitaire environment"""

	def __init__(self, verbose=False, init_fig=False, interactive_plot=False):
		'''
		Instanciates an object of the class Env by initializing the number of pegs in the game as well as their positions on the grid.

		Parameters
		----------
		verbose : bool (default False)
			Whether or not to display messages.
		init_fig : bool (default False)
			Whether or not to intialize a figure for rendering.

		Attributes
		----------
		n_pegs : int
			Number of pegs remaining on the board
		pegs : dict of tuples of ints
			Keys are the positions in the grid, and values are binary ints indicating the presence (1) or abscence (0) of pegs.
		fig :  matplotlib.figure.Figure
			The figure to render plots
		ax : matplotlib.axes.Axes
			Axes of the plot
		'''
		self.n_pegs = N_PEGS
		assert self.n_pegs == 32
		self._init_pegs()
		if init_fig:
			self.init_fig(interactive_plot)
			pass
		else:
			self.interactive_plot = False
		self.verbose = verbose


	def _init_pegs(self):
		'''
		Initializes the positions of the pegs in the grid : puts a peg on each position except the center one (0,0).
		'''
		self.pegs = dict({})
		for pos in GRID:
			self.pegs[pos] = 1
		self.pegs[(0,0)] = 0 


	def init_fig(self, interactive_plot=True):
		'''
		Initializes the figure and axes for the rendering.

		Parameters
		----------
		interactive_plot : bool (default True)
			Whether the plot is interactive or not.
		'''
		if interactive_plot:
			plt.ion()
		self.fig = plt.figure(figsize=(10,10))


	def reset(self):
		'''
		Resets the environment to its initial state.
		'''
		self.n_pegs = N_PEGS
		self._init_pegs()


	def step(self, action):
		'''
		Returns a length-3 tuple (reward, next_state, end), where reward is a float representing the reward by taking action `action`
		in the current state, nex_state is a representation of the next state after taking the action, and end is a boolean indicating 
		whether or not the game has ended after the action (no more moves available).

		Parameters
		----------
		action : tuple of ints (position_id, move_id) 
			position_id indicates the id of the position on the grid according to the variable GRID. move_id is in {0,1,2,3} and 
			indicates whether the move to make is up, down, right or left according to the variable ACTION_NAMES.

		Returns
		-------
		out : tuple (reward, next_state, end)
			reward is a float, next_state is a 2d-array, and end is a bool. 
		'''
		# update state of the env
		pos_id, move_id = action
		pos = GRID[pos_id]
		x, y = pos
		d_x, d_y = MOVES[move_id]
		self.pegs[pos] = 0 # peg moves from its current position
		self.pegs[(x + d_x, y + d_y)] = 0 # jumps over an adjacent peg
		self.pegs[(x + 2*d_x, y + 2*d_y)] = 1 # ends up in new position
		self.n_pegs -= 1

		# check for game end
		if self.n_pegs == 1:
			if self.verbose:
				print('End of the game, you solved the puzzle !')
			return 1, self.state, True

		else:
			# compute possible next moves
			if np.sum(self.feasible_actions) == 0: # no more actions available
				if self.verbose:
					print('End of the game. You lost : {} pegs remaining'.format(self.n_pegs))
				return 1/(N_PEGS-1), self.state, True
			else:
				# reward is an increasing function of the percentage of the game achieved
				#return ((N_PEGS - self.n_pegs) / (N_PEGS-1)) ** 2, self.state, False
				return 1/(N_PEGS-1), self.state, False


	def get_n_neighbours(self, pos):
		n = 0
		x,y = pos
		if (x+1,y) in self.pegs.keys() and self.pegs[(x+1,y)] == 1:
			n += 1
		if (x,y+1) in self.pegs.keys() and self.pegs[(x,y+1)] == 1:
			n += 1
		if (x-1,y) in self.pegs.keys() and self.pegs[(x-1,y)] == 1:
			n += 1
		if (x,y-1) in self.pegs.keys() and self.pegs[(x,y-1)] == 1:
			n += 1
		return n 


	def get_n_empty(self, pos):
		n = 4
		x,y = pos
		if (x+1,y) not in self.pegs.keys() or self.pegs[(x+1,y)] == 1:
			n -= 1
		if (x,y+1) not in self.pegs.keys() or self.pegs[(x,y+1)] == 1:
			n -= 1
		if (x-1,y) not in self.pegs.keys() or self.pegs[(x-1,y)] == 1:
			n -= 1
		if (x,y-1) not in self.pegs.keys() or self.pegs[(x,y-1)] == 1:
			n -= 1
		return n 


	@property
	def state(self):
		'''
		Returns the state of the env as a 2d-array of ints. The state is represented as a 7x7 grid where values are 1 if there is a peg
		at this position, and 0 otherwise (and 0 outside the board). 
		'''
		#state = np.zeros((7,7,3), dtype=np.int8)
		state = np.zeros((7,7,3), dtype=np.float32)
		for pos, value in self.pegs.items():
			state[3-pos[1], pos[0]+3,0] = value
			# state[3-pos[1], pos[0]+3,1] = self.get_n_neighbours(pos)
			# state[3-pos[1], pos[0]+3,2] = self.get_n_empty(pos)

		state[:,:,1] = (self.n_pegs - 1) / (N_PEGS-1)
		state[:,:,2] = (N_PEGS - self.n_pegs) / (N_PEGS-1)
		return state


	@property
	def feasible_actions(self):
		'''
		Returns a 2d-array of bools indicating, for each position on the grid, whether each action (up, down, right, left) is feasible
		(True) or not (False).
		'''
		actions = np.ones((len(GRID), 4), dtype=bool)
		# go through all positions
		for i, pos in enumerate(GRID):
			if self.pegs[pos] == 0: # if no peg at the position no action feasible from that position
				actions[i,:] = False
			else:
				x,y = pos
				out_of_borders = OUT_OF_BORDER_ACTIONS[i]
				actions[i, out_of_borders==True] = False
				for k in range(4):
					if out_of_borders[k] == False:
						if not self.action_jump_feasible(pos, k):
							actions[i,k] = False
		return actions


	def action_jump_feasible(self, pos, move_id):
		'''
		Returns a boolean indicating whether or not the move of id move_id (in {0,1,2,3}) of jumping over a peg is feasible at 
		position pos in the grid. The move is feasible if there is a peg to jump over and if there is no peg on the potential arrival 
		position in the grid.

		Parameters
		----------
		pos : tuple of ints
			A tuple representing the position considered in the grid. 
		move_id : int
			Gives the id of the move considered.

		Returns
		-------
		out : bool
			A boolean indicating if the move is feasible or not. 
		'''
		x,y = pos
		d_x, d_y = MOVES[move_id]
		return self.pegs[(x + d_x, y + d_y)] == 1 and self.pegs[(x + 2*d_x, y + 2*d_y)] == 0


	def render(self, action=None, show_action=False, show_axes=False):
		'''
		Renders the current state of the environment.

		Parameters
		----------
		action : tuple of ints or None (default None)
			If not None, a tuple (pos_id, move_id) of ints indicating the id of the position in the grid, and the id of the move.
		show_action : bool (default False)
			Indicates whether or not to change the color of the peg being moved and the peg being jumped over in the rendering, in 
			order to be able to visualise the action selected.
		show_axes : bool (default False)
			Whether to display the axes in the rendering or not.
		'''
		ax = plt.gca()
		ax.axes.get_xaxis().set_visible(show_axes)
		ax.axes.get_yaxis().set_visible(show_axes)
		if show_action:
			assert action is not None
			pos_id, move_id = action
			x,y = GRID[pos_id]
			dx,dy = MOVES[move_id]
			jumped_pos = (x + dx, y + dy)
			for pos, value in self.pegs.items():
				if value == 1:
					if pos == (x,y):
						ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.495, color='brown', fill=True))
					elif pos == jumped_pos:
						ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.495, color='black', fill=True))
					else:
						ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.495, color='burlywood', fill=True))
				if value == 0:
					ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.495, color='burlywood', fill=False, linewidth=1.5))

		else:
			assert action is None
			for pos, value in self.pegs.items():
			    if value == 1:
			        ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.495, color='burlywood', fill=True))
			    if value == 0:
			        ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.495, color='burlywood', fill=False, linewidth=1.5))


		plt.ylim(-4, 4)
		plt.xlim(-4, 4)
		plt.axis('scaled')
		plt.title('Current State of the Board')
		self.fig.canvas.draw()
		[p.remove() for p in reversed(ax.patches)]
