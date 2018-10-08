from __future__ import print_function
import numpy as np
from rendering import *
from border_constraints import compute_out_of_border_actions

GRID = [(i,j) for j in [-3,-2] for i in [-1,0,1]] + \
	   [(i,j) for j in [-1,0,1] for i in np.arange(-3,4)] + \
	   [(i,j) for j in [2,3] for i in [-1,0,1]]

N_PEGS = len(GRID) - 1 # center point in the grid does not contain any peg
ACTION_NAMES = ["up", "down", "right", "left"]
MOVES = [(0,1), (0,-1), (1,0), (-1,0)]
OUT_OF_BORDER_ACTIONS = compute_out_of_border_actions(GRID)

class Env(object):
	"""A class implementing the solitaire environment"""

	def __init__(self):
		'''
		instanciates an object of the class Env by initializing the number of pegs in the game as well as their positions on the grid.
		'''
		#super(ClassName, self).__init__()
		self.n_pegs = N_PEGS 
		assert self.n_pegs == 32
		self._init_pegs()


	def _init_pegs(self):
		'''
		initializes the positions of the pegs in the grid : puts a peg on each position except the center one (0,0).
		'''
		self.pegs = dict({})
		for pos in GRID:
			self.pegs[pos] = 1
		self.pegs[(0,0)] = 0 


	def reset(self):
		'''
		resets the environment to its initial state.
		'''
		self.__intit__()


	def step(self, action):
		'''
		Returns a length-3 tuple (reward, next_state, end), where reward is a float representing the reward by taking action `action`
		in the current state, nex_state is a representation of the next state after taking the action, and end is a boolean indicating 
		whether or not the game has ended after the action (no more moves available).

		Parameters
		----------
		action : tuple of ints (position_id, move_id). position_id indicates the id of the position on the grid according to the 
			variable GRID. move_id is in {0,1,2,3} and indicates whether the move to make is up, down, right or left according to the
			variable ACTION_NAMES.

		Returns
		-------
			out : a tuple (reward, next_state, end). reward is a float, next_state is a 2d-array, and end is a bool. 
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
			print('End of the game, you solved the puzzle !')
			return 1, self.pegs, True

		else:
			# compute possible next moves
			actions = self.get_feasible_actions()
			if np.sum(actions) == 0: # no more actions available
				print('End of the game. You lost : {} pegs remaining'.format(self.n_pegs))
				return -self.n_pegs, self.pegs, True
			else:
				return 0, self.pegs, False


	def get_feasible_actions(self):
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
		pos : a tuple of ints representing the position considered in the grid. 
		move_id : an int giving the id of the move considered.

		Returns
		-------
			out : a bool indicating if the move is fesible or not. 
		'''
		x,y = pos
		d_x, d_y = MOVES[move_id]
		return self.pegs[(x + d_x, y + d_y)] == 1 and self.pegs[(x + 2*d_x, y + 2*d_y)] == 0


	def render(self):
		'''

		'''
