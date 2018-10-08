import matplotlib.pyplot as plt
import matplotlib

def plot_pegs(pegs):
	'''
	Plots the pegs on the board.

	Parameters
	----------
	pegs : dict 
		Indicates for each position (in the keys of the dict) if their is a peg or not.
	'''
	fig = plt.figure(figsize=(6,6))
	ax=plt.gca()
	for key, value in pegs.items():
	    if value == 1:
	        ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='burlywood', fill=True))
	    if value == 0:
	        ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='burlywood', fill=False, linewidth=1.5))
	plt.ylim(-4, 4)
	plt.xlim(-4, 4)
	plt.axis('scaled')
	plt.title('Current State of the Board')
	plt.show()
	plt.close(fig)
	#plt.gcf().clear()


def plot_available_moves(grid, pegs, feasible_actions, action_id, action_names):
	'''
	Plots the available pegs to move for each of the 4 possible actions.

	Parameters
	----------
	grid : list of tuples of ints 
		The coordinates of the positions in the grid
	pegs : dict 
		Indicates for each position (in the keys of the dict) if their is a peg or not.
	feasible_actions : 2d-array of bools
		Indicates for each position in the grid, whether each of the 4 actions is feasible or not.
	action_id : int
		The id of the action considered ({0,1,2,3}).
	action_names : list of strrings
		The names of each action from 0 to 3.
	'''
	ax=plt.gca()
	for i, pos in enumerate(grid):
	    if pegs[pos] == 1:
	        if feasible_actions[i][action_id]:
	            ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.5, color='brown', fill=True))
	        else:
	            ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.5, color='burlywood', fill=True)) 
	    if pegs[pos] == 0:
	        ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.5, color='burlywood', fill=False, linewidth=1.5)) 
	plt.ylim(-4, 4)
	plt.xlim(-4, 4)
	plt.axis('scaled')
	plt.title('Allowed {} moves'.format(action_names[action_id]))


