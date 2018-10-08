import numpy as np

def compute_out_of_border_actions(grid):
	'''
	Returns a 2d-array whose shape is (n,m) where n is the number of positions in the grid, and m=4 for every possible move (up, down, 
	right, left).

	Parameters
	----------
	grid : list of tuples (x,y) of ints
		List of positions in the grid.

	Returns
	-------
	out : 2d-array of bools
		An array specifying, for each position, if moves will end up out of the borders of the game (True) or not (False).
	'''
	out_of_border = np.zeros((len(grid), 4), dtype=bool)
	for i, pos in enumerate(grid):
		x,y = pos

		# check up
		if y >= 0:
			if x < -1 or x > 1:
				out_of_border[i, 0] = True
			else:
				if y >= 2:
					out_of_border[i, 0] = True

		# check down
		if y <= 0:
			if x < -1 or x > 1:
				out_of_border[i, 1] = True
			else:
				if y <= -2:
					out_of_border[i, 1] = True

		# check right
		if x >= 0:
			if y < -1 or y > 1:
				out_of_border[i, 2] = True
			else:
				if x >= 2:
					out_of_border[i, 2] = True

		# check left
		if x <= 0:
			if y < -1 or y > 1:
				out_of_border[i, 3] = True
			else:
				if x <= -2:
					out_of_border[i, 3] = True

	return out_of_border