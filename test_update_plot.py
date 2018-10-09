import matplotlib.pyplot as plt
import numpy as np
from time import sleep

from env.env import Env, GRID, MOVES
import matplotlib


def plot_pegs(pegs, action=None, show_action=False):
	ax=plt.gca()
	if show_action:
		assert action is not None
		pos_id, move_id = action
		x,y = GRID[pos_id]
		dx,dy = MOVES[move_id]
		jumped_pos = (x + dx, y + dy)
		for key, value in env.pegs.items():
			if value == 1:
				if key == (x,y):
					ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='brown', fill=True))
				elif key == jumped_pos:
					ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='black', fill=True))
				else:
					ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='burlywood', fill=True))
			if value == 0:
				ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='burlywood', fill=False, linewidth=1.5))

	else:
		assert action is None
		for key, value in env.pegs.items():
		    if value == 1:
		        ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='burlywood', fill=True))
		    if value == 0:
		        ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='burlywood', fill=False, linewidth=1.5))


	plt.ylim(-4, 4)
	plt.xlim(-4, 4)
	plt.axis('scaled')
	plt.title('Current State of the Board')
	fig.canvas.draw()
	[p.remove() for p in reversed(ax.patches)]



env = Env()
end = False

plt.ion()
fig = plt.figure(figsize=(6,6))
ax=plt.gca()

i = 0
while not end:
	actions = np.argwhere(env.get_feasible_actions())
	action = actions[np.random.randint(0,len(actions))]
	# action_pos = GRID[action[0]]
	# dx, dy = MOVES[action[1]]
	# jumped_pos = (action_pos[0] + dx, action_pos[1] + dy)

	# ax=plt.gca()
	# for key, value in env.pegs.items():
	# 	if value == 1:
	# 		if key == action_pos:
	# 			ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='brown', fill=True))
	# 		elif key == jumped_pos:
	# 			ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='black', fill=True))
	# 		else:
	# 			ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='burlywood', fill=True))
	# 	if value == 0:
	# 		ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='burlywood', fill=False, linewidth=1.5))

	# plt.ylim(-4, 4)
	# plt.xlim(-4, 4)
	# plt.axis('scaled')
	# plt.title('Current State of the Board')
	# fig.canvas.draw()
	# [p.remove() for p in reversed(ax.patches)]
	plot_pegs(env.pegs, action=action, show_action=True)
	sleep(1.5)

	reward, state, end = env.step(action)

	# ax=plt.gca()
	# for key, value in env.pegs.items():
	# 	if value == 1:
	# 		ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='burlywood', fill=True))
	# 	if value == 0:
	# 		ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='burlywood', fill=False, linewidth=1.5))

	# plt.ylim(-4, 4)
	# plt.xlim(-4, 4)
	# plt.axis('scaled')
	# plt.title('Current State of the Board')
	# fig.canvas.draw()
	# [p.remove() for p in reversed(ax.patches)]
	plot_pegs(env.pegs)
	sleep(1.5)

	i += 1


plot_pegs(env.pegs)
# for key, value in env.pegs.items():
#     if value == 1:
#         ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='burlywood', fill=True))
#     if value == 0:
#         ax.add_patch(matplotlib.patches.Circle(xy=key, radius=0.5, color='burlywood', fill=False, linewidth=1.5))

# plt.ylim(-4, 4)
# plt.xlim(-4, 4)
# plt.axis('scaled')
# plt.title('Current State of the Board')
# fig.canvas.draw()
# [p.remove() for p in reversed(ax.patches)]
sleep(3)
