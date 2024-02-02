import numpy as np
import math

ROTATED_ACTIONS = np.array([[0, 1, 2, 3], [3, 2, 0, 1], [1, 0, 3, 2], [2, 3, 1, 0]], dtype=np.int)


def rot_pos(pos, angle):
    x, y = pos
    cos = np.cos(angle)
    sin = np.sin(angle)
    xr = x * cos - y * sin
    yr = y * cos + x * sin
    return (int(round(xr)), int(round(yr)))


def rotate_state_action(state, action):
    Pi = math.pi
    rotated_state = np.zeros_like(state)
    rotated_state[:, :, 1:] = state[:, :, 1:]
    angle_index = np.random.randint(0, 4)
    # angle_index = 1
    angle = angle_index * Pi / 2
    for i in range(7):
        for j in range(7):
            x, y = rot_pos((j - 3, 3 - i), angle)
            rotated_state[3 - y, x + 3, 0] = state[i, j, 0]
    rotated_move_id = ROTATED_ACTIONS[angle_index, action[2]]
    xr, yr = rot_pos((action[1] - 3, 3 - action[0]), angle)
    return rotated_state, [3 - yr, xr + 3, rotated_move_id]
