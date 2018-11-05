import yaml 
import tensorflow as tf
import numpy as np 
import os
import shutil
import glob
import math

def read_config(path):
	with open(path, 'r') as stream:
	    try:
	        config = yaml.load(stream)
	    except yaml.YAMLError as exc:
	        print(exc)

	return config


def flush_or_create(directory):
	if os.path.exists(directory):
		# if directory for that name already exists, flush it
		for root, dirs, files in os.walk(directory):
			for f in files:
				os.unlink(os.path.join(root, f))
			for d in dirs:
				shutil.rmtree(os.path.join(root, d))
		files = glob.glob(os.path.join(directory, "*"))
	else:
		os.makedirs(directory)	


def get_latest_checkpoint(path):
	with open(path) as checkpoints:
		latest_checkpoint = len(checkpoints.readlines()) - 2 # -1 for header, and -1 since checkpoint 0 is included
	return latest_checkpoint


def mask_out(policy, feasible_actions, grid):
	mask = np.zeros_like(policy)
	for i, (x,y) in enumerate(grid):
		mask[3-y, x+3, feasible_actions[i]] = policy[3-y, x+3, feasible_actions[i]]
	return mask


def softmax(target, axis=None, name=None):
	with tf.name_scope(name, 'softmax', values=[target]):
	    max_axis = tf.reduce_max(target, axis, keep_dims=True)
	    target_exp = tf.exp(target-max_axis)
	    normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
	    softmax = target_exp / normalize
	    return softmax


def entropy(target, axis=None, name=None):
	with tf.name_scope(name, 'softmax', values=[target]):
		return -tf.reduce_sum(target * tf.log(target), axis=axis)


def rot_pos(pos,angle):
    x,y = pos
    cos = np.cos(angle)
    sin = np.sin(angle)
    xr = x*cos - y*sin
    yr = y*cos + x*sin
    return (int(round(xr)), int(round(yr)))


def rotate_state_action(state, action):
	Pi = math.pi
	ROTATED_ACTIONS = np.array([[0,1,2,3], [3,2,0,1], [1,0,3,2], [2,3,1,0]], dtype=np.int)
	rotated_state = np.zeros_like(state)
	rotated_state[:,:,1:] = state[:,:,1:]
	angle_index = np.random.randint(0,4) 
	#angle_index = 1
	angle = angle_index * Pi/2
	for i in range(7):
		for j in range(7):
			x,y = rot_pos((j-3,3-i), angle)
			rotated_state[3-y,x+3,0] = state[i,j,0]
	rotated_move_id = ROTATED_ACTIONS[angle_index, action[2]]
	xr, yr = rot_pos((action[1]-3, 3-action[0]), angle)
	return rotated_state, [3-yr, xr+3, rotated_move_id]


