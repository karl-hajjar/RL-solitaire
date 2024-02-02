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
        # latest_checkpoint = len(checkpoints.readlines()) - 2 # -1 for header, and -1 since checkpoint 0 is included
        first_line = checkpoints.readline()
        checkpoint_index = int(first_line.split(' ')[1].split('"')[1].split('_')[1])
    # return latest_checkpoint
    return checkpoint_index


def mask_out(policy, feasible_actions, grid):
    mask = np.zeros_like(policy)
    for i in range(len(grid)):
        feasible_i = 4 * i + np.argwhere(feasible_actions[i])
        mask[feasible_i] = policy[feasible_i]
    return mask


# def softmax(target, axis=None, name=None):
# 	with tf.name_scope(name, 'softmax', values=[target]):
# 	    max_axis = tf.reduce_max(target, axis, keep_dims=True)
# 	    target_exp = tf.exp(target-max_axis)
# 	    normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
# 	    softmax = target_exp / normalize
# 	    return softmax


# def entropy(target, axis=None, name=None):
# 	with tf.name_scope(name, 'softmax', values=[target]):
# 		return -tf.reduce_sum(target * tf.log(target), axis=axis)
