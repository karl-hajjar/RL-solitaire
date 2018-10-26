import numpy as np
import tensorflow as tf
from build import *


ACTIVATION_DIC = dict({"identity" : tf.identity,
					   "relu" : tf.nn.relu,
					   "elu" : tf.nn.elu,
					   "sigmoid" : tf.nn.sigmoid
   					   "tanh" : tf.nn.tanh})

class Net(object):
	"""docstring for Net"""

	def __init__(self, config):
		self.name = config['name']
		self.use_bias = config['use_bias']
		self.bias_init_const = config['bias_init_const']
		self.grad_clip_norm = config['grad_clip_norm']
		self.n_filters = config['n_filters']
		self.embedding_lr = config['embedding_lr']
		self.actor_lr = config['actor_lr']
		self.critic_lr = config['critic_lr']

		self.session = tf.Session()
		


	def build():
		pass



	def initialize():
		pass

