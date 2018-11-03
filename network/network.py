import numpy as np
import tensorflow as tf
import os 
from time import sleep

from build import *
from util import entropy, flush_or_create


ACTIVATION_DICT = dict({"identity" : tf.identity,
					    "relu" : tf.nn.relu,
					    "elu" : tf.nn.elu,
					    "sigmoid" : tf.nn.sigmoid,
   					    "tanh" : tf.nn.tanh})

class Net(object):
	"""A class implementing a policy-value network for the Actor-Critic method"""

	def __init__(self, config):
		self.name = config['name']
		self.use_bias = config['use_bias']
		self.bias_init_const = config['bias_init_const']
		self.grad_clip_norm = config['grad_clip_norm']
		self.n_filters = config['n_filters']
		self.activation = ACTIVATION_DICT[config['activation']]
		self.value_activation = ACTIVATION_DICT[config['value_activation']]
		self.state_embedding_size = config['state_embedding_size']
		self.state_channels = config["state_channels"]
		self.lr = config['lr']
		self.actor_coeff = config['actor_coeff']
		self.critic_coeff = config['critic_coeff']
		self.reg_coeff = config['reg_coeff']

		# init graph and session 
		tf.reset_default_graph()
		config = tf.ConfigProto()
		config.allow_soft_placement = True
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)

		# track number of optim steps
		self.steps = 0


	def build(self):
		# init regularizer
		l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_coeff)

		with tf.name_scope('Inputs'):
			self.state = tf.stop_gradient(tf.placeholder(dtype=tf.float32, shape=(None, 7, 7, self.state_channels), name='State'))

		with tf.name_scope('Embedding'):
			state_emb = state_embedding(self.state, 
				n_filters=self.n_filters, 
				kernel_size=(2,2), 
				strides=(1,1),
				activation=self.activation, 
				kernel_regularizer=l2_regularizer,
				use_bias=self.use_bias,
				bias_init_const=self.bias_init_const)

		with tf.name_scope('Outputs'):
			self.value = value_head(state_emb, 
				n_filters=self.n_filters, 
				activation=self.activation,
				value_activation=self.value_activation, 
				kernel_regularizer=l2_regularizer,
				use_bias=self.use_bias,
				bias_init_const=self.bias_init_const,
				output_size=self.state_embedding_size)

			self.policy = policy_head(state_emb, 
						n_filters=self.n_filters, 
						kernel_size=(2,2), 
						strides=(1,1),
						activation=self.activation, 
						kernel_regularizer=l2_regularizer,
						use_bias=self.use_bias,
						bias_init_const=self.bias_init_const)

		with tf.name_scope('Entropy'):
			self.entropy = tf.map_fn(entropy, self.policy, back_prop=False)

		with tf.name_scope('Targets'):
			self.value_target = tf.placeholder(tf.float32, [None, 1], name="value_target")
			self.advantage = tf.placeholder(tf.float32, [None, 1], name="advantage")
			self.action_mask = tf.placeholder(tf.bool, [None, 7, 7, 4], name="action_mask")

		with tf.name_scope('Loss'):
			self.critic_loss = tf.losses.mean_squared_error(self.value_target, self.value)
			action_proba = tf.boolean_mask(self.policy, self.action_mask, name="action_proba")
			self.actor_loss = - tf.reduce_mean(tf.multiply(self.advantage, tf.log(action_proba, name="log_action_prob")))
			self.l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			self.loss = 0.*self.actor_coeff * self.actor_loss +\
						self.critic_coeff * self.critic_loss +\
						0.*self.reg_coeff * self.l2_loss

		with tf.name_scope('Optimizer'):
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				optimizer = tf.train.AdamOptimizer(self.lr)
				grads_and_vars = optimizer.compute_gradients(self.loss)
				if self.grad_clip_norm:
				    gradients, variables = zip(*grads_and_vars)
				    gradients, _ = tf.clip_by_global_norm(t_list=gradients,
				                                          clip_norm=self.grad_clip_norm)
				    grads_and_vars = list(zip(gradients, variables))

				self.opt_step = optimizer.minimize(self.loss, name="optim_step")
		self.build_summaries(grads_and_vars)


	def build_summaries(self, grads_and_vars):
		# define placeholders for summaries
		with tf.name_scope('summaries'):
			with tf.name_scope('values'):

				value = tf.summary.scalar('value', tf.reduce_mean(self.value))
				policy_entropy = tf.summary.scalar('policy_entropy', tf.reduce_mean(self.entropy))
				value_loss = tf.summary.scalar('value_loss', self.critic_loss)
				policy_loss = tf.summary.scalar('policy_loss', self.actor_loss)
				l2_loss = tf.summary.scalar('l2_loss', self.l2_loss)
				loss = tf.summary.scalar('loss', self.loss)

			for grad, var in grads_and_vars:
				with tf.name_scope('gradients'):
					try:
						tf.summary.histogram(var.name, grad)
					except:
						print('var {} failed'.format(var.name))
				with tf.name_scope('variables'):
					tf.summary.histogram(var.name, var)

			self.summaries = tf.summary.merge_all()


	def get_policy(self, state):
		policy = self.sess.run(self.policy, feed_dict={self.state : state})
		return policy[0]


	def get_value(self, state):
		value = self.sess.run(self.value, feed_dict={self.state : state})
		return value		


	def initialize(self, checkpoint_dir, tb_log_dir):
		# init variables
		init = tf.global_variables_initializer()
		self.sess.run(init)
		# init summary writer
		self.summary_writer = tf.summary.FileWriter(tb_log_dir, self.sess.graph)
		# saver to save (and later restore) model checkpoints
		self.saver = tf.train.Saver(max_to_keep=500)
		self.save_checkpoint(checkpoint_dir, 0)


	def optimize(self, data):
			feed_dict = {self.state : data["state"], 
						 self.value_target : data["critic_target"], 
						 self.advantage : data["advantage"],
						 self.action_mask : data["action_mask"]}
			fetches = [self.value, self.policy, self.critic_loss, self.actor_loss, self.l2_loss, self.loss, self.summaries, self.opt_step]
			value, policy, critic_loss, actor_loss, l2_loss, loss, summaries, _ = self.sess.run(fetches,
																					 feed_dict=feed_dict)

			# increment number of steps
			self.steps += 1
			return summaries, critic_loss, actor_loss, l2_loss, loss


	def save_checkpoint(self, checkpoints_dir, it):
		# save model for keeping track and in case reboot is needed
		save_path = os.path.join(checkpoints_dir, "checkpoint_{}".format(it))
		self.saver.save(self.sess, save_path)

		# print message
		if it == 0:
			print('Initial checkpoint saved')


	
