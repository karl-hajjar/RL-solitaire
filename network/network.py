import numpy as np
import tensorflow as tf
import os 
from time import sleep

from .build import *
from util import flush_or_create	
from env.env import N_ACTIONS


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
		try:
			self.activation = ACTIVATION_DICT[config['activation']]
		except ValueError:
			print('activation function must be one of {"identity", "relu", "elu", "sigmoid", "tanh"}, but was "{}".'.format(config['activation']))
			raise
		try:
			self.value_activation = ACTIVATION_DICT[config['value_activation']]
		except ValueError:
			print('value activation function must be one of {"identity", "relu", "elu", "sigmoid", "tanh"}, but was "{}".'.format(config['value_activation']))
			raise
		self.state_embedding_size = config['state_embedding_size']
		self.state_channels = config["state_channels"]
		self.lr = config['lr']
		self.actor_coeff = config['actor_coeff']
		self.critic_coeff = config['critic_coeff']
		self.reg_coeff = config['reg_coeff']
		self.entropy_coeff = config['entropy_coeff']

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
				kernel_size=(3,3), 
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

			self.policy_logits = policy_head(state_emb, 
				n_filters=self.n_filters, 
				kernel_size=(3,3), 
				strides=(1,1),
				activation=self.activation, 
				kernel_regularizer=l2_regularizer,
				use_bias=self.use_bias,
				bias_init_const=self.bias_init_const)

			self.policy = tf.nn.softmax(self.policy_logits, name="policy")

		# with tf.name_scope('Entropy'):
		# 	self.entropy = tf.map_fn(entropy, self.policy, back_prop=False)

		with tf.name_scope('Targets'):
			self.value_target = tf.placeholder(tf.float32, [None, 1], name="value_target")
			self.advantage = tf.placeholder(tf.float32, [None, 1], name="advantage")
			self.action_mask = tf.placeholder(tf.float32, [None, N_ACTIONS], name="action_mask")

		with tf.name_scope('Loss'):
			self.critic_loss = tf.losses.mean_squared_error(self.value_target, self.value)
			cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.action_mask, 
																		 	logits=self.policy_logits, 
																		 	name="cross_entropy_loss")
			# adding entropy to encourage exploration
			# batch_entropies = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.ones_like(self.policy_logits)/N_ACTIONS, 
			# 													 		 logits=self.policy_logits, 
			# 													 		 name="entropy")
			batch_entropies = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.policy, 
																 		 logits=self.policy_logits, 
																 		 name="entropy")

			policy_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.ones_like(self.policy_logits), 
																 		logits=self.policy_logits, 
																 		name="entropy")

			self.true_actor_loss = tf.reduce_mean(tf.multiply(tf.squeeze(self.advantage), cross_entropy_loss)) -\
							  self.entropy_coeff * 1/N_ACTIONS * tf.reduce_mean(policy_entropy)

			self.actor_loss = tf.maximum(self.true_actor_loss, -10.0)

			# self.policy_entropy = tf.reduce_mean(batch_entropies) 
			self.true_policy_entropy = tf.reduce_mean(batch_entropies)
			clipped_batch_entropies = tf.minimum(batch_entropies, 4.0)
			clipped_policy_entropy = tf.reduce_mean(clipped_batch_entropies)
			# self.actor_loss = tf.reduce_mean(tf.multiply(tf.squeeze(self.advantage), cross_entropy_loss)) -\
			# 				  self.entropy_coeff * self.policy_entropy
			self.l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			self.loss = self.actor_coeff * self.actor_loss -\
						self.entropy_coeff * clipped_policy_entropy +\
						self.critic_coeff * self.critic_loss +\
						self.reg_coeff * self.l2_loss

		with tf.name_scope('Optimizer'):
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				optimizer = tf.train.AdamOptimizer(self.lr)
				#optimizer = tf.train.GradientDescentOptimizer(self.lr)
				grads_and_vars = optimizer.compute_gradients(self.loss)
				# grads_and_vars = optimizer.compute_gradients(self.critic_loss)
				if self.grad_clip_norm:
					gradients, variables = zip(*grads_and_vars)
					gradients, _ = tf.clip_by_global_norm(t_list=gradients,
					                                      clip_norm=self.grad_clip_norm)
					grads_and_vars = list(zip(gradients, variables))

				#self.opt_step = optimizer.minimize(self.loss, name="optim_step")
				self.opt_step = optimizer.apply_gradients(grads_and_vars, name="optim_step")
				# use apply_gradients instead ?

		self.build_summaries(grads_and_vars)


	def build_summaries(self, grads_and_vars):
		with tf.name_scope('summaries'):
			with tf.name_scope('values'):

				value = tf.summary.scalar('value', tf.reduce_mean(self.value))
				# policy_entropy = tf.summary.scalar('policy_entropy', self.entropy_coeff * self.policy_entropy)
				# policy_entropy = tf.summary.scalar('policy_entropy', self.policy_entropy)
				policy_entropy = tf.summary.scalar('policy_entropy', self.true_policy_entropy)
				value_loss = tf.summary.scalar('value_loss', self.critic_coeff * self.critic_loss)
				policy_loss = tf.summary.scalar('policy_loss', self.actor_coeff * self.true_actor_loss)
				l2_loss = tf.summary.scalar('l2_loss', self.reg_coeff * self.l2_loss)
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


	def initialize(self, checkpoint_dir):
		# init variables
		init = tf.global_variables_initializer()
		self.sess.run(init)
		# init summary writer
		self.save_checkpoint(checkpoint_dir, 0)


	def restore(self, checkpoint_path):
		saver = tf.train.Saver()
		saver.restore(self.sess, checkpoint_path)


	def optimize(self, data):
			feed_dict = {self.state : data["state"], 
						 self.value_target : data["critic_target"], 
						 self.advantage : data["advantage"],
						 self.action_mask : data["action_mask"]}
			fetches = [self.critic_loss, self.actor_loss, self.l2_loss, self.loss, self.summaries, self.opt_step]
			critic_loss, actor_loss, l2_loss, loss, summaries, _ = self.sess.run(fetches,
																				 feed_dict=feed_dict)
			# increment number of steps
			self.steps += 1
			return summaries, self.critic_coeff * critic_loss, self.actor_coeff * actor_loss, self.reg_coeff * l2_loss, loss


	def save_checkpoint(self, checkpoints_dir, it):
		# save model for keeping track and in case reboot is needed
		save_path = os.path.join(checkpoints_dir, "checkpoint_{}".format(it))
		self.saver.save(self.sess, save_path)

		# print message
		if it == 0:
			print('Initial checkpoint saved')


	
