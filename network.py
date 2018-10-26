import numpy as np
import tensorflow as tf
from build import *


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
		self.state_embedding_size = config['state_embedding_size']
		self.embedding_lr = config['embedding_lr']
		self.actor_lr = config['actor_lr']
		self.critic_lr = config['critic_lr']
		self.actor_coeff = config['actor_loss_coeff']
		self.critic_coeff = config['critic_loss_coeff']
		self.reg_coeff = config['reg_loss_coeff']

		# init graph and session 
		tf.reset_default_graph()
		config = tf.ConfigProto()
		config.allow_soft_placement = True
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		

	def build(self):
		# init regularizer
		l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.coeff_reg_l2)

		with tf.name_scope('Inputs'):
			state = tf.stop_gradient(tf.Placeholder(dtype=tf.int8, shape=(None, 7, 7), name='State'))

		with tf.name_scope('Embedding'):
			state_emb = state_embedding(state, 
				n_filters=self.n_filters, 
				kernel_size=(2,2), 
				strides=(1,1),
				activation=self.activation, 
				kernel_regularizer=l2_regularizer,
				use_bias=self.use_bias,
				bias_init_const=self.bias_init_const)

		with tf.name_scope('Outputs'):
			self.value = self.get_value(state_emb)
			self.policy = self.get_policy(state_emb)

		with tf.name_scope('Targets'):
			self.value_target = tf.placeholder(tf.float32, [None, 1], name="value target")
			self.advantage = tf.placeholder(tf.float32, [None, 1], name="advantage")
			self.action_prob = tf.placeholder(tf.float32, [None, 1], name="action prob")

		with tf.name_scope('Loss'):
			self.critic_loss = tf.losses.mean_squared_error(self.value_target, self.value)
			self.actor_loss = - self.advantage * tf.math.log(self.action_prob, name="log action prob")
			self.l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			self.loss = self.actor_coeff * self.actor_loss +\
						self.critic_coeff * self.critic_loss +\
						self.reg_coeff * self.l2_loss

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

				self.opt_step = optimizer.minimize(self.loss, name="optim step")
		#self.build_summaries(grads_and_vars)



	def get_policy(self, state):
		# maybe use a session here to return the policy ? 

		# with tf.Session() as sess:
		# 	policy_logits = policy_head(state, 
		# 				n_filters=self.n_filters, 
		# 				kernel_size=(2,2), 
		# 				strides=(1,1),
		# 				activation=self.activation, 
		# 				kernel_regularizer=l2_regularizer,
		# 				use_bias=self.use_bias,
		# 				bias_init_const=self.bias_init_const,
		# 				output_size=self.state_embedding_size)

		#	policy = tf.nn.softmax(policy_logits)

		# sess.run(tf.global_variable_initializer())
		# policy = sess.run(policy)

		policy = policy_head(state, 
			n_filters=self.n_filters, 
			kernel_size=(2,2), 
			strides=(1,1),
			activation=self.activation, 
			kernel_regularizer=l2_regularizer,
			use_bias=self.use_bias,
			bias_init_const=self.bias_init_const)
		return policy


	def get_value(self, state):
		value = value_head(state, 
			n_filters=self.n_filters, 
			activation=self.activation, 
			kernel_regularizer=l2_regularizer,
			use_bias=self.use_bias,
			bias_init_const=self.bias_init_const,
			output_size=self.state_embedding_size)


	def initialize(self):
		initializer = tf.global_variable_initializer()
		self.session.run(initializer)

	def optimize(self, data):
			for d in data:
			feed_dict = {}

			value, policy, critic_loss, actor_loss, l2_loss, loss, _ = self.sess.run([self.value, self.policy, ..., self.opt_step])

			# track in tensorboard
