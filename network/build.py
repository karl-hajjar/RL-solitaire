import tensorflow as tf
from util import softmax
from env.env import N_ACTIONS

def state_embedding(inputs, n_filters, kernel_size, strides, activation, kernel_regularizer, use_bias, bias_init_const):
	# conv
	out1 = tf.layers.conv2d(inputs=inputs, 
                       filters=n_filters,
                       kernel_size=kernel_size,
                       strides=strides, 
                       padding="same", 
                       activation=activation,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=use_bias, 
                       kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                       bias_initializer=tf.constant_initializer(bias_init_const), 
                       name="state_conv1")

	# conv
	out = tf.layers.conv2d(inputs=out1, 
                       filters=n_filters,
                       kernel_size=kernel_size,
                       strides=strides, 
                       padding="same", 
                       activation=activation,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=use_bias, 
                       kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                       bias_initializer=tf.constant_initializer(bias_init_const), 
                       name="state_conv2")	

	# conv
	out = tf.layers.conv2d(inputs=out, 
                       filters=n_filters,
                       kernel_size=kernel_size,
                       strides=strides, 
                       padding="same", 
                       activation=tf.identity,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=use_bias, 
                       kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                       bias_initializer=tf.constant_initializer(bias_init_const), 
                       name="state_conv3")

	# skip connection
	out = out1 + out
	out1 = activation(out, name="embedding_skip_connection")

	# conv
	out = tf.layers.conv2d(inputs=out1, 
                       filters=n_filters,
                       kernel_size=kernel_size,
                       strides=strides, 
                       padding="same", 
                       activation=activation,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=use_bias, 
                       kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                       bias_initializer=tf.constant_initializer(bias_init_const), 
                       name="state_conv4")

	# conv
	out = tf.layers.conv2d(inputs=out, 
                       filters=n_filters,
                       kernel_size=kernel_size,
                       strides=strides, 
                       padding="same", 
                       activation=activation,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=use_bias, 
                       kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                       bias_initializer=tf.constant_initializer(bias_init_const), 
                       name="state_conv5")

	# skip connection
	out = out1 + out
	out = activation(out, name="embedding_skip_connection")

	return out


def value_head(inputs, n_filters, activation, value_activation, kernel_regularizer, use_bias, bias_init_const, output_size):

	# 7x7 conv to output a vector
	out = tf.layers.conv2d(inputs=inputs, 
                       filters=output_size,
                       kernel_size=(7,7),
                       strides=(1,1), 
                       padding="valid", 
                       activation=activation,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=use_bias, 
                       kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                       bias_initializer=tf.constant_initializer(bias_init_const), 
                       name="value_conv")

	flat = tf.contrib.layers.flatten(out)

	out = tf.layers.dense(inputs=flat,
	                      units=output_size*2, 
	                      activation=activation,
	                      kernel_regularizer=kernel_regularizer,
	                      use_bias=use_bias,
	                      kernel_initializer=tf.contrib.layers.xavier_initializer(), 
	                      bias_initializer=tf.constant_initializer(bias_init_const), name="value_dense1")

	out = tf.layers.dense(inputs=out,
	                      units=output_size//2, 
	                      activation=activation,
	                      kernel_regularizer=kernel_regularizer,
	                      use_bias=use_bias,
	                      kernel_initializer=tf.contrib.layers.xavier_initializer(), 
	                      bias_initializer=tf.constant_initializer(bias_init_const), name="value_dense2")

	value = tf.layers.dense(inputs=out,
	                      units=1, 
	                      activation=value_activation,
	                      kernel_regularizer=kernel_regularizer,
	                      use_bias=use_bias,
	                      kernel_initializer=tf.contrib.layers.xavier_initializer(), 
	                      bias_initializer=tf.constant_initializer(bias_init_const), name="value_output")

	return value


def policy_head(inputs, n_filters, kernel_size, strides, activation, kernel_regularizer, use_bias, bias_init_const):
	# conv
	out = tf.layers.conv2d(inputs=inputs, 
                       filters=3,
                       kernel_size=(1,1),
                       strides=(1,1), 
                       padding="valid", 
                       activation=activation,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=use_bias, 
                       kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                       bias_initializer=tf.constant_initializer(bias_init_const), 
                       name="policy_conv")

	flatten = tf.contrib.layers.flatten(out)

	# dense layer
	logits = tf.layers.dense(inputs=flatten,
	                      units=N_ACTIONS, 
	                      activation=tf.identity,
	                      kernel_regularizer=kernel_regularizer,
	                      use_bias=use_bias,
	                      kernel_initializer=tf.contrib.layers.xavier_initializer(), 
	                      bias_initializer=tf.constant_initializer(bias_init_const), name="policy_logits")


	# Return like this, and then apply softmax crosse-entropy with logits, where the target vector is the action mask 
	# corresponding to the selected action. 

	return logits