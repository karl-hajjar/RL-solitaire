import tensorflow as tf

def state_embedding(inputs, n_filters, kernel_size, strides, activation, kernel_regularizer, use_bias, bias_init_const):
	# conv
	out = tf.layers.conv2d(inputs=inputs, 
                       filters=n_filters//2,
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
                       name="state_conv2")	

	return out


def value_head(inputs, n_filters, activation, kernel_regularizer, use_bias, bias_init_const, output_size):

	# 7x7 conv to output a vector
	out = tf.layers.conv2d(inputs=inputs, 
                       filters=output_size,
                       kernel_size=kernel_size,
                       strides=strides, 
                       padding="same", 
                       activation=activation,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=use_bias, 
                       kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                       bias_initializer=tf.constant_initializer(bias_init_const), 
                       name="value_conv")

	flat = tf.contrib.layers.flatten(out)

	out = tf.layers.contrib.dense(inputs=flat,
	                      units=n_filters, 
	                      activation=activation,
	                      kernel_regularizer=kernel_regularizer,
	                      use_bias=use_bias,
	                      kernel_initializer=tf.contrib.layers.xavier_initializer(), 
	                      bias_initializer=tf.constant_initializer(bias_init_const), name="value_dense1")

	out = tf.layers.contrib.dense(inputs=out,
	                      units=1, 
	                      activation=tf.identity,
	                      kernel_regularizer=kernel_regularizer,
	                      use_bias=use_bias,
	                      kernel_initializer=tf.contrib.layers.xavier_initializer(), 
	                      bias_initializer=tf.constant_initializer(bias_init_const), name="value_output")

	return out


def policy_head(inputs, n_filters, kernel_size, strides, activation, kernel_regularizer, use_bias, bias_init_const):
	# conv
	out = tf.layers.conv2d(inputs=inputs, 
                       filters=n_filters//2,
                       kernel_size=kernel_size,
                       strides=strides, 
                       padding="same", 
                       activation=activation,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=use_bias, 
                       kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                       bias_initializer=tf.constant_initializer(bias_init_const), 
                       name="policy_conv1")

	out = tf.layers.conv2d(inputs=out, 
                       filters=4,
                       kernel_size=kernel_size,
                       strides=strides, 
                       padding="same", 
                       activation=tf.identity,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=use_bias, 
                       kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                       bias_initializer=tf.constant_initializer(bias_init_const), 
                       name="policy_output")

	return out

