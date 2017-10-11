import tensorflow as tf
import math

def bernoulli_dropout(incoming, keep_prob, mc, scale_during_training = True, name=None):
	""" Bernoulli Dropout.
	Outputs the input element multiplied by a random variable sampled from a Bernoulli distribution with either mean keep_prob (scale_during_training False) or mean 1 (scale_during_training True)
	Arguments:
		incoming : A `Tensor`. The incoming tensor.
		keep_prob : A float representing the probability that each element
			is kept.
		scale_during_training : A boolean value determining whether scaling is performed during training or testing
		mc : A boolean Tensor correponding to whether or not Monte-Carlo sampling will be used to calculate the networks output
		name : A name for this layer (optional).
	References:
		Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
		N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever & R. Salakhutdinov,
		(2014), Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
	Links:
	  [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf]
		(https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
	"""

	with tf.name_scope(name) as scope:

		inference = incoming

		def apply_bernoulli_dropout():
			if scale_during_training:
				return tf.nn.dropout(inference, keep_prob)
			else:
				return tf.scalar_mul(keep_prob,tf.nn.dropout(inference, keep_prob))
		
		if scale_during_training:
			expectation =  inference
		else:
			expectation =  tf.scalar_mul(keep_prob,inference)
		inference = tf.cond(mc, apply_bernoulli_dropout, lambda: expectation)
	return inference

def gaussian_dropout(incoming, keep_prob, mc, scale_during_training = True, name=None):
	""" Gaussian Dropout.
	Outputs the input element multiplied by a random variable sampled from a Gaussian distribution with mean 1 and either variance keep_prob*(1-keep_prob) (scale_during_training False) or (1-keep_prob)/keep_prob (scale_during_training True)
	Arguments:
		incoming : A `Tensor`. The incoming tensor.
		keep_prob : A float representing the probability that each element is kept by Bernoulli dropout which is used to set the variance of the Gaussian distribution.
		scale_during_training : A boolean determining whether to match the variance of the Gaussian distribution to Bernoulli dropout with scaling during testing (False) or training (True) 
		mc : A boolean Tensor correponding to whether or not Monte-Carlo sampling will be used to calculate the networks output
		name : A name for this layer (optional).
	References:
		Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
		N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever & R. Salakhutdinov,
		(2014), Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
	Links:
	  [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf]
		(https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
	"""

	with tf.name_scope(name) as scope:

		inference = incoming

		if scale_during_training:
			stddev = math.sqrt((1-keep_prob)/keep_prob)
		else:
			stddev = math.sqrt((1-keep_prob)*keep_prob)

		def apply_gaussian_dropout():
			return tf.mul(inference,tf.random_normal(tf.shape(inference), mean = 1, stddev = stddev))
		
		inference = tf.cond(mc, apply_gaussian_dropout, lambda: inference)

	return inference

def grid_conv2d(incoming,filter,strides,keep_prob,mc,padding,use_cudnn_on_gpu=None,data_format=None,name=None):

	""" Gaussian Random Independent Dropconnect 2D Convolution.
	Outputs the input element multiplied by a random variable sampled from a Gaussian distribution with mean 1 and variance that depends on the input receptive field
	Arguments:
		incoming : A `Tensor`. The incoming tensor.
		std_param : A float used to set the variance of the Gaussian random mask.
		mc : A boolean Tensor correponding to whether or not Monte-Carlo sampling will be used to calculate the networks output
		name : A name for this layer (optional).
	References:
		Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
		N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever & R. Salakhutdinov,
		(2014), Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
	Links:
	  [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf]
		(https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
	"""

	with tf.name_scope(name) as scope:

		mean = tf.nn.conv2d(incoming,filter,strides,padding,use_cudnn_on_gpu,data_format,name)
		
		def apply_gaussian_conv2d():
			stddev = math.sqrt((1-keep_prob)/keep_prob)
			std = tf.sqrt(tf.nn.conv2d(tf.multiply(incoming,incoming),tf.multiply(filter,filter),strides,padding,use_cudnn_on_gpu,data_format,name))
			noise = tf.random_normal(tf.shape(mean), mean = 0, stddev = stddev)
			return mean + std * noise 
		
		inference = tf.cond(mc, apply_gaussian_conv2d, lambda: mean)

	return inference

print('Bayesian dropout functions have been loaded.')