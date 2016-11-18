import tensorflow as tf
import math

def init_training_mode():
	"""  init_training_mode.
	Creates `is_training` variable and its ops if they haven't be created
	yet. This op is required if you are using layers such as dropout or
	batch normalization independently of TFLearn models (DNN or Trainer class).
	!From the TFLearn source code (https://github.com/tflearn)!
	"""
	# 'is_training' collection stores the training mode variable
	coll = tf.get_collection('is_training')
	if len(coll) == 0:
		tr_var = tf.get_variable(
			"is_training", dtype=tf.bool, shape=[],
			initializer=tf.constant_initializer(False),
			trainable=False)
		tf.add_to_collection('is_training', tr_var)
		# 'is_training_ops' stores the ops to update training mode variable
		a = tf.assign(tr_var, True)
		b = tf.assign(tr_var, False)
		tf.add_to_collection('is_training_ops', a)
		tf.add_to_collection('is_training_ops', b)

def get_training_mode():
	""" get_training_mode.
	Returns variable in-use to set training mode.
	Returns:
		A `Variable`, the training mode holder.
	!From the TFLearn source code (https://github.com/tflearn)!
	"""
	init_training_mode()
	coll = tf.get_collection('is_training')
	return coll[0]

def bernoulli_dropout(incoming, keep_prob, scale_during_training = True, mc = False, name="BernoulliDropout"):
	""" Bernoulli Dropout.
	Outputs the input element multiplied by a random variable sampled from a Bernoulli distribution with either mean keep_prob (scale_during_training False) or mean 1 (scale_during_training True)
	Arguments:
		incoming : A `Tensor`. The incoming tensor.
		keep_prob : A float representing the probability that each element
			is kept.
		scale_during_training : A boolean value determining whether scaling is performed during training or testing
		mc : A boolean value correponding to whether or not Monte-Carlo sampling will be used to calculate the networks output
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

		if mc:
			inference = apply_bernoulli_dropout()
		else:
			is_training = get_training_mode()
			if scale_during_training:
				inference = tf.cond(is_training, apply_bernoulli_dropout, lambda: inference)
			else:
				inference = tf.cond(is_training, apply_bernoulli_dropout, lambda: tf.scalar_mul(keep_prob,inference))
				
	return inference

def gaussian_dropout(incoming, keep_prob, scale_during_training = True, mc = False, name="GaussianDropout"):
	""" Gaussian Dropout.
	Outputs the input element multiplied by a random variable sampled from a Gaussian distribution with mean 1 and either variance keep_prob*(1-keep_prob) (scale_during_training False) or (1-keep_prob)/keep_prob (scale_during_training True)
	Arguments:
		incoming : A `Tensor`. The incoming tensor.
		keep_prob : A float representing the probability that each element is kept by Bernoulli dropout which is used to set the variance of the Gaussian distribution.
		scale_during_training : A boolean determining whether to match the variance of the Gaussian distribution to Bernoulli dropout with scaling during testing (False) or training (True) 
		mc : A boolean value correponding to whether or not Monte-Carlo sampling will be used to calculate the networks output
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
		
		if mc:
			inference = apply_gaussian_dropout()
		else:
			is_training = get_training_mode()
			inference = tf.cond(is_training, apply_gaussian_dropout, lambda: inference)

	return inference