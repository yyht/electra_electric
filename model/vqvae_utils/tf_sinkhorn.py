
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import six
# import tensorflow as tf

import tensorflow as tf
def check_tf_version():
  version = tf.__version__
  print("==tf version==", version)
  if int(version.split(".")[0]) >= 2 or int(version.split(".")[1]) >= 15:
    return True
  else:
    return False
if check_tf_version():
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()

def get_shape_list(tensor, expected_rank=None, name=None):
	"""Returns a list of the shape of tensor, preferring static dimensions.

	Args:
		tensor: A tf.Tensor object to find the shape of.
		expected_rank: (optional) int. The expected rank of `tensor`. If this is
			specified and the `tensor` has a different rank, and exception will be
			thrown.
		name: Optional name of the tensor for the error message.

	Returns:
		A list of dimensions of the shape of tensor. All static dimensions will
		be returned as python integers, and dynamic dimensions will be returned
		as tf.Tensor scalars.
	"""
	if name is None:
		name = tensor.name

	if expected_rank is not None:
		assert_rank(tensor, expected_rank, name)

	shape = tensor.shape.as_list()

	non_static_indexes = []
	for (index, dim) in enumerate(shape):
		if dim is None:
			non_static_indexes.append(index)

	if not non_static_indexes:
		return shape

	dyn_shape = tf.shape(tensor)
	for index in non_static_indexes:
		shape[index] = dyn_shape[index]
	return shape


def reshape_to_matrix(input_tensor):
	"""Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
	ndims = input_tensor.shape.ndims
	if ndims < 2:
		raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
										 (input_tensor.shape))
	if ndims == 2:
		return input_tensor

	width = input_tensor.shape[-1]
	output_tensor = tf.reshape(input_tensor, [-1, width])
	return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
	"""Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
	if len(orig_shape_list) == 2:
		return output_tensor

	output_shape = get_shape_list(output_tensor)

	orig_dims = orig_shape_list[0:-1]
	width = output_shape[-1]

	return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
	"""Raises an exception if the tensor rank is not of the expected rank.

	Args:
		tensor: A tf.Tensor to check the rank of.
		expected_rank: Python integer or list of integers, expected rank.
		name: Optional name of the tensor for the error message.

	Raises:
		ValueError: If the expected shape doesn't match the actual shape.
	"""
	if name is None:
		name = tensor.name

	expected_rank_dict = {}
	if isinstance(expected_rank, six.integer_types):
		expected_rank_dict[expected_rank] = True
	else:
		for x in expected_rank:
			expected_rank_dict[x] = True

	actual_rank = tensor.shape.ndims
	if actual_rank not in expected_rank_dict:
		scope_name = tf.get_variable_scope().name
		raise ValueError(
				"For the tensor `%s` in scope `%s`, the actual rank "
				"`%d` (shape = %s) is not equal to the expected rank `%s`" %
				(name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def cost_matrix(x,y, x_weight, y_weight, p=2, distance_type='wmd'):
	"Returns the cost matrix C_{ij}=|x_i - y_j|^p"
	# [batch_size, seq, dim]
	x_shape_list = get_shape_list(x, expected_rank=3)
	y_shape_list = get_shape_list(y, expected_rank=3)
	x_col = tf.expand_dims(x, axis=2)
	y_lin = tf.expand_dims(y, axis=1)
	x_weight = tf.expand_dims(x_weight, axis=2)
	y_weight = tf.expand_dims(y_weight, axis=1)
	if distance_type == 'wmd':
		c = tf.sqrt(tf.reduce_sum((tf.abs(x_col-y_lin))**p,axis=-1))
	elif distance_type == 'wrd':
		c = (tf.reduce_sum((tf.abs(x_col-y_lin))**p,axis=-1)) / 2.0
	c_weight = (x_weight * y_weight)
	c = c * c_weight
	c += (1.0-c_weight)*1000.0
	return c, c_weight

def sinkhorn_loss(x, y, x_weight, y_weight, epsilon, 
								numItermax=10, stopThr=1e-9, p=2, distance_type='wmd'):
	"""
	Given two emprical measures with n points each with locations x and y
	outputs an approximation of the OT cost with regularization parameter epsilon
	niter is the max. number of steps in sinkhorn loop
	
	Inputs:
			x,y:  The input sets representing the empirical measures.  Each are a tensor of shape (n,D)
			epsilon:  The entropy weighting factor in the sinkhorn distance, epsilon -> 0 gets closer to the true wasserstein distance
			n:  The number of support points in the empirical measures
			niter:  The number of iterations in the sinkhorn algorithm, more iterations yields a more accurate estimate
	Outputs:
	
	"""
	x_shape_list = get_shape_list(x, expected_rank=3)
	x_batch_size = x_shape_list[0]
	x_seq_num = x_shape_list[1]
	x_hidden_dims = x_shape_list[2]

	y_shape_list = get_shape_list(y, expected_rank=3)
	y_batch_size = y_shape_list[0]
	y_seq_num = y_shape_list[1]
	y_hidden_dims = y_shape_list[2]
	
	# both marginals are fixed with equal weights
	if x_weight is not None and y_weight is not None:
		x_weight = tf.cast(x_weight, dtype=tf.float32)
		y_weight = tf.cast(y_weight, dtype=tf.float32)
	else:
		x_weight = tf.ones((x_batch_size, x_seq_num), dtype=tf.float32)
		y_weight = tf.ones((y_batch_size, y_seq_num), dtype=tf.float32)
	
	if distance_type == 'wrd':
	
		x_norm = tf.sqrt(tf.reduce_sum(tf.pow(x, 2.0), axis=-1, keepdims=True))
		y_norm = tf.sqrt(tf.reduce_sum(tf.pow(y, 2.0), axis=-1, keepdims=True))
		x_normed = x / (x_norm+1e-10)
		y_normed = y / (y_norm+1e-10)

		# The Sinkhorn algorithm takes as input three variables :

		cost, cost_weight = cost_matrix(x_normed, y_normed, x_weight, y_weight, 
																		p=p, distance_type=distance_type)  # Wasserstein cost function
		# mu_weight = tf.expand_dims(x_weight, axis=-1)
		# mu = x_norm * mu_weight / tf.reduce_sum(x_norm*mu_weight, axis=-2, keepdims=True)
		# nu_weight = tf.expand_dims(y_weight, axis=-1)
		# nu = y_norm * nu_weight / tf.reduce_sum(y_norm*nu_weight, axis=-2, keepdims=True)
		# mu = tf.squeeze(mu, axis=-1)
		# nu = tf.squeeze(nu, axis=-1)

		mu = x_weight / tf.reduce_sum(x_weight, axis=-1, keepdims=True)
		nu = y_weight / tf.reduce_sum(y_weight, axis=-1, keepdims=True)

	elif distance_type == 'wmd':
			
		cost, cost_weight = cost_matrix(x, y, x_weight, y_weight, 
																		p=p, distance_type=distance_type)  # Wasserstein cost function
		mu = x_weight / tf.reduce_sum(x_weight, axis=-1, keep_dims=True) # [batch_size, seq]
		nu = y_weight / tf.reduce_sum(y_weight, axis=-1, keep_dims=True) # [batch_size, seq]

	u = tf.zeros_like(mu)
	v = tf.zeros_like(nu)

	# To check if algorithm terminates because of threshold
	cpt = tf.constant(0)
	err = tf.constant(1.0)

	c = lambda cpt, u, v, err: tf.logical_and(cpt < numItermax, err > stopThr)

	# Elementary operations
	def M(cost, u, v, cost_weight=None):
		"Modified cost for logarithmic updates"
		"$M_{ij} = (-c_{ij} + u_i + v_j) / \\epsilon$"
		return (-cost + tf.expand_dims(u, -1) + tf.expand_dims(v, -2) )/epsilon

	def loop_func(cpt, u, v, err):
		u1 = tf.identity(u)
		u = epsilon * x_weight * (tf.log(mu+1e-10) - tf.reduce_logsumexp(M(cost, u, v, cost_weight), axis=-1)) + u
		v = epsilon * y_weight * (tf.log(nu+1e-10) - tf.reduce_logsumexp(tf.transpose(M(cost, u, v, cost_weight), perm=(0, 2, 1)), axis=-1)) + v
		
		err = tf.reduce_mean(tf.reduce_sum(tf.abs(u - u1), axis=-1))

		cpt = tf.add(cpt, 1)
		return cpt, u, v, err

	_, u, v, _ = tf.while_loop(c, loop_func, loop_vars=[cpt, u, v, err])
	u_final, v_final = u, v
	pi = tf.exp(M(cost, u_final, v_final, cost_weight=None))
	# Sinkhorn distance
	# ignore the gradient flow through \pi
	pi = tf.stop_gradient(pi)
	final_cost = tf.reduce_sum(pi * cost * cost_weight, axis=(-2, -1))
	return final_cost