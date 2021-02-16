
import tensorflow as tf
import collections
import copy
import json
import math
import re
import six
import tensorflow as tf

def iterative_inv(mat, n_iter=6):

  """
  https://downloads.hindawi.com/journals/aaa/2014/563787.pdf
  A New Iterative Method for Finding Approximate Inverses of
  Complex Matrices
  """

  mat_shape = get_shape_list(mat, expected_rank=[2,3,4])
  I = tf.cast(tf.eye(mat_shape[-1]), dtype=tf.float32)
  K = tf.identity(mat) 
  # [B, N, n-landmarks, n-landmarks]
  V = 1 / (tf.reduce_max(tf.reduce_sum(tf.abs(K), axis=-2)) * tf.reduce_max(tf.reduce_sum(tf.abs(K), axis=-1))) * tf.transpose(K, [0,1,3,2])

  for _ in range(n_iter):
      KV = tf.matmul(K, V)
      V = tf.matmul(0.25 * V, 13 * I - tf.matmul(KV, 15 * I - tf.matmul(KV, 7 * I - KV)))
  # [B, N, n-landmarks, n-landmarks]
  # V = tf.stop_gradient(V)
  return V

def iterative_inv_v1(mat, n_iter=6):
  """
  https://downloads.hindawi.com/journals/aaa/2014/563787.pdf
  A New Iterative Method for Finding Approximate Inverses of
  Complex Matrices
  """

  mat_shape = get_shape_list(mat, expected_rank=[2,3,4])
  I = tf.cast(tf.eye(mat_shape[-1]), dtype=tf.float32)
  K = tf.identity(mat) 
  # [B, N, n-landmarks, n-landmarks]
  V = 1 / (tf.reduce_max(tf.reduce_sum(tf.abs(K), axis=-2)) * tf.reduce_max(tf.reduce_sum(tf.abs(K), axis=-1))) * tf.transpose(K, [0,1,3,2])

  cpt = tf.constant(0)

  c = lambda cpt, V: tf.logical_and(cpt < n_iter, cpt < n_iter)

  def loop_func(cpt, V):
    V_old = tf.identity(V)
    KV = tf.matmul(K, V)
    V = tf.matmul(0.25 * V, 13 * I - tf.matmul(KV, 15 * I - tf.matmul(KV, 7 * I - KV)))
    
    cpt = tf.add(cpt, 1)
    return cpt, V

  _, V_final = tf.while_loop(c, loop_func, loop_vars=[cpt, V])
  # V_final = tf.stop_gradient(V_final)
  return V_final

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
