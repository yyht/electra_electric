from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re

import numpy as np
import six
import tensorflow as tf

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
  if isinstance(tensor, np.ndarray) or isinstance(tensor, list):
    shape = np.array(tensor).shape
    if isinstance(expected_rank, six.integer_types):
      assert len(shape) == expected_rank
    elif expected_rank is not None:
      assert len(shape) in expected_rank
    return shape

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
  # output_shape = tf.shape(output_tensor)

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

def _to_term_frequency(x, vocab_size):
  """Creates a SparseTensor of term frequency for every doc/term pair.
  Args:
    x : a SparseTensor of int64 representing string indices in vocab.
    vocab_size: A scalar int64 Tensor - the count of vocab used to turn the
        string into int64s including any OOV buckets.
  Returns:
    a SparseTensor with the count of times a term appears in a document at
        indices <doc_index_in_batch>, <term_index_in_vocab>,
        with size (num_docs_in_batch, vocab_size).
  """
  # Construct intermediary sparse tensor with indices
  # [<doc>, <term_index_in_doc>, <vocab_id>] and tf.ones values.
  vocab_size = tf.convert_to_tensor(value=vocab_size, dtype=tf.int64)
  split_indices = tf.cast(
      tf.split(x.indices, axis=1, num_or_size_splits=2), dtype=tf.int64)
  expanded_values = tf.cast(tf.expand_dims(x.values, 1), dtype=tf.int64)
  next_index = tf.concat(
      [split_indices[0], split_indices[1], expanded_values], axis=1)

  next_values = tf.ones_like(x.values)
  expanded_vocab_size = tf.expand_dims(vocab_size, 0)
  next_shape = tf.concat(
      [x.dense_shape, expanded_vocab_size], 0)

  next_tensor = tf.SparseTensor(
      indices=tf.cast(next_index, dtype=tf.int64),
      values=next_values,
      dense_shape=next_shape)

  # Take the intermediary tensor and reduce over the term_index_in_doc
  # dimension. This produces a tensor with indices [<doc_id>, <term_id>]
  # and values [count_of_term_in_doc] and shape batch x vocab_size
  term_count_per_doc = tf.sparse_reduce_sum_sparse(next_tensor, 1)

  dense_doc_sizes = tf.cast(
      tf.sparse.reduce_sum(
          tf.SparseTensor(
              indices=x.indices,
              values=tf.ones_like(x.values),
              dense_shape=x.dense_shape), 1),
      dtype=tf.float64)

  gather_indices = term_count_per_doc.indices[:, 0]
  gathered_doc_sizes = tf.gather(dense_doc_sizes, gather_indices)

  term_frequency = (
      tf.cast(term_count_per_doc.values, dtype=tf.float64) /
      tf.cast(gathered_doc_sizes, dtype=tf.float64))
  term_count = tf.cast(term_count_per_doc.values, dtype=tf.float64)

  sparse_term_freq = tf.SparseTensor(
              indices=term_count_per_doc.indices,
              values=term_frequency,
              dense_shape=term_count_per_doc.dense_shape)

  sparse_term_count = tf.SparseTensor(
              indices=term_count_per_doc.indices,
              values=term_count,
              dense_shape=term_count_per_doc.dense_shape)

  return sparse_term_freq, sparse_term_count

def _to_sparse(x):
  tensor_shape = get_shape_list(x, expected_rank=[1,2])
  idx = tf.where(tf.not_equal(x, 0))
  # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape() if tensor shape is dynamic
  sparse = tf.SparseTensor(idx, tf.gather_nd(x, idx), tensor_shape)
  return sparse

def _to_vocab_range(x, vocab_size):
  """Enforces that the vocab_ids in x are positive."""
  output = tf.SparseTensor(
      indices=x.indices,
      values=tf.mod(x.values, vocab_size),
      dense_shape=x.dense_shape)
  return output

def sparse_idf2dense(sparse_term_freq, sparse_term_count):
  dense_term_freq = tf.sparse.to_dense(sparse_term_freq)
  dense_term_count = tf.sparse.to_dense(sparse_term_count)
  return dense_term_freq, dense_term_count

def tokenid2tf(input_ids, vocab_size, **kargs):
  sparse_input_ids = _to_sparse(input_ids)
  cleaned_input = _to_vocab_range(sparse_input_ids, vocab_size)
  [sparse_term_freq, 
  sparse_term_count] = _to_term_frequency(cleaned_input, 
                        vocab_size)
  
  [term_freq,
  term_count] = sparse_idf2dense(sparse_term_freq, 
                    sparse_term_count)

  term_binary = tf.minimum(term_count, 1)
  term_freq = tf.cast(term_freq, dtype=tf.float32)
  term_binary = tf.cast(term_binary, dtype=tf.float32)
  term_count = tf.cast(term_count, dtype=tf.float32)
  return term_count, term_binary, term_freq

def tokenid2tf_tpu(input_ids, vocab_size, **kargs):
  input_mask = tf.cast(tf.not_equal(input_ids, kargs.get('[PAD]', 0)), 
            tf.float32)
  input_mask = tf.expand_dims(input_mask, axis=-1)
  # one_hot_input_ids = tf.one_hot(input_ids, depth=vocab_size)
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])
  input_shape = get_shape_list(input_ids)
  flat_input_ids = tf.reshape(input_ids, [-1])
  one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
  one_hot_input_ids = tf.reshape(one_hot_input_ids,
        input_shape[0:-1] + [input_shape[-1] * vocab_size])

  # [batch, seq, vocab_size]
  output = tf.cast(one_hot_input_ids, tf.float32) * input_mask
  # [batch, vocab_size]
  term_count = tf.reduce_sum(output, axis=1)
  # [batch, vocab_size]
  term_binary = tf.minimum(tf.reduce_sum(output, 1), 1)
  term_freq = tf.reduce_sum(output, axis=1) / (1e-10+tf.reduce_sum(output, axis=(1, 2), keepdims=False))
  return term_count, term_binary, term_freq