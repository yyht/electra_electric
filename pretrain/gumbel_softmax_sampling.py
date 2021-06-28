from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow.compat.v1 as tf
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

import configure_pretraining
from model import modeling
from model import tokenization
from pretrain import pretrain_data

def gather_positions(sequence, positions):
  """Gathers the vectors at the specific positions over a minibatch.

  Args:
    sequence: A [batch_size, seq_length] or
        [batch_size, seq_length, depth] tensor of values
    positions: A [batch_size, n_positions] tensor of indices

  Returns: A [batch_size, n_positions] or
    [batch_size, n_positions, depth] tensor of the values at the indices
  """
  shape = modeling.get_shape_list(sequence, expected_rank=[2, 3])
  depth_dimension = (len(shape) == 3)
  if depth_dimension:
    B, L, D = shape
  else:
    B, L = shape
    D = 1
    sequence = tf.expand_dims(sequence, -1)
  position_shift = tf.expand_dims(L * tf.range(B), -1)
  flat_positions = tf.reshape(positions + position_shift, [-1])
  flat_sequence = tf.reshape(sequence, [B * L, D])
  gathered = tf.gather(flat_sequence, flat_positions)
  if depth_dimension:
    return tf.reshape(gathered, [B, -1, D])
  else:
    return tf.reshape(gathered, [B, -1])

def scatter_update(sequence, updates, positions):
  """Scatter-update a sequence.

  Args:
    sequence: A [batch_size, seq_len] or [batch_size, seq_len, depth] tensor
    updates: A tensor of size batch_size*seq_len(*depth)
    positions: A [batch_size, n_positions] tensor

  Returns: A tuple of two tensors. First is a [batch_size, seq_len] or
    [batch_size, seq_len, depth] tensor of "sequence" with elements at
    "positions" replaced by the values at "updates." Updates to index 0 are
    ignored. If there are duplicated positions the update is only applied once.
    Second is a [batch_size, seq_len] mask tensor of which inputs were updated.
  """
  shape = modeling.get_shape_list(sequence, expected_rank=[2, 3])
  depth_dimension = (len(shape) == 3)
  if depth_dimension:
    B, L, D = shape
  else:
    B, L = shape
    D = 1
    sequence = tf.expand_dims(sequence, -1)
  N = modeling.get_shape_list(positions)[1]

  shift = tf.expand_dims(L * tf.range(B), -1)
  flat_positions = tf.reshape(positions + shift, [-1, 1])
  flat_updates = tf.reshape(updates, [-1, D])
  updates = tf.scatter_nd(flat_positions, flat_updates, [B * L, D])
  updates = tf.reshape(updates, [B, L, D])

  flat_updates_mask = tf.ones([B * N], tf.int32)
  updates_mask = tf.scatter_nd(flat_positions, flat_updates_mask, [B * L])
  updates_mask = tf.reshape(updates_mask, [B, L])
  not_first_token = tf.concat([tf.zeros((B, 1), tf.int32),
                               tf.ones((B, L - 1), tf.int32)], -1)
  updates_mask *= not_first_token
  updates_mask_3d = tf.expand_dims(updates_mask, -1)

  # account for duplicate positions
  if sequence.dtype == tf.float32:
    updates_mask_3d = tf.cast(updates_mask_3d, tf.float32)
    updates /= tf.maximum(1.0, updates_mask_3d)
  else:
    assert sequence.dtype == tf.int32
    updates = tf.math.floordiv(updates, tf.maximum(1, updates_mask_3d))
  updates_mask = tf.minimum(updates_mask, 1)
  updates_mask_3d = tf.minimum(updates_mask_3d, 1)

  updated_sequence = (((1 - updates_mask_3d) * sequence) +
                      (updates_mask_3d * updates))
  if not depth_dimension:
    updated_sequence = tf.squeeze(updated_sequence, -1)

  return updated_sequence, updates_mask

def sample_from_softmax(logits, logits_temp=1.0, gumbel_temp=0.1, disallow=None, straight_through=False):
  if disallow is not None:
    logits -= 1000.0 * disallow
  uniform_noise = tf.random.uniform(
      modeling.get_shape_list(logits), minval=0, maxval=1)
  logits /= logits_temp
  gumbel_noise = -tf.log(-tf.log(uniform_noise + 1e-9) + 1e-9)
  gumbel_logits = (logits + gumbel_noise) / gumbel_temp
  gumbel_probs = tf.nn.softmax(gumbel_logits)
  hard_token_ids = tf.one_hot(tf.argmax(gumbel_probs, axis=-1, output_type=tf.int32),
                    logits.shape[-1])
  if straight_through:
    gumbel_dense = tf.stop_gradient(hard_token_ids-gumbel_probs) + gumbel_probs
  else:
    gumbel_dense = gumbel_probs
  return gumbel_dense

def sample_from_top_k(logits, logits_temp=1.0, 
                  gumbel_temp=0.1, 
                  disallow=None, 
                  straight_through=False, 
                  k=20):
  print(logits, '===========')
  logits_shape = modeling.get_shape_list(logits, expected_rank=[2,3])
  depth_dimension = (len(logits_shape) == 3)
  if depth_dimension:
    reshape_logits = tf.reshape(logits, [-1, logits_shape[-1]])
  else:
    reshape_logits = logits
  print(reshape_logits, '======')
  reshape_logits_shape = modeling.get_shape_list(reshape_logits, expected_rank=[2])
  batch = reshape_logits_shape[0]
  
  values, _ = tf.nn.top_k(reshape_logits, k=k)
  min_values = values[:, -1, tf.newaxis]

  reshape_topk_logits = tf.where(
          reshape_logits < min_values,
          tf.ones_like(reshape_logits, dtype=logits.dtype) * -1e10,
          reshape_logits,
      )
  topk_logits = tf.reshape(reshape_topk_logits, logits_shape)
  if disallow is not None:
    topk_logits -= 1e10 * disallow
  uniform_noise = tf.random.uniform(modeling.get_shape_list(topk_logits), minval=0, maxval=1)
  gumbel_noise = -tf.log(-tf.log(uniform_noise + 1e-9) + 1e-9)
  topk_logits /= logits_temp
  gumbel_logits = (topk_logits + gumbel_noise) / gumbel_temp
  gumbel_probs = tf.nn.softmax(gumbel_logits)
  hard_token_ids = tf.one_hot(tf.argmax(gumbel_probs, axis=-1, output_type=tf.int32),
                    topk_logits.shape[-1])

  if straight_through:
    gumbel_dense = tf.stop_gradient(hard_token_ids-gumbel_probs) + gumbel_probs
  else:
    gumbel_dense = gumbel_probs
  return gumbel_dense

def sample_from_top_p(logits, logits_temp=1.0, 
                  gumbel_temp=0.1, 
                  disallow=None, 
                  straight_through=False,
                  p=0.95):
  """Nucleus sampling
  https://github.com/wouterkool/ancestral-gumbel-top-k-sampling
  """
  logits_shape = modeling.get_shape_list(logits, expected_rank=[2,3])
  depth_dimension = (len(logits_shape) == 3)
  if depth_dimension:
    reshape_logits = tf.reshape(logits, [-1, logits_shape[-1]])
  else:
    reshape_logits = logits
  print(reshape_logits, '======')
  reshape_logits_shape = modeling.get_shape_list(reshape_logits, expected_rank=[2])
  batch = reshape_logits_shape[0]
  sorted_logits = tf.sort(reshape_logits, direction='DESCENDING', axis=-1)
  cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
  indices = tf.stack([
      tf.range(0, batch),
      # number of indices to include
      tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
  ], axis=-1)
  min_values = tf.gather_nd(sorted_logits, indices)
  min_values = tf.expand_dims(min_values, axis=-1)
  reshape_topp_logits = tf.where(
      reshape_logits < min_values,
      tf.ones_like(reshape_logits) * -1e10,
      reshape_logits,
  )
  topp_logits = tf.reshape(reshape_topp_logits, logits_shape)
  print(topp_logits, '====topp_logits====')
  if disallow is not None:
    topp_logits -= 1e10 * disallow
  uniform_noise = tf.random.uniform(modeling.get_shape_list(topp_logits), minval=0, maxval=1)
  gumbel_noise = -tf.log(-tf.log(uniform_noise + 1e-9) + 1e-9)
  topp_logits /= logits_temp
  gumbel_logits = (topp_logits + gumbel_noise) / gumbel_temp
  gumbel_probs = tf.nn.softmax(gumbel_logits)
  hard_token_ids = tf.one_hot(tf.argmax(gumbel_probs, axis=-1, output_type=tf.int32),
                    topp_logits.shape[-1])

  if straight_through:
    gumbel_dense = tf.stop_gradient(hard_token_ids-gumbel_probs) + gumbel_probs
  else:
    gumbel_dense = gumbel_probs
  return gumbel_dense
