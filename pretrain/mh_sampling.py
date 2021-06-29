# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for pre-training. These mainly deal with the gathering and
scattering needed so the generator only makes predictions for the small number
of masked tokens.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow.compat.v1 as tf
# import tensorflow as tf

import tensorflow as tf
tf.disable_v2_behavior()

def check_tf_version():
  version = tf.__version__
  print("==tf version==", version)
  if int(version.split(".")[0]) >= 2 or int(version.split(".")[1]) >= 15:
    return True
  else:
    return False
# if check_tf_version():
#   import tensorflow.compat.v1 as tf
#   tf.disable_v2_behavior()

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


VOCAB_MAPPING = {}


def get_vocab(config):
  """Memoized load of the vocab file."""
  if config.vocab_file not in VOCAB_MAPPING:
    vocab = tokenization.FullTokenizer(
        config.vocab_file, do_lower_case=True).vocab
    VOCAB_MAPPING[config.vocab_file] = vocab
  return VOCAB_MAPPING[config.vocab_file]


def get_candidates_mask(config,
                        inputs,
                        disallow_from_mask=None):
  """Returns a mask tensor of positions in the input that can be masked out."""
  vocab = get_vocab(config)
  ignore_ids = [vocab["[SEP]"], vocab["[CLS]"], vocab["[MASK]"], vocab["[UNK]"]]
  candidates_mask = tf.ones_like(inputs.input_ids, tf.bool)
  for ignore_id in ignore_ids:
    candidates_mask &= tf.not_equal(inputs.input_ids, ignore_id)
  candidates_mask &= tf.cast(inputs.input_mask, tf.bool)
  if disallow_from_mask is not None:
    candidates_mask &= ~disallow_from_mask
  return candidates_mask


def mask(config,
         inputs, mask_prob, proposal_distribution=1.0,
         disallow_from_mask=None, already_masked=None,
         features=None):
  """Implementation of dynamic masking. The optional arguments aren't needed for
  BERT/ELECTRA and are from early experiments in "strategically" masking out
  tokens instead of uniformly at random.

  Args:
    config: configure_pretraining.PretrainingConfig
    inputs: pretrain_data.Inputs containing input input_ids/input_mask
    mask_prob: percent of tokens to mask
    proposal_distribution: for non-uniform masking can be a [B, L] tensor
                           of scores for masking each position.
    disallow_from_mask: a boolean tensor of [B, L] of positions that should
                        not be masked out
    already_masked: a boolean tensor of [B, N] of already masked-out tokens
                    for multiple rounds of masking
  Returns: a pretrain_data.Inputs with masking added
  """
  # Get the batch size, sequence length, and max masked-out tokens
  N = config.max_predictions_per_seq
  B, L = modeling.get_shape_list(inputs.input_ids)

  # Find indices where masking out a token is allowed
  vocab = get_vocab(config)
  candidates_mask = get_candidates_mask(config, inputs, disallow_from_mask)

  # Set the number of tokens to mask out per example
  num_tokens = tf.cast(tf.reduce_sum(inputs.input_mask, -1), tf.float32)
  
  global_step = tf.train.get_or_create_global_step()
  mask_ratio = tf.train.polynomial_decay(
                          config.initial_ratio,
                          global_step,
                          int(config.num_train_steps*0.1),
                          end_learning_rate=config.final_ratio,
                          power=1.0,
                          cycle=True)

  num_to_predict = tf.maximum(1, tf.minimum(
      N, tf.cast(tf.round(num_tokens * mask_ratio), tf.int32)))
  masked_lm_weights = tf.cast(tf.sequence_mask(num_to_predict, N), tf.float32)
  if already_masked is not None:
    masked_lm_weights *= (1 - already_masked)

  # Get a probability of masking each position in the sequence
  candidate_mask_float = tf.cast(candidates_mask, tf.float32)
  sample_prob = (proposal_distribution * candidate_mask_float)
  sample_prob /= tf.reduce_sum(sample_prob, axis=-1, keepdims=True)

  # Sample the positions to mask out
  sample_prob = tf.stop_gradient(sample_prob)
  sample_logits = tf.log(sample_prob)
  masked_lm_positions = tf.random.categorical(
      sample_logits, N, dtype=tf.int32)
  masked_lm_positions *= tf.cast(masked_lm_weights, tf.int32)

  # Get the ids of the masked-out tokens
  shift = tf.expand_dims(L * tf.range(B), -1)
  flat_positions = tf.reshape(masked_lm_positions + shift, [-1, 1])
  masked_lm_ids = tf.gather_nd(tf.reshape(inputs.input_ids, [-1]),
                               flat_positions)
  masked_lm_ids = tf.reshape(masked_lm_ids, [B, -1])
  masked_lm_ids *= tf.cast(masked_lm_weights, tf.int32)

  # Update the input ids
  replace_with_mask_positions = masked_lm_positions * tf.cast(
      tf.less(tf.random.uniform([B, N]), 0.85), tf.int32)
  inputs_ids, _ = scatter_update(
      inputs.input_ids, tf.fill([B, N], vocab["[MASK]"]),
      replace_with_mask_positions)

  return pretrain_data.get_updated_inputs(
      inputs,
      input_ids=tf.stop_gradient(inputs_ids),
      masked_lm_positions=masked_lm_positions,
      masked_lm_ids=masked_lm_ids,
      masked_lm_weights=masked_lm_weights
  )

def unmask(inputs):
  unmasked_input_ids, _ = scatter_update(
      inputs.input_ids, inputs.masked_lm_ids, inputs.masked_lm_positions)
  return pretrain_data.get_updated_inputs(inputs, input_ids=unmasked_input_ids)

def greedy_from_softmax(logits, logits_temp=1.0, gumbel_temp=0.1, disallow=None):
  if disallow is not None:
    logits -= 1000.0 * disallow

  log_prob = tf.nn.log_softmax(logits/logits_temp)
  onehot_tokenids = tf.one_hot(tf.argmax(logits, -1,
                              output_type=tf.int32), 
                    logits.shape[-1])
  # [batch_size, masked_pos]
  tokenids_seq_logprob = tf.reduce_sum(tf.cast(onehot_tokenids, dtype=log_prob.dtype)*log_prob, axis=-1)
  # [batch_size]
  tokenids_logprob = tf.reduce_sum(tokenids_seq_logprob, axis=-1)
  return onehot_tokenids, tokenids_logprob

def sample_from_softmax(logits, logits_temp=1.0, gumbel_temp=0.1, disallow=None):
  if disallow is not None:
    logits -= 1000.0 * disallow
  uniform_noise = tf.random.uniform(
      modeling.get_shape_list(logits), minval=0, maxval=1)
  gumbel_noise = -tf.log(-tf.log(uniform_noise + 1e-9) + 1e-9)
  logits /= logits_temp
  log_prob = tf.nn.log_softmax(logits, axis=-1)
  onehot_tokenids = tf.one_hot(tf.argmax(tf.nn.softmax((logits + gumbel_noise)/gumbel_temp), -1,
                              output_type=tf.int32), logits.shape[-1])
  # [batch_size, masked_pos]
  tokenids_seq_logprob = tf.reduce_sum(tf.cast(onehot_tokenids, dtype=log_prob.dtype)*log_prob, axis=-1)
  # [batch_size]
  tokenids_logprob = tf.reduce_sum(tokenids_seq_logprob, axis=-1)
  return onehot_tokenids, tokenids_logprob

def sample_from_top_k(logits, logits_temp=1.0, gumbel_temp=0.1, disallow=None, k=20):
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
  topk_logprob = tf.nn.log_softmax(topk_logits, axis=-1)
  onehot_tokenids = tf.one_hot(tf.argmax(tf.nn.softmax((topk_logits + gumbel_noise)/gumbel_temp), -1,
                              output_type=tf.int32), topk_logits.shape[-1])
  # [batch_size, seq_length]
  tokenids_seq_logprob = tf.reduce_sum(tf.cast(onehot_tokenids, dtype=topk_logprob.dtype)*topk_logprob, axis=-1)
  # [batch_size]
  tokenids_logprob = tf.reduce_sum(tokenids_seq_logprob, axis=-1)
  return onehot_tokenids, tokenids_logprob

def sample_from_top_p(logits, logits_temp=1.0, gumbel_temp=0.1, disallow=None, p=0.95):
  """Nucleus sampling
  https://github.com/wouterkool/ancestral-gumbel-top-k-sampling
  """
  logits_shape = modeling.get_shape_list(logits, expected_rank=[2,3])
  depth_dimension = (len(logits_shape) == 3)
  if depth_dimension:
    reshape_logits = tf.reshape(logits, [-1, logits_shape[-1]])
  else:
    reshape_logits = logits
  # print(reshape_logits, '======')
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
  # print(topp_logits, '====topp_logits====')
  if disallow is not None:
    topp_logits -= 1e10 * disallow
  uniform_noise = tf.random.uniform(modeling.get_shape_list(topp_logits), minval=0, maxval=1)
  gumbel_noise = -tf.log(-tf.log(uniform_noise + 1e-9) + 1e-9)
  topp_logits /= logits_temp
  topp_logprob = tf.nn.log_softmax(topp_logits, axis=-1)
  onehot_tokenids = tf.one_hot(tf.argmax(tf.nn.softmax((topp_logits + gumbel_noise)/gumbel_temp), -1,
                            output_type=tf.int32), topp_logits.shape[-1])
  # [batch_size, masked_pos]
  tokenids_seq_logprob = tf.reduce_sum(tf.cast(onehot_tokenids, dtype=topp_logprob.dtype)*topp_logprob, axis=-1)
  # [batch_size]
  tokenids_logprob = tf.reduce_sum(tokenids_seq_logprob, axis=-1)
  return onehot_tokenids, tokenids_logprob
