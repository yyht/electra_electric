"""Create input function for estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import collections

import numpy as np
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

from pretrain import pretrain_data
from pretrain import pretrain_helpers
# from model.vqvae_utils import tfidf_utils

def check_tf_version():
  version = tf.__version__
  print("==tf version==", version)
  if int(version.split(".")[0]) >= 2 or int(version.split(".")[1]) >= 15:
    return True
  else:
    return False

special_symbols_mapping = collections.OrderedDict([
    ("<unk>", "unk_id"),
    ("<s>", "bos_id"),
    ("</s>", "eos_id"),
    ("<cls>", "cls_id"),
    ("<sep>", "sep_id"),
    ("<pad>", "pad_id"),
    ("<mask>", "mask_id"),
    ("<eod>", "eod_id"),
    ("<eop>", "eop_id")
])

def _get_boundary_indices(tokenizer, seg, reverse=False):
  """Get all boundary indices of whole words."""
  seg_len = len(seg)
  if reverse:
    seg = np.flip(seg, 0)

  boundary_indices = []
  for idx, token in enumerate(seg):
    if tokenizer.is_start_token(token) and not tokenizer.is_func_token(token):
      boundary_indices.append(idx)
  boundary_indices.append(seg_len)

  if reverse:
    boundary_indices = [seg_len - idx for idx in boundary_indices]

  return boundary_indices


def setup_special_ids(FLAGS, tokenizer):
  """Set up the id of special tokens."""
  FLAGS.vocab_size = tokenizer.get_vocab_size()
  tf.logging.info("Set vocab_size: %d.", FLAGS.vocab_size)
  for sym, sym_id_str in special_symbols_mapping.items():
    try:
      sym_id = tokenizer.get_token_id(sym)
      setattr(FLAGS, sym_id_str, sym_id)
      tf.logging.info("Set %s to %d.", sym_id_str, sym_id)
    except KeyError:
      tf.logging.warning("Skip %s: not found in tokenizer's vocab.", sym)


def format_filename(prefix, suffix, seq_len, uncased):
  """Format the name of the tfrecord/meta file."""
  seq_str = "seq-{}".format(seq_len)
  if uncased:
    case_str = "uncased"
  else:
    case_str = "cased"

  file_name = "{}.{}.{}.{}".format(prefix, seq_str, case_str, suffix)

  return file_name


def convert_example(example, use_bfloat16=False):
  """Cast int64 into int32 and float32 to bfloat16 if use_bfloat16."""
  for key in list(example.keys()):
    val = example[key]
    if tf.keras.backend.is_sparse(val):
      val = tf.sparse.to_dense(val)
    if val.dtype == tf.int64:
      val = tf.cast(val, tf.int32)
    if use_bfloat16 and val.dtype == tf.float32:
      val = tf.cast(val, tf.bfloat16)

    example[key] = val


def sparse_to_dense(example):
  """Convert sparse feature to dense ones."""
  for key in list(example.keys()):
    val = example[key]
    if tf.keras.backend.is_sparse(val):
      val = tf.sparse.to_dense(val)
    example[key] = val

  return example

def prepare_text_infilling(input_ids, duplicate_ids=103):
  input_left_shift = tf.concat((input_ids[1:], [0]), axis=0)
  mask_left_shift = tf.logical_or(tf.not_equal(input_ids - input_left_shift, 0), tf.not_equal(input_ids, duplicate_ids))
  dup_mask = tf.concat(([True], mask_left_shift[:-1]), axis=0)
  dup_input_ids_out = tf.boolean_mask(input_ids, dup_mask)
  return dup_input_ids_out, dup_mask

def _idx_pair_to_mask(FLAGS, beg_indices, end_indices, inputs, tgt_len, num_predict):
  """Turn beg and end indices into actual mask."""
  non_func_mask = tf.logical_and(
      tf.not_equal(inputs, FLAGS.sep_id),
      tf.not_equal(inputs, FLAGS.cls_id))
  all_indices = tf.where(
      non_func_mask,
      tf.range(tgt_len, dtype=tf.int32),
      tf.constant(-1, shape=[tgt_len], dtype=tf.int32))
  candidate_matrix = tf.cast(
      tf.logical_and(
          all_indices[None, :] >= beg_indices[:, None],
          all_indices[None, :] < end_indices[:, None]),
      tf.float32)
  cumsum_matrix = tf.reshape(
      tf.cumsum(tf.reshape(candidate_matrix, [-1])),
      [-1, tgt_len])
  masked_matrix = tf.cast(cumsum_matrix <= tf.cast(num_predict, dtype=cumsum_matrix.dtype), tf.float32)
  target_mask = tf.reduce_sum(candidate_matrix * masked_matrix, axis=0)
  is_target = tf.cast(target_mask, tf.bool)

  return is_target, target_mask


def _word_span_mask(FLAGS, inputs, tgt_len, num_predict, boundary, stride=1):
  """Sample whole word spans as prediction targets."""
  # Note: 1.2 is roughly the token-to-word ratio

  input_mask = tf.cast(tf.not_equal(inputs, FLAGS.pad_id), dtype=tf.int32)
  num_tokens = tf.cast(tf.reduce_sum(input_mask, -1), tf.int32)
  num_predict = tf.cast(num_predict, tf.int32)

  non_pad_len = num_tokens + 1 - stride

  chunk_len_fp = tf.cast(non_pad_len / num_predict / 1.2, dtype=tf.float32)
  round_to_int = lambda x: tf.cast(tf.round(x), tf.int64)

  # Sample span lengths from a zipf distribution
  span_len_seq = np.arange(FLAGS.min_word, FLAGS.max_word + 1)
  probs = np.array([1.0 /  (i + 1) for i in span_len_seq])
  probs /= np.sum(probs)
  logits = tf.constant(np.log(probs), dtype=tf.float32)

  if check_tf_version():
    span_lens = tf.random.categorical(
        logits=logits[None],
        num_samples=num_predict,
        dtype=tf.int64,
    )[0] + FLAGS.min_word
  else:
    span_lens = tf.multinomial(
        logits=logits[None],
        num_samples=num_predict,
        output_dtype=tf.int64,
    )[0] + FLAGS.min_word

  # Sample the ratio [0.0, 1.0) of left context lengths
  span_lens_fp = tf.cast(span_lens, tf.float32)
  left_ratio = tf.random.uniform(shape=[num_predict], minval=0.0, maxval=1.0)
  left_ctx_len = left_ratio * span_lens_fp * (chunk_len_fp - 1)

  left_ctx_len = round_to_int(left_ctx_len)
  right_offset = round_to_int(span_lens_fp * chunk_len_fp) - left_ctx_len

  beg_indices = (tf.cumsum(left_ctx_len) +
                 tf.cumsum(right_offset, exclusive=True))
  end_indices = beg_indices + tf.cast(span_lens, dtype=tf.int32)

  # Remove out of range `boundary` indices
  max_boundary_index = tf.cast(tf.shape(boundary)[0] - 1, tf.int64)
  valid_idx_mask = end_indices < max_boundary_index
  beg_indices = tf.boolean_mask(beg_indices, valid_idx_mask)
  end_indices = tf.boolean_mask(end_indices, valid_idx_mask)

  beg_indices = tf.gather(boundary, beg_indices)
  end_indices = tf.gather(boundary, end_indices)

  # Shuffle valid `position` indices
  num_valid = tf.cast(tf.shape(beg_indices)[0], tf.int64)
  order = tf.random.shuffle(tf.range(num_valid, dtype=tf.int64))
  beg_indices = tf.gather(beg_indices, order)
  end_indices = tf.gather(end_indices, order)

  return _idx_pair_to_mask(FLAGS, beg_indices, end_indices, inputs, tgt_len,
                           num_predict)


def _token_span_mask(FLAGS, inputs, tgt_len, num_predict, stride=1):
  """Sample token spans as prediction targets."""
  # non_pad_len = tgt_len + 1 - stride

  input_mask = tf.cast(tf.not_equal(inputs, FLAGS.pad_id), dtype=tf.int32)
  num_tokens = tf.cast(tf.reduce_sum(input_mask, -1), tf.int32)
  num_predict = tf.cast(num_predict, tf.int32)

  non_pad_len = num_tokens + 1 - stride

  chunk_len_fp = tf.cast(non_pad_len / num_predict, dtype=tf.float32)
  round_to_int = lambda x: tf.cast(tf.round(x), tf.int32)

  # Sample span lengths from a zipf distribution
  # span_len_seq = np.arange(FLAGS.min_tok, FLAGS.max_tok + 1)
  probs = [FLAGS.p * (1-FLAGS.p)**(i - FLAGS.min_tok) for i in range(FLAGS.min_tok, FLAGS.max_tok+1)] 
  # probs = [x / (sum(len_distrib)) for x in len_distrib]
  # probs = np.array([1.0 /  (i + 1) for i in span_len_seq])

  probs /= np.sum(probs)
  tf.logging.info("** sampling probs **")
  tf.logging.info(probs)

  logits = tf.constant(np.log(probs), dtype=tf.float32)
  if check_tf_version():
    span_lens = tf.random.categorical(
        logits=logits[None],
        num_samples=num_predict,
        dtype=tf.int64,
    )[0] + FLAGS.min_tok
  else:
    span_lens = tf.multinomial(
        logits=logits[None],
        num_samples=num_predict,
        output_dtype=tf.int64,
    )[0] + FLAGS.min_tok

  # Sample the ratio [0.0, 1.0) of left context lengths
  span_lens_fp = tf.cast(span_lens, tf.float32)
  left_ratio = tf.random.uniform(shape=[num_predict], minval=0.0, maxval=1.0)
  left_ctx_len = left_ratio * span_lens_fp * (chunk_len_fp - 1)
  left_ctx_len = round_to_int(left_ctx_len)

  # Compute the offset from left start to the right end
  right_offset = round_to_int(span_lens_fp * chunk_len_fp) - left_ctx_len

  # Get the actual begin and end indices
  beg_indices = (tf.cumsum(left_ctx_len) +
                 tf.cumsum(right_offset, exclusive=True))
  end_indices = beg_indices + tf.cast(span_lens, dtype=tf.int32)

  # Remove out of range indices
  valid_idx_mask = end_indices < non_pad_len
  beg_indices = tf.boolean_mask(beg_indices, valid_idx_mask)
  end_indices = tf.boolean_mask(end_indices, valid_idx_mask)

  # Shuffle valid indices
  num_valid = tf.cast(tf.shape(beg_indices)[0], tf.int64)
  order = tf.random.shuffle(tf.range(num_valid, dtype=tf.int64))
  beg_indices = tf.gather(beg_indices, order)
  end_indices = tf.gather(end_indices, order)

  return _idx_pair_to_mask(FLAGS, beg_indices, end_indices, inputs, tgt_len,
                           num_predict)


def _whole_word_mask(FLAGS, inputs, tgt_len, num_predict, boundary):
  """Sample whole words as prediction targets."""
  pair_indices = tf.concat([boundary[:-1, None], boundary[1:, None]], axis=1)
  cand_pair_indices = tf.random.shuffle(pair_indices)[:num_predict]
  beg_indices = cand_pair_indices[:, 0]
  end_indices = cand_pair_indices[:, 1]

  return _idx_pair_to_mask(FLAGS, beg_indices, end_indices, inputs, tgt_len,
                           num_predict)


def _single_token_mask(FLAGS, inputs, tgt_len, num_predict, exclude_mask=None):
  """Sample individual tokens as prediction targets."""
  func_mask = tf.equal(inputs, FLAGS.cls_id)
  func_mask = tf.logical_or(func_mask, tf.equal(inputs, FLAGS.sep_id))
  func_mask = tf.logical_or(func_mask, tf.equal(inputs, FLAGS.pad_id))
  if exclude_mask is None:
    exclude_mask = func_mask
  else:
    exclude_mask = tf.logical_or(func_mask, exclude_mask)
  candidate_mask = tf.logical_not(exclude_mask)

  input_mask = tf.cast(tf.not_equal(inputs, FLAGS.pad_id), dtype=tf.int64)
  num_tokens = tf.cast(tf.reduce_sum(input_mask, -1), tf.int64)

  all_indices = tf.range(tgt_len, dtype=tf.int64)
  candidate_indices = tf.boolean_mask(all_indices, candidate_mask)
  masked_pos = tf.random.shuffle(candidate_indices)
  if check_tf_version():
    masked_pos = tf.sort(masked_pos[:num_predict])
  else:
    masked_pos = tf.contrib.framework.sort(masked_pos[:num_predict])
  target_mask = tf.sparse_to_dense(
      sparse_indices=masked_pos,
      output_shape=[tgt_len],
      sparse_values=1.0,
      default_value=0.0)
  is_target = tf.cast(target_mask, tf.bool)

  return is_target, target_mask


def _online_sample_masks(FLAGS, 
    inputs, tgt_len, num_predict, boundary=None, stride=1):
  """Sample target positions to predict."""

  # Set the number of tokens to mask out per example
  input_mask = tf.cast(tf.not_equal(inputs, FLAGS.pad_id), dtype=tf.int64)
  num_tokens = tf.cast(tf.reduce_sum(input_mask, -1), tf.float32)

  # global_step = tf.train.get_or_create_global_step()
  # mask_prob = tf.train.polynomial_decay(
  #                         FLAGS.initial_ratio,
  #                         global_step,
  #                         int(FLAGS.num_train_steps*0.1),
  #                         end_learning_rate=FLAGS.final_ratio,
  #                         power=1.0,
  #                         cycle=True)

  mask_prob = FLAGS.final_ratio
  tf.logging.info("mask_prob: `%s`.", mask_prob)

  num_predict = tf.maximum(1, tf.minimum(
      num_predict, tf.cast(tf.round(num_tokens * mask_prob), tf.int32)))
  num_predict = tf.cast(num_predict, tf.int32)

  tf.logging.info("Online sample with strategy: `%s`.", FLAGS.sample_strategy)
  if FLAGS.sample_strategy == "single_token":
    return _single_token_mask(inputs, tgt_len, num_predict)
  else:
    if FLAGS.sample_strategy == "whole_word":
      assert boundary is not None, "whole word sampling requires `boundary`"
      is_target, target_mask = _whole_word_mask(FLAGS, inputs, tgt_len, num_predict,
                                                boundary)
    elif FLAGS.sample_strategy == "token_span":
      is_target, target_mask = _token_span_mask(FLAGS, inputs, tgt_len, num_predict,
                                                stride=stride)
    elif FLAGS.sample_strategy == "word_span":
      assert boundary is not None, "word span sampling requires `boundary`"
      is_target, target_mask = _word_span_mask(FLAGS, inputs, tgt_len, num_predict,
                                               boundary, stride=stride)
    else:
      raise NotImplementedError

    valid_mask = tf.not_equal(inputs, FLAGS.pad_id)
    is_target = tf.logical_and(valid_mask, is_target)
    target_mask = target_mask * tf.cast(valid_mask, tf.float32)

    # Fill in single tokens if not full
    cur_num_masked = tf.reduce_sum(tf.cast(is_target, tf.int32))
    extra_mask, extra_tgt_mask = _single_token_mask(FLAGS,
        inputs, tgt_len, num_predict - cur_num_masked, is_target)
    return tf.logical_or(is_target, extra_mask), target_mask + extra_tgt_mask


def discrepancy_correction(FLAGS, inputs, is_target, tgt_len):
  """Construct the masked input."""
  random_p = tf.random.uniform([tgt_len], maxval=1.0)
  mask_ids = tf.constant(FLAGS.mask_id, dtype=inputs.dtype, shape=[tgt_len])
  masked_ids = tf.where(is_target, mask_ids, inputs)

  if FLAGS.leak_ratio > 0:
    change_to_mask = tf.logical_and(random_p > FLAGS.leak_ratio, is_target)
    masked_ids_with_leak = tf.where(change_to_mask, mask_ids, inputs)
  else:
    masked_ids_with_leak = tf.identity(masked_ids)

  if FLAGS.rand_ratio > 0:
    change_to_rand = tf.logical_and(
        FLAGS.leak_ratio < random_p,
        random_p < FLAGS.leak_ratio + FLAGS.rand_ratio)
    change_to_rand = tf.logical_and(change_to_rand, is_target)
    rand_ids = tf.random.uniform([tgt_len], maxval=FLAGS.vocab_size,
                                 dtype=masked_ids_with_leak.dtype)
    masked_ids_random_with_leak = tf.where(change_to_rand, rand_ids, masked_ids_with_leak)
  else:
    masked_ids_random_with_leak = tf.identity(masked_ids_with_leak)

  return masked_ids, masked_ids_random_with_leak


def create_target_mapping(
    example, is_target, seq_len, num_predict, **kwargs):
  """Create target mapping and retrieve the corresponding kwargs."""
  if num_predict is not None:
    # Get masked indices
    indices = tf.range(seq_len, dtype=tf.int64)
    indices = tf.boolean_mask(indices, is_target)

    # Handle the case that actual_num_predict < num_predict
    actual_num_predict = tf.shape(indices)[0]
    pad_len = num_predict - actual_num_predict

    # Create target mapping
    target_mapping = tf.one_hot(indices, seq_len, dtype=tf.float32)
    paddings = tf.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
    target_mapping = tf.concat([target_mapping, paddings], axis=0)
    example["target_mapping"] = tf.reshape(target_mapping,
                                           [num_predict, seq_len])

    # Handle fields in kwargs
    for k, v in kwargs.items():
      pad_shape = [pad_len] + v.shape.as_list()[1:]
      tgt_shape = [num_predict] + v.shape.as_list()[1:]
      example[k] = tf.concat([
          tf.boolean_mask(v, is_target),
          tf.zeros(shape=pad_shape, dtype=v.dtype)], 0)
      example[k].set_shape(tgt_shape)
  else:
    for k, v in kwargs.items():
      example[k] = v

def prepare_ilm(masked_input, duplicate_ids, pad_mask):
  [text_infilling_ids, 
  text_infilling_mask] = prepare_text_infilling(masked_input, duplicate_ids=duplicate_ids)
  text_infilling_mask = tf.cast(text_infilling_mask, tf.float32)
  
  text_infilling_ids = tf.boolean_mask(text_infilling_ids, tf.not_equal(text_infilling_ids, 0))

  return text_infilling_ids

def _decode_record(FLAGS, record, num_predict,
                  seq_len, 
                  use_bfloat16=False, 
                  truncate_seq=False, 
                  stride=1):
  max_seq_length = seq_len
  record_spec = {
        "input_ori_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64)
  }
  if FLAGS.sample_strategy in ["whole_word", "word_span"]:
    tf.logging.info("Add `boundary` spec for %s", FLAGS.sample_strategy)
    record_spec["boundary"] = tf.VarLenFeature(tf.int64)

  example = tf.parse_single_example(record, record_spec)
  inputs = example.pop("input_ori_ids")
  if FLAGS.sample_strategy in ["whole_word", "word_span"]:
    boundary = tf.sparse.to_dense(example.pop("boundary"))
  else:
    boundary = None
  if truncate_seq and stride > 1:
    tf.logging.info("Truncate pretrain sequence with stride %d", stride)
    # seq_len = 8, stride = 2:
    #   [cls 1 2 sep 4 5 6 sep] => [cls 1 2 sep 4 5 sep pad]
    padding = tf.constant([FLAGS.sep_id] + [FLAGS.pad_id] * (stride - 1),
                          dtype=inputs.dtype)
    inputs = tf.concat([inputs[:-stride], padding], axis=0)
    if boundary is not None:
      valid_boundary_mask = boundary < seq_len - stride
      boundary = tf.boolean_mask(boundary, valid_boundary_mask)

  is_target, target_mask = _online_sample_masks(FLAGS,
        inputs, seq_len, num_predict, boundary=boundary, stride=stride)

  [ilm_masked_input, masked_input] = discrepancy_correction(FLAGS, inputs, is_target, seq_len)
  masked_input = tf.reshape(masked_input, [max_seq_length])
  is_mask = tf.equal(masked_input, FLAGS.mask_id)
  is_pad = tf.equal(masked_input, FLAGS.pad_id)

  origin_input_mask = tf.equal(inputs, FLAGS.pad_id)
  masked_input *= (1 - tf.cast(origin_input_mask, dtype=tf.int64))

  example["masked_input"] = masked_input
  example["origin_input"] = inputs
  example["is_target"] = tf.cast(is_target, dtype=tf.int64) * (1 - tf.cast(origin_input_mask, dtype=tf.int64))
  # example["input_mask"] = tf.cast(tf.logical_or(is_mask, is_pad), tf.float32)
  # example["pad_mask"] = tf.cast(is_pad, tf.float32)
  input_mask = tf.logical_or(tf.logical_or(is_mask, is_pad), origin_input_mask)
  example["masked_mask"] = 1.0 - tf.cast(tf.logical_or(is_mask, is_pad), dtype=tf.float32)
  pad_mask = tf.logical_or(origin_input_mask, is_pad)
  example["pad_mask"] = 1.0 - tf.cast(pad_mask, tf.float32)

  # create target mapping
  create_target_mapping(
      example, is_target, seq_len, num_predict,
      target_mask=target_mask, target=inputs)

  example["masked_lm_positions"] = tf.argmax(example['target_mapping'], axis=-1)
  example["masked_lm_weights"] = example['target_mask']
  example["masked_lm_ids"] = example['target']

  if FLAGS.ilm_v1:

    tf.logging.info("** apply same placeholder [MASK] **")

    # ['[CLS]', [mask], 'a', 'b', '[SEP]']
    ilm_prefix = prepare_ilm(ilm_masked_input, FLAGS.mask_id, example["pad_mask"])
    suffix_ids = tf.cast((1.0 - target_mask) * FLAGS.seg_id, dtype=inputs.dtype) + inputs * tf.cast(target_mask, dtype=inputs.dtype)
    ilm_suffix = prepare_ilm(suffix_ids, FLAGS.seg_id, example["pad_mask"])

    ilm_prefix_segment_ids = tf.zeros_like(ilm_prefix)
    ilm_suffix_segment_ids = tf.ones_like(ilm_suffix)

    ilm_input = tf.concat([ilm_prefix, 
                          ilm_suffix[1:], 
                          tf.constant([FLAGS.sep_id], dtype=tf.int64)], axis=0)
    ilm_segment_ids = tf.concat([ilm_prefix_segment_ids, 
                          ilm_suffix_segment_ids[1:], 
                          tf.constant([1], dtype=tf.int64)], axis=0)
    
    ilm_len = tf.reduce_sum(tf.cast(tf.not_equal(ilm_input, 0), dtype=tf.int32))
    ilm_pad = tf.zeros((max_seq_length+num_predict-ilm_len), dtype=ilm_input.dtype)
    ilm_input = tf.concat([ilm_input, ilm_pad], axis=0)
    ilm_segment_ids = tf.concat([ilm_segment_ids, ilm_pad], axis=0)
    ilm_input_mask = tf.cast(tf.not_equal(ilm_input, 0), dtype=tf.int32)

  elif FLAGS.ilm_v2:

    tf.logging.info("** apply different placeholder [unused] **")

    ilm_prefix = prepare_ilm(ilm_masked_input, FLAGS.mask_id, example["pad_mask"])
    ilm_prefix_mask = tf.cast(tf.equal(ilm_prefix, FLAGS.mask_id), dtype=tf.int64)
    ilm_seg_prefix = tf.cast(tf.cumsum(ilm_prefix_mask), dtype=tf.int64) * ilm_prefix_mask
    
    ilm_prefix = (1-ilm_prefix_mask) * ilm_prefix + ilm_seg_prefix
    
    suffix_ids = tf.cast((1.0 - target_mask) * FLAGS.seg_id, dtype=inputs.dtype) + inputs * tf.cast(target_mask, dtype=inputs.dtype)
    ilm_suffix = prepare_ilm(suffix_ids, FLAGS.seg_id, example["pad_mask"])[1:]
    
    ilm_suffix_mask = tf.cast(tf.equal(ilm_suffix, FLAGS.seg_id), dtype=tf.int64)
    ilm_seg_suffix = tf.cast(tf.cumsum(ilm_suffix_mask), dtype=tf.int64) * ilm_suffix_mask
    
    ilm_suffix = (1-ilm_suffix_mask) * ilm_suffix + ilm_seg_suffix
    
    ilm_prefix_segment_ids = tf.zeros_like(ilm_prefix)
    ilm_suffix_segment_ids = tf.ones_like(ilm_suffix)

    ilm_input = tf.concat([ilm_prefix, 
                          ilm_suffix, 
                          tf.constant([FLAGS.sep_id], dtype=tf.int64)], axis=0)
    ilm_segment_ids = tf.concat([ilm_prefix_segment_ids, 
                          ilm_suffix_segment_ids, 
                          tf.constant([1], dtype=tf.int64)], axis=0)
    
    ilm_len = tf.reduce_sum(tf.cast(tf.not_equal(ilm_input, 0), dtype=tf.int32))
    ilm_pad = tf.zeros((max_seq_length+88-ilm_len), dtype=ilm_input.dtype)
    ilm_input = tf.concat([ilm_input, ilm_pad], axis=0)
    ilm_segment_ids = tf.concat([ilm_segment_ids, ilm_pad], axis=0)
    ilm_input_mask = tf.cast(tf.not_equal(ilm_input, 0), dtype=tf.int32)

  if FLAGS.ilm_v2 or FLAGS.ilm_v1:
    tgt_shape = inputs.shape.as_list()
    tgt_shape[0] = max_seq_length+num_predict
    ilm_input.set_shape(tgt_shape)
    ilm_segment_ids.set_shape(tgt_shape)
    ilm_input_mask.set_shape(tgt_shape)
    
    example['ilm_input'] = ilm_input
    example['ilm_segment_ids'] = ilm_segment_ids
    example['ilm_input_mask'] = ilm_input_mask

  # type cast for example
  convert_example(example, use_bfloat16)

  for k, v in example.items():
    tf.logging.info("%s: %s", k, v)

  return example

def get_input_fn(config, is_training,
                 num_cpu_threads=4):

  input_files = []
  for input_pattern in config.pretrain_tfrecords.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))

  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(config, record, 
                  config.max_predictions_per_seq,
                  config.max_seq_length, 
                  use_bfloat16=config.use_bfloat16, 
                  truncate_seq=config.truncate_seq, 
                  stride=config.stride),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    d = d.apply(tf.data.experimental.ignore_errors())
    return d

  return input_fn

Inputs = collections.namedtuple(
    "Inputs", ["input_ids", "input_mask", "segment_ids", "masked_lm_positions",
               "masked_lm_ids", "masked_lm_weights"])

def features_to_inputs(features):
  return Inputs(
      input_ids=features["origin_input"],
      input_mask=features["pad_mask"],
      segment_ids=features["segment_ids"],
      masked_lm_positions=(features["masked_lm_positions"]
                           if "masked_lm_positions" in features else None),
      masked_lm_ids=(features["masked_lm_ids"]
                     if "masked_lm_ids" in features else None),
      masked_lm_weights=(features["masked_lm_weights"]
                         if "masked_lm_weights" in features else None),
  )

def mask(config,
         inputs, mask_prob, proposal_distribution=1.0,
         disallow_from_mask=None, already_masked=None,
         features=None):
  if features is not None:
    masked_inputs = Inputs(
      input_ids=features["masked_input"],
      input_mask=features["pad_mask"],
      segment_ids=features["segment_ids"],
      masked_lm_positions=features["masked_lm_positions"],
      masked_lm_ids=features["masked_lm_ids"],
      masked_lm_weights=features["masked_lm_weights"]
    )
  else:
    masked_inputs = pretrain_data.mask(config,
                      inputs, 
                      mask_prob, 
                      proposal_distribution,
                      disallow_from_mask,
                      already_masked)
  return masked_inputs