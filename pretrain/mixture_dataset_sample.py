
from pretrain import span_mask_utils_ilm
import tensorflow as tf
import numpy as np

def shape_list(x, out_type=tf.int32):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x, out_type=out_type)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def _decode_finetune_record(FLAGS, record, name_to_features, 
          real_max_length):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  
  actual_len = tf.reduce_sum(example['input_mask'])

  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t
    pad_tensor = tf.zeros((real_max_length-actual_len), dtype=example[name].dtype)
    example[name] = tf.concat([example[name], pad_tensor], axis=0)

  mapping = {
  	'ipnut_ids': 'ilm_input_ids',
  	'segment_ids': 'ilm_segment_ids',
  	'input_mask': 'ilm_input_mask'
  }

  output_example = {}
  for name in mapping:
    output_example[mapping[name]] = example[name]
    pad_tensor = tf.zeros((real_max_length-actual_len), dtype=example[name].dtype)
    output_example[mapping[name]] = tf.concat([output_example[mapping[name]], pad_tensor], axis=0)

  example['fintune_loss_multipilier'] = tf.constant([1])
  example['pretrain_loss_multipilier'] = tf.constant([0])
  return example

def _decode_pretrain_record(FLAGS, record, name_to_features, 
          real_max_length):
  """Decodes a record to a TensorFlow example."""
  example = span_mask_utils_ilm._decode_record(FLAGS, record, 
                  FLAGS.max_predictions_per_seq,
                  FLAGS.max_seq_length, 
                  use_bfloat16=FLAGS.use_bfloat16, 
                  truncate_seq=FLAGS.truncate_seq, 
                  stride=FLAGS.stride)

  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  output_example = {}
  actual_len = tf.reduce_sum(example['ilm_input_mask'])
  
  mapping = {
  	'ilm_input': 'ilm_input_ids',
  	'ilm_segment_ids': 'ilm_segment_ids',
  	'ilm_input_mask': 'ilm_input_mask'
  }

  for name in mapping:
    output_example[mapping[name]] = example[name]
    pad_tensor = tf.zeros((real_max_length-actual_len), dtype=example[name].dtype)
    output_example[mapping[name]] = tf.concat([output_example[mapping[name]], pad_tensor], axis=0)
    
  output_example['fintune_loss_multipilier'] = tf.constant([0])
  output_example['pretrain_loss_multipilier'] = tf.constant([1])

  return output_example
 