
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

  tf.logging.info("** finetune input **")
  for k, v in example.items():
    tf.logging.info("%s: %s", k, v)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  actual_len = example['input_mask'].shape.as_list()[0]

  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  mapping = {
    'input_ids': 'ilm_input_ids',
    'segment_ids': 'ilm_segment_ids',
    'input_mask': 'ilm_input_mask'
  }

  output_example = {}
  for name in mapping:
    output_example[mapping[name]] = example[name]
    pad_tensor = tf.zeros((real_max_length-actual_len), dtype=example[name].dtype)
    output_example[mapping[name]] = tf.concat([output_example[mapping[name]], pad_tensor], axis=0)

    tgt_shape = example['input_ids'].shape.as_list()
    tgt_shape[0] = real_max_length
    output_example[mapping[name]].set_shape(tgt_shape)
    
  # output_example['fintune_loss_multipilier'] = tf.constant([1], dtype=tf.int32)
  # output_example['pretrain_loss_multipilier'] = tf.constant([0], dtype=tf.int32)
  tf.logging.info("** finetune input **")
  for k, v in output_example.items():
    tf.logging.info("%s: %s", k, v)
  return output_example

def _decode_pretrain_record(FLAGS, record, name_to_features, 
          real_max_length):
  """Decodes a record to a TensorFlow example."""
  name_to_features = {
      "input_ori_ids": tf.io.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
  }

  example = span_mask_utils_ilm._decode_record(FLAGS, record, 
                  FLAGS.max_predictions_per_seq,
                  FLAGS.max_seq_length, 
                  record_spec=name_to_features,
                  input_ids_name='input_ori_ids',
                  use_bfloat16=FLAGS.use_bfloat16, 
                  truncate_seq=FLAGS.truncate_seq, 
                  stride=FLAGS.stride)

  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  output_example = {}
  actual_len = example['ilm_input_mask'].shape.as_list()[0]
  
  mapping = {
    'ilm_input': 'ilm_input_ids',
    'ilm_segment_ids': 'ilm_segment_ids',
    'ilm_input_mask': 'ilm_input_mask'
  }

  for name in mapping:
    output_example[mapping[name]] = tf.identity(example[name])
    pad_tensor = tf.zeros((real_max_length-actual_len), dtype=example[name].dtype)
    output_example[mapping[name]] = tf.concat([output_example[mapping[name]], pad_tensor], axis=0)
    
    tgt_shape = example['masked_input'].shape.as_list()
    tgt_shape[0] = real_max_length
    output_example[mapping[name]].set_shape(tgt_shape)
    
  # output_example['fintune_loss_multipilier'] = tf.constant((0,), dtype=tf.int32)
  # output_example['pretrain_loss_multipilier'] = tf.constant((1, ), dtype=tf.int32)

  tf.logging.info("** pretrain input **")
  for k, v in output_example.items():
    tf.logging.info("%s: %s", k, v)
  return output_example

def _decode_pretrain_record_v1(FLAGS, record, name_to_features, 
          real_max_length,
          record_spec=None,
          input_ids_name=None):
  """Decodes a record to a TensorFlow example."""
  example = span_mask_utils_ilm._decode_record(FLAGS, record, 
                  FLAGS.max_predictions_per_seq,
                  FLAGS.max_seq_length, 
                  use_bfloat16=FLAGS.use_bfloat16, 
                  truncate_seq=FLAGS.truncate_seq, 
                  stride=FLAGS.stride,
                  record_spec=record_spec,
                  input_ids_name=input_ids_name)

  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  output_example = {}
  actual_len = example['ilm_input_mask'].shape.as_list()[0]
  
  mapping = {
    'ilm_input': 'ilm_input_ids',
    'ilm_segment_ids': 'ilm_segment_ids',
    'ilm_input_mask': 'ilm_input_mask'
  }

  for name in mapping:
    output_example[mapping[name]] = tf.identity(example[name])
    pad_tensor = tf.zeros((real_max_length-actual_len), dtype=example[name].dtype)
    output_example[mapping[name]] = tf.concat([output_example[mapping[name]], pad_tensor], axis=0)
    
    tgt_shape = example['masked_input'].shape.as_list()
    tgt_shape[0] = real_max_length
    output_example[mapping[name]].set_shape(tgt_shape)
    
  mappinp_1 = {
    'origin_input': 'input_ids',
    'segment_ids': 'segment_ids',
    'pad_mask': 'input_mask'
  }
  for key in mappinp_1:
    output_example[mappinp_1[key]] = example[key]
  # output_example['fintune_loss_multipilier'] = tf.constant((0,), dtype=tf.int32)
  # output_example['pretrain_loss_multipilier'] = tf.constant((1, ), dtype=tf.int32)

  tf.logging.info("** pretrain input **")
  for k, v in output_example.items():
    tf.logging.info("%s: %s", k, v)
  return output_example
 