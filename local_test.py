

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

from pretrain import pretrain_data
from pretrain import pretrain_helpers
from util import training_utils
from util import utils, log_utils


import argparse
import collections
import json

# import tensorflow.compat.v1 as tf
import tensorflow as tf
tf.disable_v2_behavior()

import configure_pretraining
from model import modeling
from model import optimization
from pretrain import pretrain_data
from pretrain import pretrain_helpers
from util import training_utils
from util import utils, log_utils

name_to_features = {
    "input_ori_ids": tf.io.FixedLenFeature([512], tf.int64),
    "segment_ids": tf.io.FixedLenFeature([512], tf.int64),
}

def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  # example = tf.io.parse_single_example(record, name_to_features)
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    example[name] = t
  input_mask = tf.cast(tf.not_equal(example['input_ori_ids'], 
                                    0), tf.int32)
  example['input_mask'] = input_mask
  example['input_ids'] = example['input_ori_ids']
  example['target_ids'] = example['input_ori_ids']

  excelude_cls_target_mask = tf.cast(tf.equal(example['target_ids'], 
                            101), tf.int32) # [cls]
  excelude_sep_target_mask = tf.cast(tf.equal(example['target_ids'], 
                            102), tf.int32) # [sep]
  excelude_unk_target_mask = tf.cast(tf.equal(example['target_ids'], 
                            100), tf.int32) # [unk]
  excelud_target_mask = excelude_unk_target_mask + excelude_cls_target_mask + excelude_sep_target_mask
  example['target_mask'] = input_mask * (1 - excelud_target_mask)


  return example

def train_input_fn(input_file, _parse_fn, name_to_features,
                    **kargs):

    dataset = tf.data.TFRecordDataset(input_file, buffer_size=1)
    dataset = dataset.map(lambda x:_parse_fn(x, name_to_features))
    dataset = dataset.batch(1)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features

def get_softmax_output(logits, targets, weights, vocab_size):
  oh_labels = tf.one_hot(targets, depth=vocab_size, dtype=tf.float32)
  preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
  probs = tf.nn.softmax(logits)
  log_probs = tf.nn.log_softmax(logits)
  # [batch_size, num_masked]
  label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)
  numerator = tf.reduce_sum(weights * label_log_probs, axis=-1)
  # [batch_size, num_masked]
  denominator = tf.reduce_sum(weights, axis=-1)
  # pseudo_logprob = -numerator / (denominator + 1e-6)
  pseudo_logprob = -numerator
  print("== get_softmax_output ==", pseudo_logprob)
  loss = tf.reduce_sum(numerator) / (tf.reduce_sum(denominator) + 1e-6)
  SoftmaxOutput = collections.namedtuple(
    "SoftmaxOutput", ["logits", "probs", "loss", "per_example_loss", "preds",
            "weights", "pseudo_logprob"])
  return SoftmaxOutput(
    logits=logits, probs=probs, per_example_loss=label_log_probs,
    loss=loss, preds=preds, weights=weights,
    pseudo_logprob=pseudo_logprob
    )

def _get_masked_lm_output(inputs, model):
  """Masked language modeling softmax layer."""
  with tf.variable_scope("generator_predictions"):
    
    logits = tf.zeros(21228)
    logits_tiled = tf.zeros(
      modeling.get_shape_list(inputs.masked_lm_ids) +
      [21228])
    logits_tiled += tf.reshape(logits, [1, 1, 21228])
    logits = logits_tiled
    
    return get_softmax_output(
      logits, inputs.masked_lm_ids, inputs.masked_lm_weights,
      21228)

def _get_fake_data(inputs, mlm_logits):
  """Sample from the generator to create corrupted input."""
  masked_lm_weights = inputs.masked_lm_weights
  inputs = pretrain_helpers.unmask(inputs)
  disallow = None
  sampled_tokens = tf.stop_gradient(pretrain_helpers.sample_from_softmax(
      mlm_logits / 1.0, disallow=disallow))

  # sampled_tokens: [batch_size, n_pos, n_vocab]
  # mlm_logits: [batch_size, n_pos, n_vocab]
  sampled_tokens_fp32 = tf.cast(sampled_tokens, dtype=tf.float32)
  print(sampled_tokens_fp32, "===sampled_tokens_fp32===")
  # [batch_size, n_pos]
  # mlm_logprobs: [batch_size, n_pos. n_vocab]
  mlm_logprobs = tf.nn.log_softmax(mlm_logits, axis=-1)
  pseudo_logprob = tf.reduce_sum(mlm_logprobs*sampled_tokens_fp32, axis=-1)
  pseudo_logprob *= tf.cast(masked_lm_weights, dtype=tf.float32)
  # [batch_size]
  pseudo_logprob = tf.reduce_sum(pseudo_logprob, axis=-1)
  # [batch_size]
  # pseudo_logprob /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32), axis=-1))
  print("== _get_fake_data pseudo_logprob ==", pseudo_logprob)
  sampled_tokids = tf.argmax(sampled_tokens, -1, output_type=tf.int32)
  updated_input_ids, masked = pretrain_helpers.scatter_update(
      inputs.input_ids, sampled_tokids, inputs.masked_lm_positions)
  
  labels = masked * (1 - tf.cast(
        tf.equal(updated_input_ids, inputs.input_ids), tf.int32))
  updated_inputs = pretrain_data.get_updated_inputs(
      inputs, input_ids=updated_input_ids)
  FakedData = collections.namedtuple("FakedData", [
      "inputs", "is_fake_tokens", "sampled_tokens", "pseudo_logprob"])
  return FakedData(inputs=updated_inputs, is_fake_tokens=labels,
                   sampled_tokens=sampled_tokens,
                   pseudo_logprob=pseudo_logprob)

from bunch import Bunch
config = Bunch({})
config.mask_prob = 0.15
config.embedding_size = 768
config.max_predictions_per_seq = 128
config.vocab_file = "./vocab/vocab_ch.txt"
config.num_train_steps = 100
config.initial_ratio = 0.2
config.final_ratio = 0.2

def test_data_generator(features):

  # Mask the input
  unmasked_inputs = pretrain_data.features_to_inputs(features)
  masked_inputs = pretrain_helpers.mask(
    config, unmasked_inputs, config.mask_prob)

  # Generator
  embedding_size = (
    768 if config.embedding_size is None else
    config.embedding_size)
  cloze_output = None

  mlm_output = _get_masked_lm_output(masked_inputs, None)
  fake_data = _get_fake_data(masked_inputs, mlm_output.logits)

  return features, unmasked_inputs, fake_data, masked_inputs

output = ['/Users/xuhaotian/Downloads/chinese_sub_task_0.tfrecord']
input_fn = train_input_fn(output[0], _decode_record, name_to_features)

[features, unmasked_inputs, fake_data, masked_inputs] = test_data_generator(input_fn)
sess = tf.Session()

init_op = tf.group(
            tf.local_variables_initializer(),
            tf.global_variables_initializer())

sess.run(init_op)

while True:
    features_lst = sess.run([unmasked_inputs.input_ids,
                            features['target_ids'],
                            features['target_mask'],
                            fake_data.inputs.input_ids,
                            fake_data.inputs.input_mask,
                            fake_data.inputs.segment_ids,
                            masked_inputs.input_ids,
                            masked_inputs.masked_lm_positions,
                            masked_inputs.masked_lm_ids,
                            masked_inputs.masked_lm_weights])
    break
print(features_lst)
import _pickle as pkl
pkl.dump(features_lst, open("/Users/xuhaotian/Downloads/test_electra.pkl", 'wb'))