# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow as tf
# tf.disable_v2_behavior()

import tensorflow as tf
# tf.disable_v2_behavior()

def check_tf_version():
  version = tf.__version__
  print("==tf version==", version)
  if int(version.split(".")[0]) >= 2 or int(version.split(".")[1]) >= 15:
    return True
  else:
    return False

if check_tf_version():
  import tensorflow as tf
  tf.disable_v2_behavior()

import os
from model import modeling_relative_position_unilm
from model import optimization
from util import utils, log_utils
from pretrain.span_mask_utils_ilm import _decode_record as span_decode_record
from bunch import Bunch
from model import circle_loss_utils

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "input_data_dir", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("monitoring", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("weight_decay_rate", 0.01, "The initial learning rate for Adam.")
flags.DEFINE_float("lr_decay_power", 1.0, "The initial learning rate for Adam.")
flags.DEFINE_float("layerwise_lr_decay_power", 1.0, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_string("mask_strategy", 'span_mask', "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_float("simcse_ratio", 1.0, "The initial learning rate for Adam.")
flags.DEFINE_float("kld_ratio", 1.0, "The initial learning rate for Adam.")
flags.DEFINE_string("model_fn_type", 'normal', "[Optional] TensorFlow master URL.")
flags.DEFINE_bool("if_simcse", False, "[Optional] TensorFlow master URL.")

def kld(x_logprobs, y_logprobs, mask_weights=None):
  x_prob = tf.exp(x_logprobs)
  tf.logging.info("** x_prob **")
  tf.logging.info(x_prob)
  tf.logging.info("** y_prob **")
  tf.logging.info(y_logprobs)

  kl_per_example_div = x_prob * (x_logprobs - y_logprobs)
  kl_per_example_div = tf.reduce_sum(kl_per_example_div, axis=-1)
  
  tf.logging.info("**  kl_per_example_div **")
  tf.logging.info(kl_per_example_div)

  if mask_weights is not None:
    mask_weights = tf.reshape(mask_weights, [-1])
    kl_div = tf.reduce_mean(kl_per_example_div*mask_weights, axis=0)
  else:
    kl_div = tf.reduce_mean(kl_per_example_div)
  
  tf.logging.info("** kl_div **")
  tf.logging.info(kl_div)

  return kl_per_example_div, kl_div
  
def shape_list(x, out_type=tf.int32):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x, out_type=out_type)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def smooth_labels(labels, factor=0.1):
  # smooth the labels
  labels *= (1 - factor)
  label_shapes = shape_list(labels)
  labels += (factor / label_shapes[-1])
  # returned the smoothed labels
  return labels

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("**** * Input Features *****")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, with shape = %s" % (name, features[name].shape))

    segment_ids = features["segment_ids"]
    input_ids = features['masked_input']
    input_mask = tf.cast(features['pad_mask'], dtype=tf.int32)

    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]

    ilm_input_ids = features['ilm_input']
    ilm_input_mask = features['ilm_input_mask']
    ilm_segment_ids = features['ilm_segment_ids']

    ilm_shape = modeling_relative_position_unilm.get_shape_list(ilm_input_ids)
    input_shape = modeling_relative_position_unilm.get_shape_list(input_ids)
    
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    mlm_model = modeling_relative_position_unilm.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        if_use_unilm=False)

    (masked_lm_loss_onehot,
    masked_lm_loss_labels_smooth,
    masked_lm_example_loss_onehot,
    masked_lm_example_loss_labels_smooth,
    masked_lm_log_probs) = get_masked_lm_output(
         bert_config, mlm_model.get_sequence_output(), 
         mlm_model.get_embedding_table(),
         masked_lm_positions, 
         masked_lm_ids, 
         masked_lm_weights)

    masked_lm_preds = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)

    ilm_model = modeling_relative_position_unilm.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=ilm_input_ids,
        input_mask=ilm_input_mask,
        token_type_ids=ilm_segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        if_use_unilm=True)

    (ilm_loss_onehot, 
    ilm_loss_labels_smooth,
    ilm_per_example_loss_onehot, 
    ilm_per_example_loss_labels_smooth,
    ilm_log_probs) = get_lm_output(bert_config, 
                  ilm_model.get_sequence_output()[:, :-1, :], 
                  ilm_model.get_embedding_table(), 
                  label_ids=ilm_input_ids[:, 1:], 
                  label_mask=ilm_segment_ids[:, 1:])

    tf.logging.info("** ilm_model sequence_output **")
    tf.logging.info(ilm_model.get_sequence_output()[:, :-1, :])

    tf.logging.info("** ilm_model ilm_input_ids **")
    tf.logging.info(ilm_input_ids[:, 1:])

    tf.logging.info("** ilm_model ilm_segment_ids **")
    tf.logging.info(ilm_segment_ids[:, 1:])

    tf.logging.info("** ilm_model ilm_log_probs **")
    tf.logging.info(ilm_log_probs)

    ilm_preds = tf.argmax(ilm_log_probs, axis=-1, output_type=tf.int32)

    tf.logging.info("** ilm_model ilm_preds **")
    tf.logging.info(ilm_preds)

    total_loss = (masked_lm_loss_onehot + ilm_loss_onehot)
    monitor_total_loss = (masked_lm_loss_onehot + ilm_loss_onehot)
    monitor_dict = {}

    tvars = tf.trainable_variables()
    for tvar in tvars:
      print(tvar, "=====tvar=====")

    eval_fn_inputs = {
        "masked_lm_preds": masked_lm_preds,
        "masked_lm_loss": masked_lm_example_loss_labels_smooth,
        "masked_lm_weights": masked_lm_weights,
        "masked_lm_ids": masked_lm_ids,
        "ilm_preds": ilm_preds,
        "ilm_ids": ilm_input_ids[:, 1:],
        "ilm_weights": ilm_segment_ids[:, 1:],
        "ilm_loss": ilm_per_example_loss_labels_smooth
    }

    eval_fn_keys = eval_fn_inputs.keys()
    eval_fn_values = [eval_fn_inputs[k] for k in eval_fn_keys]

    def monitor_fn(eval_fn_inputs, keys):
      # d = {k: arg for k, arg in zip(eval_fn_keys, args)}
      d = {}
      for key in eval_fn_inputs:
        if key in keys:
          d[key] = eval_fn_inputs[key]
      monitor_dict = dict()
      masked_lm_ids = tf.reshape(d["masked_lm_ids"], [-1])
      masked_lm_preds = tf.reshape(d["masked_lm_preds"], [-1])
      masked_lm_weights = tf.reshape(d["masked_lm_weights"], [-1])
      
      masked_lm_acc = tf.cast(tf.equal(masked_lm_preds, masked_lm_ids), dtype=tf.float32)
      masked_lm_acc = tf.reduce_sum(masked_lm_acc*tf.cast(masked_lm_weights, dtype=tf.float32))
      masked_lm_acc /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      masked_lm_loss = tf.reshape(d["masked_lm_loss"], [-1])
      masked_lm_loss = tf.reduce_sum(masked_lm_loss*tf.cast(masked_lm_weights, dtype=tf.float32))
      masked_lm_loss /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      monitor_dict['masked_lm_loss'] = masked_lm_loss
      monitor_dict['masked_lm_acc'] = masked_lm_acc

      ilm_ids = tf.reshape(d["ilm_ids"], [-1])
      ilm_preds = tf.reshape(d["ilm_preds"], [-1])
      ilm_weights = tf.reshape(d["ilm_weights"], [-1])
      ilm_acc = tf.cast(tf.equal(ilm_preds, ilm_ids), dtype=tf.float32)
      ilm_acc = tf.reduce_sum(ilm_acc*tf.cast(ilm_weights, dtype=tf.float32))
      ilm_acc /= (1e-10+tf.reduce_sum(tf.cast(ilm_weights, dtype=tf.float32)))

      ilm_loss = tf.reshape(d["ilm_loss"], [-1])
      ilm_loss = tf.reduce_sum(ilm_loss*tf.cast(ilm_weights, dtype=tf.float32))
      ilm_loss /= (1e-10+tf.reduce_sum(tf.cast(ilm_weights, dtype=tf.float32)))

      monitor_dict['ilm_loss'] = ilm_loss
      monitor_dict['ilm_acc'] = ilm_acc
      monitor_dict['ilm_weights'] = tf.reduce_mean(tf.reduce_sum(d["ilm_weights"], axis=-1))

      return monitor_dict

    monitor_dict = monitor_fn(eval_fn_inputs, eval_fn_keys)

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling_relative_position_unilm.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op, output_learning_rate = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, 
          weight_decay_rate=FLAGS.weight_decay_rate,
          use_tpu=use_tpu,
          warmup_steps=num_warmup_steps,
          lr_decay_power=FLAGS.lr_decay_power)

      monitor_dict['learning_rate'] = output_learning_rate
      if FLAGS.monitoring and monitor_dict:
        host_call = log_utils.construct_scalar_host_call_v1(
                                    monitor_dict=monitor_dict,
                                    model_dir=FLAGS.output_dir,
                                    prefix="train/")
      else:
        host_call = None

      print(host_call, "====host_call====")

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=monitor_total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          host_call=host_call)

    return output_spec

  return model_fn

def get_lm_output(config, input_tensor, output_weights, label_ids, label_mask):
  """Get loss and log probs for the LM."""
  with tf.variable_scope("cls/predictions", reuse=tf.AUTO_REUSE):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=config.hidden_size,
          activation=modeling_relative_position_unilm.get_activation(config.hidden_act),
          kernel_initializer=modeling_relative_position_unilm.create_initializer(config.initializer_range))
      input_tensor = modeling_relative_position_unilm.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    logits_shape = modeling_relative_position_unilm.get_shape_list(logits, expected_rank=3)
    logits = tf.reshape(logits, [logits_shape[0]*logits_shape[1], logits_shape[2]])
    log_probs = tf.reshape(log_probs, [logits_shape[0]*logits_shape[1], logits_shape[2]])

    label_ids = tf.reshape(label_ids, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=config.vocab_size, dtype=tf.float32)
    one_hot_labels_smooth = smooth_labels(one_hot_labels)

    # per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #                     labels=label_ids, 
    #                     logits=logits)

    label_mask = tf.reshape(label_mask, [-1])
    loss_mask = tf.cast(label_mask, tf.float32)

    per_example_loss_labels_smooth = -tf.reduce_sum(log_probs * one_hot_labels_smooth, axis=[-1])

    numerator_labels_smooth = tf.reduce_sum(loss_mask * per_example_loss_labels_smooth)
    denominator_labels_smooth = tf.reduce_sum(loss_mask) + 1e-5
    loss_labels_smooth = numerator_labels_smooth / (denominator_labels_smooth)

    per_example_loss_onehot = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator_onehot = tf.reduce_sum(loss_mask * per_example_loss_onehot)
    denominator_onehot = tf.reduce_sum(loss_mask) + 1e-5
    loss_onehot = numerator_onehot / denominator_onehot

    print(log_probs, '==ilm log_probs==')
    print(label_ids, '==ilm label_ids==')
    print(loss_mask, '==ilm loss_mask==')
    print(per_example_loss_labels_smooth, '==ilm per_example_loss_labels_smooth==')
    print(loss_labels_smooth, '==ilm loss_labels_smooth==')

    print(per_example_loss_onehot, '==ilm per_example_loss_onehot==')
    print(loss_onehot, '==ilm loss_onehot==')

  return (loss_onehot, loss_labels_smooth, 
        per_example_loss_onehot, 
        per_example_loss_labels_smooth,
        log_probs)

def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions", reuse=tf.AUTO_REUSE):
    # We apply one more non-linear  transformation before the output layer.
    # This matrix is not used after  pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling_relative_position_unilm.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling_relative_position_unilm.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling_relative_position_unilm.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    one_hot_labels_smooth = smooth_labels(one_hot_labels)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss_labels_smooth = -tf.reduce_sum(log_probs * one_hot_labels_smooth, axis=[-1])
    numerator_labels_smooth = tf.reduce_sum(label_weights * per_example_loss_labels_smooth)
    denominator_labels_smooth = tf.reduce_sum(label_weights) + 1e-5
    loss_labels_smooth = numerator_labels_smooth / denominator_labels_smooth

    per_example_loss_onehot = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator_onehot = tf.reduce_sum(label_weights * per_example_loss_onehot)
    denominator_onehot = tf.reduce_sum(label_weights) + 1e-5
    loss_onehot = numerator_onehot / denominator_onehot

    print(log_probs, '==mlm log_probs==')
    print(label_ids, '==mlm label_ids==')
    print(label_weights, '==mlm label_weights==')
    print(per_example_loss_labels_smooth, '==mlm per_example_loss_labels_smooth==')
    print(loss_labels_smooth, '==mlm loss==')

    print(per_example_loss_onehot, '==mlm per_example_loss_onehot==')
    print(loss_onehot, '==mlm loss==')

  return (loss_onehot, loss_labels_smooth, 
        per_example_loss_onehot, 
        per_example_loss_labels_smooth,
        log_probs)

def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling_relative_position_unilm.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor

data_config = Bunch({})
data_config.min_tok = 2
data_config.max_tok = 10
data_config.sep_id = 102
data_config.pad_id = 0
data_config.cls_id = 101
data_config.mask_id = 103
data_config.leak_ratio = 0.1
data_config.rand_ratio = 0.1
data_config.mask_prob = 0.15
data_config.sample_strategy = 'token_span'
data_config.truncate_seq = False
data_config.stride = 1
data_config.p = 0.1
data_config.use_bfloat16 = False
data_config.max_predictions_per_seq = int(data_config.mask_prob*(FLAGS.max_seq_length))
data_config.max_seq_length = FLAGS.max_seq_length
data_config.initial_ratio = 0.15
data_config.final_ratio = 0.15
data_config.num_train_steps = FLAGS.num_train_steps
data_config.seg_id = 105 # <S>
data_config.ilm_v1 = False
data_config.ilm_v2 = True

def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     vocab_size,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  data_config.vocab_size = vocab_size
  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ori_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
    }

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
            lambda record: span_decode_record(data_config, record, 
                data_config.max_predictions_per_seq,
                data_config.max_seq_length, 
                record_spec=name_to_features,
                input_ids_name='input_ori_ids',
                use_bfloat16=data_config.use_bfloat16, 
                truncate_seq=data_config.truncate_seq, 
                stride=data_config.stride),
          batch_size=batch_size,
          num_parallel_batches=num_cpu_threads,
          drop_remainder=True))
    # d = d.apply(tf.data.experimental.ignore_errors())
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError(" At least one of  `do_train` or `do_eval` must be True.")

  bert_config = modeling_relative_position_unilm.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files = []
  import os
  input_file = os.path.join(FLAGS.input_data_dir, FLAGS.input_file)
  with tf.gfile.GFile(input_file, "r") as reader:
    for index, line in enumerate(reader):
        content = line.strip()
        if 'tfrecord' in content:
            train_file_path = os.path.join(FLAGS.input_data_dir, content)
            # print(train_file_path, "====train_file_path====")
            input_files.append(train_file_path)
  print("===total input files===", len(input_files))
  # for input_pattern in FLAGS.input_file.split(","):
  #   input_files.extend(tf.gfile.Glob(input_pattern))

  import random
  random.shuffle(input_files)

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  if FLAGS.init_checkpoint:
    init_checkpoint = os.path.join(FLAGS.input_data_dir, FLAGS.init_checkpoint)
  else:
    init_checkpoint = FLAGS.init_checkpoint

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    tf.logging.info("******* Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        vocab_size=bert_config.vocab_size,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

if __name__ == "__main__":
  tf.app.run()
