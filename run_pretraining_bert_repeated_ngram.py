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

import os, json
from model import modeling_bert
from model import optimization
from repeated_ngram_mask import (data_generator, 
                        model_fn_utils, 
                        data_generator_tpu)
from util import utils, log_utils
from bunch import Bunch

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "buckets", None,
    "buckets. "
    "This specifies the model architecture.")


flags.DEFINE_string(
    "vocab_path", None,
    "vocab path. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "data_path_dict", None,
    "data path dict. "
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

flags.DEFINE_integer("doc_num", 5, "Total batch size for eval.")

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
flags.DEFINE_integer("doc_stride", 64, "doc_stride.")
flags.DEFINE_float("mask_ratio", 0.15, "mask_ratio.")
flags.DEFINE_float("random_ratio", 0.1, "random_ratio.")
flags.DEFINE_integer("min_tok", 3, "min_tok.")
flags.DEFINE_integer("max_tok", 10, "max_tok.")
flags.DEFINE_integer("mask_id", 103, "mask_id.")
flags.DEFINE_integer("sep_id", 102, "sep_id.")
flags.DEFINE_integer("cls_id", 101, "cls_id.")
flags.DEFINE_integer("pad_id", 0, "pad_id.")
flags.DEFINE_float("geometric_p", 0.1, "geometric_p.")
flags.DEFINE_integer("max_pair_targets", 10, "max_pair_targets.")
flags.DEFINE_bool("random_next_sentence", False, "random_next_sentence.")
flags.DEFINE_string('break_mode', 'doc', 'doc break mode')

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
  x_prob = tf.nn.softmax(x_logprobs, axis=-1)
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

def rdropout_model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """ The `model_fn` for TPUEstimator."""

    tf.logging.info("*** *Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    segment_ids = features["segment_ids"]
    input_ids = features['masked_input']
    input_mask = tf.cast(features['input_mask'], dtype=tf.int32)

    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    sent_rel_label_ids = features["sent_rel_label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (
      model,
      masked_lm_loss,
      masked_lm_example_loss,
      sentence_order_loss,
      sentence_order_example_loss,
      masked_lm_log_probs,
      sentence_order_log_probs
    ) = model_fn_utils.model_fn(
            bert_config, 
            input_ids,
            input_mask,
            segment_ids,
            masked_lm_positions,
            masked_lm_ids,
            masked_lm_weights,
            sent_rel_label_ids,
            is_training=is_training,
            number_of_classes=2)

    (
      rdropout_model,
      rdropout_masked_lm_loss,
      rdropout_masked_lm_example_loss,
      rdropout_sentence_order_loss,
      rdropout_sentence_order_example_loss,
      rdropout_masked_lm_log_probs,
      rdropout_sentence_order_log_probs
    ) = model_fn_utils.model_fn(
            bert_config, 
            input_ids,
            input_mask,
            segment_ids,
            masked_lm_positions,
            masked_lm_ids,
            masked_lm_weights,
            sent_rel_label_ids,
            is_training=is_training,
            number_of_classes=2)

    tf.logging.info("** apply rdropout forward **")
    masked_lm_preds = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)
    rdropout_masked_lm_preds = tf.argmax(rdropout_masked_lm_log_probs, axis=-1, output_type=tf.int32)

    (kl_inclusive_per_example_loss,
    kl_inclusive_loss) = kld(rdropout_masked_lm_log_probs,
                      masked_lm_log_probs, 
                      masked_lm_weights)

    (kl_exclusive_per_example_loss,
    kl_exclusive_loss) = kld(masked_lm_log_probs,
                      rdropout_masked_lm_log_probs,
                      masked_lm_weights)

    tf.logging.info("** kl ratio **")
    tf.logging.info(FLAGS.kld_ratio)
    kl_loss = (kl_inclusive_loss+kl_exclusive_loss) * FLAGS.kld_ratio / 2.0

    total_loss = masked_lm_loss + rdropout_masked_lm_loss + kl_loss
    total_loss += (sentence_order_loss + rdropout_sentence_order_loss)
    monitor_dict = {}

    tvars = tf.trainable_variables()
    for tvar in tvars:
      print(tvar, "=====tvar=====")

    eval_fn_inputs = {
        "masked_lm_preds": masked_lm_preds,
        "masked_lm_loss": masked_lm_example_loss,
        "masked_lm_weights": masked_lm_weights,
        "masked_lm_ids": masked_lm_ids,
        "rdropout_masked_lm_loss": rdropout_masked_lm_example_loss,
        "kl_inclusive_loss": kl_inclusive_per_example_loss,
        "kl_exclusive_loss": kl_exclusive_per_example_loss,
        "sentence_order_loss": sentence_order_example_loss,
        "rdropout_sentence_order_loss": rdropout_sentence_order_example_loss
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
      
      #  masked_lm_pred_ids = tf.argmax(masked_lm_preds, axis=-1, 
      #                             output_type=tf.int32)
      masked_lm_acc = tf.cast(tf.equal(masked_lm_preds, masked_lm_ids), dtype=tf.float32)
      masked_lm_acc = tf.reduce_sum(masked_lm_acc*tf.cast(masked_lm_weights, dtype=tf.float32))
      masked_lm_acc /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      masked_lm_loss = tf.reshape(d["masked_lm_loss"], [-1])
      masked_lm_loss = tf.reduce_sum(masked_lm_loss*tf.cast(masked_lm_weights, dtype=tf.float32))
      masked_lm_loss /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      kl_inclusive_loss = tf.reshape(d["kl_inclusive_loss"], [-1])
      kl_inclusive_loss = tf.reduce_sum(kl_inclusive_loss*tf.cast(masked_lm_weights, dtype=tf.float32))
      kl_inclusive_loss /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      kl_exclusive_loss = tf.reshape(d["kl_exclusive_loss"], [-1])
      kl_exclusive_loss = tf.reduce_sum(kl_exclusive_loss*tf.cast(masked_lm_weights, dtype=tf.float32))
      kl_exclusive_loss /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      rdropout_masked_lm_loss = tf.reshape(d["rdropout_masked_lm_loss"], [-1])
      rdropout_masked_lm_loss = tf.reduce_sum(rdropout_masked_lm_loss*tf.cast(masked_lm_weights, dtype=tf.float32))
      rdropout_masked_lm_loss /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      monitor_dict['masked_lm_loss'] = masked_lm_loss
      monitor_dict['masked_lm_acc'] = masked_lm_acc
      monitor_dict['rdropout_masked_lm_loss'] = rdropout_masked_lm_loss
      monitor_dict['kl_inclusive_loss'] = kl_inclusive_loss
      monitor_dict['kl_exclusive_loss'] = kl_exclusive_loss
      
      monitor_dict['rdropout_sop_loss'] = tf.reduce_mean(d['rdropout_sentence_order_loss'])
      monitor_dict['sop_loss'] = tf.reduce_mean(d['sentence_order_loss'])

      return monitor_dict

    monitor_dict = monitor_fn(eval_fn_inputs, eval_fn_keys)

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling_bert.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
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
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          host_call=host_call)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, 
                    masked_lm_log_probs, 
                    masked_lm_ids,
                    masked_lm_weights):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss
          }

      eval_metrics = (metric_fn, [
          masked_lm_example_loss, 
          masked_lm_log_probs, 
          masked_lm_ids,
          masked_lm_weights
      ])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn
  

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** *Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    segment_ids = features["segment_ids"]
    input_ids = features['masked_input']
    input_mask = tf.cast(features['input_mask'], dtype=tf.int32)

    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    sent_rel_label_ids = features["sent_rel_label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (
      model,
      masked_lm_loss,
      masked_lm_example_loss,
      sentence_order_loss,
      sentence_order_example_loss,
      masked_lm_log_probs,
      sentence_order_log_probs
    ) = model_fn_utils.model_fn(
            bert_config, 
            input_ids,
            input_mask,
            segment_ids,
            masked_lm_positions,
            masked_lm_ids,
            masked_lm_weights,
            sent_rel_label_ids,
            is_training=is_training,
            number_of_classes=2)

    masked_lm_preds = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)

    total_loss = masked_lm_loss + sentence_order_loss
    monitor_dict = {}

    tvars = tf.trainable_variables()
    for tvar in tvars:
      print(tvar, "=====tvar=====")

    eval_fn_inputs = {
        "masked_lm_preds": masked_lm_preds,
        "masked_lm_loss": masked_lm_example_loss,
        "masked_lm_weights": masked_lm_weights,
        "masked_lm_ids": masked_lm_ids,
        "sentence_order_loss": sentence_order_example_loss
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
      
      # masked_lm_pred_ids = tf.argmax(masked_lm_preds, axis=-1, 
      #                             output_type=tf.int32)
      masked_lm_acc = tf.cast(tf.equal(masked_lm_preds, masked_lm_ids), dtype=tf.float32)
      masked_lm_acc = tf.reduce_sum(masked_lm_acc*tf.cast(masked_lm_weights, dtype=tf.float32))
      masked_lm_acc /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      masked_lm_loss = tf.reshape(d["masked_lm_loss"], [-1])
      masked_lm_loss = tf.reduce_sum(masked_lm_loss*tf.cast(masked_lm_weights, dtype=tf.float32))
      masked_lm_loss /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      monitor_dict['masked_lm_loss'] = masked_lm_loss
      monitor_dict['masked_lm_acc'] = masked_lm_acc
      monitor_dict['sop_loss'] = tf.reduce_mean(d['sentence_order_loss'])

      return monitor_dict

    monitor_dict = monitor_fn(eval_fn_inputs, eval_fn_keys)

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling_bert.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
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
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          host_call=host_call)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, 
                    masked_lm_log_probs, 
                    masked_lm_ids,
                    masked_lm_weights):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss
          }

      eval_metrics = (metric_fn, [
          masked_lm_example_loss, 
          masked_lm_log_probs, 
          masked_lm_ids,
          masked_lm_weights
      ])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn

# This function is not used by this file but is still used by the Colab and
def input_fn_builder(data_generator,
                    data_path_dict,
                    types,
                    shapes,
                    names,
                    is_training,
                    dataset_merge_method='sample',
                    distributed_mode=None,
                    worker_count=None,
                    task_index=0,
                    data_prior_dict={},
                    use_tpu=False):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  def input_fn(params):
    """The actual input function."""
    if FLAGS.use_tpu:
      batch_size = params["batch_size"]
    else:
      batch_size = FLAGS.train_batch_size

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = data_generator.to_dataset(
            data_path_dict, types, shapes, 
            names=names, 
            padded_batch=False,
            is_training=is_training,
            data_prior_dict=data_prior_dict,
            dataset_merge_method=dataset_merge_method,
            distributed_mode=distributed_mode,
            worker_count=worker_count,
            task_index=task_index)
    # Since we evaluate for a fixed number of steps we don't want to encounter
    # out-of-range exceptions.
    return d

  return input_fn

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling_bert.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

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

  with tf.gfile.Open(FLAGS.data_path_dict, "r") as frobj:
    data_path_dict = json.load(frobj)

  import os

  for key in data_path_dict['data_path']:
    data_path_dict['data_path'][key]['data'] = os.path.join(FLAGS.buckets, data_path_dict['data_path'][key]['data'])
    tf.logging.info("** data path **")
    tf.logging.info(data_path_dict['data_path'][key]['data'])

  data_prior_dict = {}
  for key in data_path_dict['data_path']:
    data_prior_dict[key] = data_path_dict['data_path'][key]['ratio']

  names = ['origin_input',
           'masked_input',
           'input_mask',
           'segment_ids',
           "masked_lm_positions",
           'masked_lm_weights',
           'masked_lm_ids',
           'sent_rel_label_ids'
           ]

  types = [tf.int32]*len(names)
  from tensorflow.python.framework import tensor_shape
  shapes = [
      tensor_shape.TensorShape([FLAGS.max_seq_length]),
      tensor_shape.TensorShape([FLAGS.max_seq_length]),
      tensor_shape.TensorShape([FLAGS.max_seq_length]),
      tensor_shape.TensorShape([FLAGS.max_seq_length]),
      tensor_shape.TensorShape([FLAGS.max_predictions_per_seq]),
      tensor_shape.TensorShape([FLAGS.max_predictions_per_seq]),
      tensor_shape.TensorShape([FLAGS.max_predictions_per_seq]),
      tensor_shape.TensorShape([])
      ]

  import os
  # vocab_path = os.path.join(FLAGS.buckets, FLAGS.vocab_path)
  vocab_path = FLAGS.vocab_path
  data_gen = data_generator_tpu.PretrainGenerator(
      vocab_path=vocab_path,
      batch_size=FLAGS.train_batch_size, 
      buffer_size=1024, 
      do_lower_case=True,
      max_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      mask_ratio=FLAGS.mask_ratio,
      random_ratio=FLAGS.random_ratio,
      min_tok=FLAGS.min_tok,
      max_tok=FLAGS.max_tok,
      mask_id=FLAGS.mask_id,
      cls_id=FLAGS.cls_id,
      sep_id=FLAGS.sep_id,
      pad_id=FLAGS.pad_id,
      geometric_p=FLAGS.geometric_p,
      max_pair_targets=FLAGS.max_pair_targets,
      random_next_sentence=FLAGS.random_next_sentence,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq,
      break_mode=FLAGS.break_mode,
      doc_num=FLAGS.doc_num
  )

  input_fn = input_fn_builder(
          data_generator=data_gen,
          data_path_dict=data_path_dict,
          types=types,
          shapes=shapes,
          names=names,
          is_training=True,
          dataset_merge_method='sample',
          worker_count=1,
          task_index=0,
          distributed_mode=None,
          data_prior_dict=data_prior_dict
          )

  if FLAGS.model_fn_type == 'normal':
    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    tf.logging.info("** normal model fn **")
  elif FLAGS.model_fn_type == 'rdropout':
    model_fn = rdropout_model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    tf.logging.info("** rdropout model fn **")

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    
    estimator.train(input_fn=input_fn, max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    result = estimator.evaluate(
        input_fn=input_fn, steps=FLAGS.max_eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
  # flags.mark_flag_as_required("input_file")
  # flags.mark_flag_as_required("bert_config_file")
  # flags.mark_flag_as_required("output_dir")
  tf.app.run()
