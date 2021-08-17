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

"""Pre-trains an ELECTRA model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
from model import model_io as model_io_fn

# import tensorflow.compat.v1 as tf
# import tensorflow as tf
# tf.disable_v2_behavior()

import tensorflow as tf

def check_tf_version():
  version = tf.__version__
  print("==tf version==", version)
  if int(version.split(".")[0]) >= 2 or int(version.split(".")[1]) >= 15:
    return True
  else:
    return False
if check_tf_version():
  tf.disable_v2_behavior()

def shape_list(x, out_type=tf.int32):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x, out_type=out_type)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


import configure_pretraining
from model import modeling
from model import modeling_tta
from model import modeling_convbert
from model import optimization
from pretrain import pretrain_data
from pretrain import gumbel_softmax_sampling
from pretrain import pretrain_helpers
from util import training_utils
from util import utils, log_utils
from pretrain import span_mask_utils
from model import spectural_utils

class PretrainingModel(object):
  """Transformer pre-training using the replaced-token-detection task."""

  def __init__(self, config,
               features, is_training):
    # Set up model config
    self._config = config
    self._bert_config = training_utils.get_bert_config(config)
    self._generator_config = training_utils.get_bert_generator_config(config)
    # self._bert_config_generator = training_utils.get_bert_generator_config(config)
    self.is_training = is_training
    if config.debug:
      self._bert_config.num_hidden_layers = 3
      self._bert_config.hidden_size = 144
      self._bert_config.intermediate_size = 144 * 4
      self._bert_config.num_attention_heads = 4

    # Mask the input
    if self._config.mask_strategy == 'electra':
      unmasked_inputs = pretrain_data.features_to_inputs(features)
      masked_inputs = pretrain_helpers.mask(
          config, unmasked_inputs, config.mask_prob)
      print("==apply electra random mask strategy==")
    elif self._config.mask_strategy == 'span_mask':
      unmasked_inputs = span_mask_utils.features_to_inputs(features)
      masked_inputs = span_mask_utils.mask(
          config, unmasked_inputs, config.mask_prob,
          features=features)
      print("==apply span_mask random mask strategy==")

    sampled_masked_inputs = pretrain_helpers.mask(
          config, unmasked_inputs, config.mask_prob)
    print("==apply unigram-span_mask random mask strategy==")

    self.monitor_dict = {}
    tf.logging.info("** untied_generator **")
    tf.logging.info(config.untied_generator)
    if config.untied_generator:
      if self._config.use_pretrained_generator:
        self.generator_scope = 'generator/bert'
        self.generator_cls_scope = 'generator/cls/predictions'
        self.generator_cloze_scope = 'generator/cls/predictions'
        self.generator_exclude_scope = 'generator'
        self.generator_embedding_size = (
          self._generator_config.hidden_size if config.generator_embedding_size is None else
          config.embedding_size)
        self.untied_generator_embeddings = True
        tf.logging.info("==apply pretrained generator==")
      else:
        self.generator_scope = 'generator'
        self.generator_cls_scope = 'generator_predictions'
        self.generator_cloze_scope = 'cloze_predictions'
        self.generator_embedding_size = (
          self._bert_config.hidden_size if config.embedding_size is None else
          config.embedding_size)
        self.untied_generator_embeddings = config.untied_generator_embeddings
        self.generator_exclude_scope = ''
      if self._config.use_pretrained_discriminator:
        self.discriminator_scope = 'discriminator/bert'
        self.discriminator_exclude_scope = 'discriminator'
        self.discriminator_cls_scope = 'discriminator/cls/predictions'
        self.discriminator_embedding_size = (
          self._bert_config.hidden_size if config.embedding_size is None else
          config.embedding_size)
        self.untied_discriminator_embeddings = True
      else:
        self.discriminator_scope = 'electra'
        self.discriminator_embedding_size = (
          self._bert_config.hidden_size if config.embedding_size is None else
          config.embedding_size)
        self.discriminator_exclude_scope = ''
        self.untied_discriminator_embeddings = config.untied_generator_embeddings
        self.discriminator_cls_scope = 'electra_predictions'
    else:
      if self._config.use_pretrained_generator or self._config.use_pretrained_discriminator:
        self.generator_scope = 'discriminator/bert'
        self.generator_cls_scope = 'discriminator/cls/predictions'
        self.generator_cloze_scope = 'discriminator/cls/predictions'
        self.discriminator_scope = 'discriminator/bert'
        self.discriminator_exclude_scope = 'discriminator'
        self.discriminator_cls_scope = 'discriminator/cls/predictions'
        self.generator_exclude_scope = 'discriminator'
        self.discriminator_embedding_size = (
          self._bert_config.hidden_size if config.embedding_size is None else
          config.embedding_size)
        self.generator_embedding_size = (
          self._bert_config.hidden_size if config.embedding_size is None else
          config.embedding_size)
        self.untied_generator_embeddings = True
        self.untied_discriminator_embeddings = True
      else:
        self.discriminator_scope = 'electra'
        self.generator_scope = 'electra'
        self.generator_cls_scope = 'generator_predictions'
        self.generator_cloze_scope = 'cloze_predictions'
        self.discriminator_embedding_size = (
          self._bert_config.hidden_size if config.embedding_size is None else
          config.embedding_size)
        self.generator_embedding_size = (
          self._bert_config.hidden_size if config.embedding_size is None else
          config.embedding_size)
        self.discriminator_exclude_scope = ''
        self.discriminator_cls_scope = 'electra_predictions'
        self.generator_exclude_scope = ''
        self.untied_generator_embeddings = True
        self.untied_discriminator_embeddings = True

    if config.generator_transformer_type == 'bert':
      generator_fn = build_transformer
    elif config.generator_transformer_type == 'conv_bert':
      generator_fn = build_conv_transformer
    elif config.generator_transformer_type == 'tta':
      generator_fn = build_tta_transformer
    else:
      generator_fn = build_transformer

    if config.discriminator_transformer_type == 'bert':
      discriminator_fn = build_transformer
    elif config.discriminator_transformer_type == 'conv_bert':
      discriminator_fn = build_conv_transformer
    elif config.discriminator_transformer_type == 'tta':
      discriminator_fn = build_tta_transformer
    else:
      discriminator_fn = build_transformer

    # Generator
    if config.uniform_generator:
      # simple generator sampling fakes uniformly at random
      # mlm_output = self._get_masked_lm_output(masked_inputs, None)
      mlm_output = self._get_masked_lm_output(masked_inputs, None)
      self.gen_params = []
    elif ((config.electra_objective or config.electric_objective or config.electra_nce_objective)
          and config.untied_generator):
      # generator_config = get_generator_config(config, self._bert_config)
      if config.two_tower_generator:
        # two-tower cloze model generator used for electric
        generator = TwoTowerClozeTransformer(
            config, self._generator_config, unmasked_inputs, is_training,
            self.generator_embedding_size)
        cloze_output = self._get_cloze_outputs(unmasked_inputs, generator)
        mlm_output = get_softmax_output(
            pretrain_helpers.gather_positions(
                cloze_output.logits, masked_inputs.masked_lm_positions),
            masked_inputs.masked_lm_ids, masked_inputs.masked_lm_weights,
            self._bert_config.vocab_size)

        self.gen_params = []
        self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator_ltr')
        self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator_rtl')
        self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_cloze_scope)
        if not config.untied_generator_embeddings:
          self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.discriminator_scope+"/embeddings")
      else:
        # small masked language model generator
        generator = generator_fn(
            config, masked_inputs, is_training, self._generator_config,
            embedding_size=self.generator_embedding_size,
            untied_embeddings=self.untied_generator_embeddings,
            scope=self.generator_scope,
            reuse=tf.AUTO_REUSE)
        tf.logging.info("** apply generator transformer **")
        tf.logging.info(generator_fn)
        mlm_output = self._get_masked_lm_output(masked_inputs, generator, self.generator_cls_scope)

        self.gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_scope)
        self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_cls_scope)
        if not config.untied_generator_embeddings:
          tf.logging.info("** add shared embeddings to generator **")
          self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.discriminator_scope+"/embeddings")
    else:
      # full-sized masked language model generator if using BERT objective or if
      # the generator and discriminator have tied weights
      generator = generator_fn(
          config, masked_inputs, is_training, self._bert_config,
          embedding_size=self.discriminator_embedding_size,
          untied_embeddings=self.untied_generator_embeddings,
          scope=self.generator_scope)
      tf.logging.info("** apply generator transformer **")
      tf.logging.info(generator_fn)
      
      mlm_output = self._get_masked_lm_output(masked_inputs, generator, self.generator_cls_scope)

      self.gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_scope)
      self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_cls_scope)
    
    fake_data = self._get_fake_data(masked_inputs, mlm_output.logits, config.straight_through)
    self.mlm_output = mlm_output
    if config.two_tower_generator:
      self.total_loss = config.gen_weight * cloze_output.loss
      tf.logging.info("** apply cloze loss **")
    else:
      self.total_loss = config.gen_weight * mlm_output.loss
      tf.logging.info("** apply mlm loss **")

    if config.two_tower_generator:
      self.gen_loss = cloze_output.loss
    else:
      self.gen_loss = mlm_output.loss
    # Discriminator
    disc_output = None
    if config.electra_objective or config.electric_objective:
      fake_discriminator = discriminator_fn(
          config, fake_data.inputs, is_training, self._bert_config,
          reuse=tf.AUTO_REUSE, 
          embedding_size=self.discriminator_embedding_size,
          untied_embeddings=self.untied_discriminator_embeddings,
          scope=self.discriminator_scope,
          if_reuse_dropout=True)
      tf.logging.info("** apply transformer **")
      tf.logging.info(discriminator_fn)
      
      disc_output = self._get_discriminator_output_v1(
          fake_data.inputs, fake_discriminator, fake_data.is_fake_tokens,
          scope=self.discriminator_cls_scope,
          cloze_output=mlm_output)

      self.disc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.discriminator_scope)
      self.disc_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.discriminator_cls_scope)
      self.disc_loss = disc_output.loss

      self.total_loss += config.disc_weight * disc_output.loss
      
    if config.contras:
      real_discriminator = discriminator_fn(
            config, unmasked_inputs, is_training, self._bert_config,
            reuse=tf.AUTO_REUSE, 
            embedding_size=self.discriminator_embedding_size,
            untied_embeddings=self.untied_discriminator_embeddings,
            scope=self.discriminator_scope,
            if_reuse_dropout=True)
      fake_view = fake_discriminator.get_pooled_output()
      fake_view = tf.nn.l2_normalize(fake_view, axis=-1)

      real_view = real_discriminator.get_pooled_output()
      real_view = tf.nn.l2_normalize(real_view, axis=-1)

      from model import circle_loss_utils
      sim_matrix = tf.matmul(fake_view, 
                        real_view, 
                        transpose_b=True)

      sim_matrix_shape = shape_list(sim_matrix)
      pos_true_mask = tf.eye(sim_matrix_shape[0])
      pos_true_mask = tf.cast(pos_true_mask, dtype=tf.float32)
      neg_true_mask = tf.ones_like(pos_true_mask) - pos_true_mask

      sim_per_example_loss = circle_loss_utils.circle_loss(
                              sim_matrix, 
                              pos_true_mask, 
                              neg_true_mask,
                              margin=0.25,
                              gamma=32)
      simcse_loss = tf.reduce_mean(sim_per_example_loss)
      self.total_loss += simcse_loss * config.simcse_ratio

    # Evaluation
    eval_fn_inputs = {
        "input_ids": masked_inputs.input_ids,
        "masked_lm_preds": mlm_output.preds,
        "mlm_loss": mlm_output.per_example_loss,
        "masked_lm_ids": masked_inputs.masked_lm_ids,
        "masked_lm_weights": masked_inputs.masked_lm_weights,
        "input_mask": masked_inputs.input_mask,
        "sampled_masked_lm_preds":mlm_output.preds,
        "sampled_masked_lm_ids":masked_inputs.masked_lm_ids,
        "sampled_masked_lm_weights":masked_inputs.masked_lm_weights
    }
    if config.electra_objective or config.electric_objective:
      eval_fn_inputs.update({
          "disc_loss": disc_output.per_example_loss,
          "disc_labels": disc_output.labels,
          "disc_probs": disc_output.probs,
          "disc_preds": disc_output.preds,
          "sampled_tokids": tf.argmax(fake_data.sampled_tokens, -1,
                                      output_type=tf.int32)
      })
    if config.contras:
      eval_fn_inputs.update({
          "simcse_loss": simcse_loss
      })
    
    eval_fn_keys = eval_fn_inputs.keys()
    eval_fn_values = [eval_fn_inputs[k] for k in eval_fn_keys]

    def electra_electric_monitor_fn(eval_fn_inputs, keys):
      d = {}
      for key in eval_fn_inputs:
        if key in keys:
          d[key] = eval_fn_inputs[key]
      monitor_dict = dict()
      masked_lm_ids = tf.reshape(d["masked_lm_ids"], [-1])
      masked_lm_preds = tf.reshape(d["masked_lm_preds"], [-1])
      masked_lm_weights = tf.reshape(d["masked_lm_weights"], [-1])

      sampled_masked_lm_ids = tf.reshape(d["sampled_masked_lm_ids"], [-1])
      sampled_masked_lm_preds = tf.reshape(d["sampled_masked_lm_preds"], [-1])
      sampled_masked_lm_weights = tf.reshape(d["sampled_masked_lm_weights"], [-1])

      mlm_acc = tf.cast(tf.equal(masked_lm_preds, masked_lm_ids), dtype=tf.float32)
      mlm_acc = tf.reduce_sum(mlm_acc*tf.cast(masked_lm_weights, dtype=tf.float32))
      mlm_acc /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      mlm_loss = tf.reshape(d["mlm_loss"], [-1])
      mlm_loss = tf.reduce_sum(mlm_loss*tf.cast(masked_lm_weights, dtype=tf.float32))
      mlm_loss /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      monitor_dict['generator_mlm_loss'] = mlm_loss
      monitor_dict['generator_mlm_acc'] = mlm_acc

      sampled_lm_ids = tf.reshape(d["sampled_masked_lm_ids"], [-1])
      sampled_lm_pred_ids = tf.reshape(d["sampled_tokids"], [-1])
      sampeld_mlm_acc = tf.cast(tf.equal(sampled_lm_pred_ids, sampled_lm_ids), dtype=tf.float32)
      sampeld_mlm_acc = tf.reduce_sum(sampeld_mlm_acc*tf.cast(sampled_masked_lm_weights, dtype=tf.float32))
      sampeld_mlm_acc /= (1e-10+tf.reduce_sum(tf.cast(sampled_masked_lm_weights, dtype=tf.float32)))

      monitor_dict['generator_sampled_mlm_acc'] = sampeld_mlm_acc

      token_acc = tf.cast(tf.equal(d["disc_preds"], d['disc_labels']),
                                dtype=tf.float32)
      token_acc_mask = tf.cast(d["input_mask"], dtype=tf.float32)
      token_acc *= token_acc_mask
      token_acc = tf.reduce_sum(token_acc) / (1e-10+tf.reduce_sum(token_acc_mask))

      monitor_dict['disriminator_token_acc'] = token_acc
      monitor_dict['disriminator_token_loss'] = tf.reduce_mean(d['disc_loss'])

      token_precision = tf.cast(tf.equal(d["disc_preds"], d['disc_labels']),
                                dtype=tf.float32)
      token_precision_mask = tf.cast(d["disc_preds"] * d["input_mask"], dtype=tf.float32)
      token_precision *= token_precision_mask
      token_precision = tf.reduce_sum(token_precision) / (1e-10+tf.reduce_sum(token_precision_mask))

      monitor_dict['disriminator_token_precision'] = token_precision

      token_recall = tf.cast(tf.equal(d["disc_preds"], d['disc_labels']),
                                dtype=tf.float32)
      token_recall_mask = tf.cast(d["disc_labels"] * d["input_mask"], dtype=tf.float32)
      token_recall *= token_recall_mask
      token_recall = tf.reduce_sum(token_recall) / (1e-10+tf.reduce_sum(token_recall_mask))

      monitor_dict['disriminator_token_recall'] = token_recall
      monitor_dict['simcse_loss'] = d['simcse_loss']
      return monitor_dict

    self.monitor_dict = electra_electric_monitor_fn(eval_fn_inputs, eval_fn_keys)
    
    def metric_fn(*args):
      """Computes the loss and accuracy of the model."""
      d = {k: arg for k, arg in zip(eval_fn_keys, args)}
      metrics = dict()
      metrics["masked_lm_accuracy"] = tf.metrics.accuracy(
          labels=tf.reshape(d["masked_lm_ids"], [-1]),
          predictions=tf.reshape(d["masked_lm_preds"], [-1]),
          weights=tf.reshape(d["masked_lm_weights"], [-1]))
      metrics["masked_lm_loss"] = tf.metrics.mean(
          values=tf.reshape(d["mlm_loss"], [-1]),
          weights=tf.reshape(d["masked_lm_weights"], [-1]))
      if config.electra_objective or config.electric_objective:
        metrics["sampled_masked_lm_accuracy"] = tf.metrics.accuracy(
            labels=tf.reshape(d["masked_lm_ids"], [-1]),
            predictions=tf.reshape(d["sampled_tokids"], [-1]),
            weights=tf.reshape(d["masked_lm_weights"], [-1]))
        if config.disc_weight > 0:
          metrics["disc_loss"] = tf.metrics.mean(d["disc_loss"])
          metrics["disc_auc"] = tf.metrics.auc(
              d["disc_labels"] * d["input_mask"],
              d["disc_probs"] * tf.cast(d["input_mask"], tf.float32))
          metrics["disc_accuracy"] = tf.metrics.accuracy(
              labels=d["disc_labels"], predictions=d["disc_preds"],
              weights=d["input_mask"])
          metrics["disc_precision"] = tf.metrics.accuracy(
              labels=d["disc_labels"], predictions=d["disc_preds"],
              weights=d["disc_preds"] * d["input_mask"])
          metrics["disc_recall"] = tf.metrics.accuracy(
              labels=d["disc_labels"], predictions=d["disc_preds"],
              weights=d["disc_labels"] * d["input_mask"])
      return metrics
    self.eval_metrics = (metric_fn, eval_fn_values)

  def _get_masked_lm_output(self, inputs, model, scope='', reuse=None):
    """Masked language modeling softmax layer."""
    # if scope:
    #   pretrained_scope = scope + 'cls/predictions'
    # else:
    #   pretrained_scope = 'generator_predictions'
    with tf.variable_scope(scope if scope else 'generator_predictions',
                reuse=tf.AUTO_REUSE):
      if self._config.uniform_generator:
        logits = tf.zeros(self._bert_config.vocab_size)
        logits_tiled = tf.zeros(
            modeling.get_shape_list(inputs.masked_lm_ids) +
            [self._bert_config.vocab_size])
        logits_tiled += tf.reshape(logits, [1, 1, self._bert_config.vocab_size])
        logits = logits_tiled
      else:
        relevant_reprs = pretrain_helpers.gather_positions(
            model.get_sequence_output(), inputs.masked_lm_positions)
        # logtis: [batch_size, num_masked, vocab]
        logits = get_token_logits(
            relevant_reprs, model.get_embedding_table(), self._bert_config)
      return get_softmax_output(
          logits, inputs.masked_lm_ids, inputs.masked_lm_weights,
          self._bert_config.vocab_size)
 
  def _get_discriminator_output(
      self, inputs, discriminator, labels, scope='', cloze_output=None):
    """Discriminator binary classifier."""
    with tf.variable_scope(scope if scope else "discriminator_predictions"):
      hidden = tf.layers.dense(
          discriminator.get_sequence_output(),
          units=self._bert_config.hidden_size,
          activation=modeling.get_activation(self._bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              self._bert_config.initializer_range))
      logits = tf.squeeze(tf.layers.dense(hidden, units=1), -1)
      if self._config.electric_objective:
        log_q = tf.reduce_sum(
            tf.nn.log_softmax(cloze_output.logits) * tf.one_hot(
                inputs.input_ids, depth=self._bert_config.vocab_size,
                dtype=tf.float32), -1)
        log_q = tf.stop_gradient(log_q)
        logits += log_q
        logits += tf.log(self._config.mask_prob / (1 - self._config.mask_prob))
        tf.logging.info("==apply electric_objective==")

      weights = tf.cast(inputs.input_mask, tf.float32)
      labelsf = tf.cast(labels, tf.float32)
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          logits=logits, labels=labelsf) * weights
      per_example_loss = (tf.reduce_sum(losses, axis=-1) /
                          (1e-6 + tf.reduce_sum(weights, axis=-1)))
      loss = tf.reduce_sum(losses) / (1e-6 + tf.reduce_sum(weights))
      probs = tf.nn.sigmoid(logits)
      preds = tf.cast(tf.round((tf.sign(logits) + 1) / 2), tf.int32)
      DiscOutput = collections.namedtuple(
          "DiscOutput", ["loss", "per_example_loss", "probs", "preds",
                         "labels"])
      return DiscOutput(
          loss=loss, per_example_loss=per_example_loss, probs=probs,
          preds=preds, labels=labels,
      )

  def _get_discriminator_output_v1(
      self, inputs, discriminator, labels, scope='', cloze_output=None):
    """Discriminator binary classifier."""
    with tf.variable_scope(scope if scope else "discriminator_predictions"):
      hidden = tf.layers.dense(
          discriminator.get_sequence_output(),
          units=shape_list(discriminator.get_embedding_table())[-1],
          activation=modeling.get_activation(self._bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              self._bert_config.initializer_range))
      
      extra_weight = tf.get_variable(
          name="extra_weight",
          shape=[1, shape_list(discriminator.get_embedding_table())[-1]],
          initializer=tf.zeros_initializer())

      output_bias = tf.get_variable(
        "output_bias",
        shape=[self._bert_config.vocab_size],
        initializer=tf.zeros_initializer())

      # [vocab_size+1, embedding_size]
      weight_matrix = tf.concat([discriminator.get_embedding_table(), extra_weight],
                    axis=0)

      # hidden: [batch_size, seq, hidden_dims]
      # [batch_size, seq, vocab-size+1]
      if check_tf_version():
        logits = tf.matmul(hidden, weight_matrix, transpose_b=True)
      else:
        logits = tf.einsum("abc,dc->abd", hidden, weight_matrix)
      logits += output_bias
      # being negative-token logits
      logits = tf.nn.log_softmax(logits, axis=-1)[:, :, -1]

      if self._config.electric_objective:
        log_q = tf.reduce_sum(
            tf.nn.log_softmax(cloze_output.logits) * tf.one_hot(
                inputs.input_ids, depth=self._bert_config.vocab_size,
                dtype=tf.float32), -1)
        log_q = tf.stop_gradient(log_q)
        logits += log_q
        logits += tf.log(self._config.mask_prob / (1 - self._config.mask_prob))
        tf.logging.info("==apply electric_objective==")

      weights = tf.cast(inputs.input_mask, tf.float32)
      labelsf = tf.cast(labels, tf.float32)
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          logits=logits, labels=labelsf) * weights
      per_example_loss = (tf.reduce_sum(losses, axis=-1) /
                          (1e-6 + tf.reduce_sum(weights, axis=-1)))
      loss = tf.reduce_sum(losses) / (1e-6 + tf.reduce_sum(weights))
      probs = tf.nn.sigmoid(logits)
      preds = tf.cast(tf.round((tf.sign(logits) + 1) / 2), tf.int32)
      DiscOutput = collections.namedtuple(
          "DiscOutput", ["loss", "per_example_loss", "probs", "preds",
                         "labels"])
      return DiscOutput(
          loss=loss, per_example_loss=per_example_loss, probs=probs,
          preds=preds, labels=labels,
      )

  def _get_fake_data(self, inputs, mlm_logits, straight_through=True):
    """Sample from the generator to create corrupted input."""
    masked_lm_weights = inputs.masked_lm_weights
    inputs = pretrain_helpers.unmask(inputs)
    disallow = tf.one_hot(
        inputs.masked_lm_ids, depth=self._bert_config.vocab_size,
        dtype=tf.float32) if self._config.disallow_correct else None
    if self._config.stop_gradient:
      fn = tf.stop_gradient
      tf.logging.info("** stop disc gradient **")
    else:
      fn = tf.identity
      tf.logging.info("** enable disc gradient **")
    if self._config.fake_data_sample == 'sample_from_softmax':
      sampled_tokens = fn(gumbel_softmax_sampling.sample_from_softmax(
        mlm_logits, self._config.logits_temp, 
        self._config.gumbel_temp, 
        straight_through=straight_through,
        disallow=disallow
        ))
      tf.logging.info("***** apply sample_from_softmax *****")
    elif self._config.fake_data_sample == 'sample_from_top_k':
      sampled_tokens = fn(gumbel_softmax_sampling.sample_from_top_k(
        mlm_logits, self._config.logits_temp, 
        self._config.gumbel_temp, 
        disallow=disallow, 
        straight_through=straight_through,
        k=self._config.topk))
      tf.logging.info("***** apply sample_from_top_k *****")
    elif self._config.fake_data_sample == 'sample_from_top_p':
      sampled_tokens = fn(gumbel_softmax_sampling.sample_from_top_p(
        mlm_logits, self._config.logits_temp, 
        self._config.gumbel_temp, 
        disallow=disallow, 
        straight_through=straight_through,
        p=self._config.topp))
      tf.logging.info("***** apply sample_from_top_p *****")
    else:
      sampled_tokens = fn(gumbel_softmax_sampling.sample_from_softmax(
        mlm_logits, 
        self._config.logits_temp, 
        self._config.gumbel_temp, 
        straight_through=straight_through,
        disallow=disallow))
      tf.logging.info("***** apply sample_from_softmax *****")

    tf.logging.info("** sampled_tokens **")
    tf.logging.info(sampled_tokens)
    # sampled_tokens: [batch_size, n_pos, n_vocab]
    # mlm_logits: [batch_size, n_pos, n_vocab]
    sampled_tokens_fp32 = tf.cast(sampled_tokens, dtype=tf.float32)
    onehot_input_ids = tf.one_hot(inputs.input_ids, 
                  depth=self._bert_config.vocab_size)
    updated_input_ids, masked = pretrain_helpers.scatter_update(
        tf.cast(onehot_input_ids, dtype=tf.float32), 
        sampled_tokens_fp32, 
        inputs.masked_lm_positions)
    updated_input_ids_ = tf.argmax(updated_input_ids, axis=-1)
    updated_input_ids_ = tf.cast(updated_input_ids_, dtype=tf.int32)
    if self._config.electric_objective:
      labels = masked
    else:
      labels = masked * (1 - tf.cast(
          tf.equal(updated_input_ids_, inputs.input_ids), tf.int32))
    updated_inputs = pretrain_data.get_updated_inputs(
        inputs, input_ids=updated_input_ids)
    FakedData = collections.namedtuple("FakedData", [
        "inputs", "is_fake_tokens", "sampled_tokens"])
    return FakedData(inputs=updated_inputs, is_fake_tokens=labels,
                     sampled_tokens=sampled_tokens)

  def _get_cloze_outputs(self, inputs, model, scope=''):
    weights = tf.cast(pretrain_helpers.get_candidates_mask(
        self._config, inputs), tf.float32)
    
    with tf.variable_scope(scope if scope else 'cloze_predictions'):
      logits = get_token_logits(model.get_sequence_output(),
                                model.get_embedding_table(), self._bert_config)
      return get_softmax_output(logits, inputs.input_ids, weights,
                                self._bert_config.vocab_size)

def get_token_logits(input_reprs, embedding_table, bert_config):
  with tf.variable_scope("transform"): 
    hidden = tf.layers.dense(
        input_reprs,
        units=modeling.get_shape_list(embedding_table)[-1],
        activation=modeling.get_activation(bert_config.hidden_act),
        kernel_initializer=modeling.create_initializer(
            bert_config.initializer_range))
    hidden = modeling.layer_norm(hidden)
  output_bias = tf.get_variable(
      "output_bias",
      shape=[bert_config.vocab_size],
      initializer=tf.zeros_initializer())
  logits = tf.matmul(hidden, embedding_table, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)
  return logits


def get_softmax_output(logits, targets, weights, vocab_size):
  oh_labels = tf.one_hot(targets, depth=vocab_size, dtype=tf.float32)
  preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
  probs = tf.nn.softmax(logits)

  log_probs = tf.nn.log_softmax(logits)
  label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)

  numerator = tf.reduce_sum(weights * label_log_probs)
  denominator = tf.reduce_sum(weights) + 1e-6
  
  loss = tf.reduce_sum(numerator) / (tf.reduce_sum(denominator) + 1e-6)
  SoftmaxOutput = collections.namedtuple(
      "SoftmaxOutput", ["logits", "probs", "loss", "per_example_loss", "preds",
                        "weights", "targets"])
  return SoftmaxOutput(
      logits=logits, probs=probs, per_example_loss=label_log_probs,
      loss=loss, preds=preds, weights=weights,
      targets=targets
      )

class TwoTowerClozeTransformer(object):
  """Build a two-tower Transformer used as Electric's generator."""

  def __init__(self, config, bert_config, inputs,
               is_training, embedding_size):
    ltr = build_transformer(
        config, inputs, is_training, bert_config,
        untied_embeddings=config.untied_generator_embeddings,
        embedding_size=(None if config.untied_generator_embeddings
                        else embedding_size),
        scope="generator_ltr", ltr=True)
    rtl = build_transformer(
        config, inputs, is_training, bert_config,
        untied_embeddings=config.untied_generator_embeddings,
        embedding_size=(None if config.untied_generator_embeddings
                        else embedding_size),
        scope="generator_rtl", rtl=True)
    ltr_reprs = ltr.get_sequence_output()
    rtl_reprs = rtl.get_sequence_output()
    self._sequence_output = tf.concat([roll(ltr_reprs, -1),
                                       roll(rtl_reprs, 1)], -1)
    self._embedding_table = ltr.embedding_table

  def get_sequence_output(self):
    return self._sequence_output

  def get_embedding_table(self):
    return self._embedding_table


def build_transformer(config,
                      inputs, is_training,
                      bert_config, reuse=False, **kwargs):
  """Build a transformer encoder network."""
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
    return modeling.BertModel(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=inputs.input_ids,
        input_mask=inputs.input_mask,
        token_type_ids=inputs.segment_ids,
        use_one_hot_embeddings=config.use_tpu,
        **kwargs)

def build_conv_transformer(config,
                      inputs, is_training,
                      bert_config, reuse=False, **kwargs):
  """Build a transformer encoder network."""
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
    return modeling_convbert.BertModel(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=inputs.input_ids,
        input_mask=inputs.input_mask,
        token_type_ids=inputs.segment_ids,
        use_one_hot_embeddings=config.use_tpu,
        **kwargs)


def build_tta_transformer(config,
                      inputs, is_training,
                      bert_config, reuse=False, **kwargs):
  """Build a transformer encoder network."""
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
    return modeling_tta.BertModel(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=inputs.input_ids,
        input_mask=inputs.input_mask,
        token_type_ids=inputs.segment_ids,
        use_one_hot_embeddings=config.use_tpu,
        **kwargs)


def roll(arr, direction):
  """Shifts embeddings in a [batch, seq_len, dim] tensor to the right/left."""
  return tf.concat([arr[:, direction:, :], arr[:, :direction, :]], axis=1)


def get_generator_config(config,
                         bert_config):
  """Get model config for the generator network."""
  gen_config = modeling.BertConfig.from_dict(bert_config.to_dict())
  gen_config.hidden_size = int(round(
      bert_config.hidden_size * config.generator_hidden_size))
  gen_config.num_hidden_layers = int(round(
      bert_config.num_hidden_layers * config.generator_layers))
  gen_config.intermediate_size = 4 * gen_config.hidden_size
  gen_config.num_attention_heads = max(1, int(round(gen_config.hidden_size // 64)))
  return gen_config


def model_fn_builder(config):
  """Build the model for training."""

  def model_fn(features, labels, mode, params):
    """Build the model for training."""
    model = PretrainingModel(config, features,
                             mode == tf.estimator.ModeKeys.TRAIN)
    print("Model is built!")
    tvars = tf.trainable_variables()
    model.gen_params = list(set(model.gen_params))
    model.disc_params = list(set(model.disc_params))

    for tvar in tvars:
      print(tvar, "========tvar========")
    for tvar in model.gen_params:
      print(tvar, "========gen_params========")
    for tvar in model.disc_params:
      print(tvar, "========disc_params========")

    if mode == tf.estimator.ModeKeys.TRAIN:
      if config.stage == 'one_stage':
        train_op, output_learning_rate = optimization.create_optimizer(
            model.total_loss, config.learning_rate, config.num_train_steps,
            weight_decay_rate=config.weight_decay_rate,
            use_tpu=config.use_tpu,
            warmup_steps=config.num_warmup_steps,
            lr_decay_power=config.lr_decay_power
        )
        model.monitor_dict["learning_rate"] = output_learning_rate
      elif config.stage == 'two_stage':
        prev_op = tf.no_op()
        all_params = [model.gen_params, model.disc_params]
        all_loss = [model.gen_loss, model.disc_loss]
        all_global_step_name = ['gen_step', 'disc_step']
        all_lr = [config.gen_learning_rate, 
              config.disc_learning_rate]
        global_step = tf.train.get_or_create_global_step()
        for params, loss, step_name, lr in zip(all_params, all_loss, 
                                          all_global_step_name,
                                          all_lr):
          with tf.control_dependencies([prev_op]):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
              prev_op, pre_learning_rate = optimization.create_optimizer_v1(
                loss, lr, config.num_train_steps,
                weight_decay_rate=config.weight_decay_rate,
                use_tpu=config.use_tpu,
                warmup_steps=config.num_warmup_steps,
                lr_decay_power=config.lr_decay_power,
                tvars=params,
                global_step_name=step_name
              )
              model.monitor_dict[step_name+"_learning_rate"] = pre_learning_rate
        
        with tf.control_dependencies([prev_op]):
          train_op = global_step.assign_add(1)

      elif config.stage == 'one_stage_merged':
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        global_step = tf.train.get_or_create_global_step()
        gen_op, gen_pre_learning_rate = optimization.create_optimizer_v1(
                model.gen_loss, config.gen_learning_rate, 
                config.num_train_steps,
                weight_decay_rate=config.weight_decay_rate,
                use_tpu=config.use_tpu,
                warmup_steps=config.num_warmup_steps,
                lr_decay_power=config.lr_decay_power,
                tvars=model.gen_params,
                global_step_name="gen_step"
              )
        disc_op, disc_pre_learning_rate = optimization.create_optimizer_v1(
                model.disc_loss, config.disc_learning_rate, 
                config.num_train_steps,
                weight_decay_rate=config.weight_decay_rate,
                use_tpu=config.use_tpu,
                warmup_steps=config.num_warmup_steps,
                lr_decay_power=config.lr_decay_power,
                tvars=model.disc_params,
                global_step_name="disc_step"
              )

        all_op = tf.group([update_ops, gen_op, disc_op])
        with tf.control_dependencies([all_op]):
          train_op = global_step.assign_add(1)

      if config.monitoring:
        if model.monitor_dict:
          host_call = log_utils.construct_scalar_host_call_v1(
                                    monitor_dict=model.monitor_dict,
                                    model_dir=config.model_dir,
                                    prefix="train/")
        else:
          host_call = None
        print("==host_call==", host_call)

      var_checkpoint_dict_list = []
      if config.use_pretrained_generator:
        generator_dict = {
              "tvars":model.gen_params,
              "init_checkpoint":config.generator_init_checkpoint,
              "exclude_scope":model.generator_exclude_scope,
              "restore_var_name":[]
          }
        var_checkpoint_dict_list.append(generator_dict)
      if config.use_pretrained_discriminator:
        discriminator_dict = {
              "tvars":model.disc_params,
              "init_checkpoint":config.discriminator_init_checkpoint,
              "exclude_scope":model.discriminator_exclude_scope,
              "restore_var_name":[]
          }
        var_checkpoint_dict_list.append(discriminator_dict)
      if var_checkpoint_dict_list:
        for item in var_checkpoint_dict_list:
          for key in item:
            print(key, item[key], '===========')
        scaffold_fn = model_io_fn.load_multi_pretrained(
                        var_checkpoint_dict_list,
                        use_tpu=True)
      else:
        scaffold_fn = None

      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=model.disc_loss,
          train_op=train_op,
          # training_hooks=[training_utils.ETAHook(
          #     {} if config.use_tpu else dict(loss=model.total_loss),
          #     config.num_train_steps, config.iterations_per_loop,
          #     config.use_tpu)],
          host_call=host_call if config.monitoring else None,
          scaffold_fn=scaffold_fn
      )
    elif mode == tf.estimator.ModeKeys.EVAL:
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=model.total_loss,
          eval_metrics=model.eval_metrics,
          # evaluation_hooks=[training_utils.ETAHook(
          #     {} if config.use_tpu else dict(loss=model.total_loss),
          #     config.num_eval_steps, config.iterations_per_loop,
          #     config.use_tpu, is_training=False)]
          )
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported")
    return output_spec

  return model_fn


def train_or_eval(config):
  """Run pre-training or evaluate the pre-trained model."""
  if config.do_train == config.do_eval:
    raise ValueError("Exactly one of `do_train` or `do_eval` must be True.")
  if config.debug and config.do_train:
    utils.rmkdir(config.model_dir)
  # utils.heading("Config:")
  # utils.log_config(config)
  for key, value in sorted(config.__dict__.items()):
    print(key, value, "=======")

  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  tpu_cluster_resolver = None
  if config.use_tpu and config.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        config.tpu_name, zone=config.tpu_zone, project=config.gcp_project)
  tpu_config = tf.estimator.tpu.TPUConfig(
      iterations_per_loop=config.iterations_per_loop,
      num_shards=config.num_tpu_cores,
      tpu_job_name=config.tpu_job_name,
      per_host_input_for_training=is_per_host)
  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=config.model_dir,
      save_checkpoints_steps=config.save_checkpoints_steps,
      keep_checkpoint_max=config.keep_checkpoint_max,
      tpu_config=tpu_config)
  model_fn = model_fn_builder(config=config)
  estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=config.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=config.train_batch_size,
      eval_batch_size=config.eval_batch_size)

  if config.do_train:
    # utils.heading("Running training")
    tf.logging.info("** Running training **")

    if config.mask_strategy == 'electra':
      estimator.train(input_fn=pretrain_data.get_input_fn(config, True),
                      max_steps=config.num_train_steps)
    elif config.mask_strategy == 'span_mask':
      estimator.train(input_fn=span_mask_utils.get_input_fn(config, True),
                      max_steps=config.num_train_steps)
  if config.do_eval:
    # utils.heading("Running evaluation")
    tf.logging.info("Running evaluation")
    if config.mask_strategy == 'electra':
      result = estimator.evaluate(
          input_fn=pretrain_data.get_input_fn(config, False),
          steps=config.num_eval_steps)
    elif config.mask_strategy == 'span_mask':
      result = estimator.evaluate(
          input_fn=span_mask_utils.get_input_fn(config, False),
          steps=config.num_eval_steps)
    for key in sorted(result.keys()):
      print("  {:} = {:}".format(key, str(result[key])))
    return result


def train_one_step(config):
  """Builds an ELECTRA model an trains it for one step; useful for debugging."""
  if config.mask_strategy == 'electra':
    train_input_fn = pretrain_data.get_input_fn(config, True)
  elif config.mask_strategy == 'span_mask':
    train_input_fn = span_mask_utils.get_input_fn(config, True)
  features = tf.data.make_one_shot_iterator(train_input_fn(dict(
      batch_size=config.train_batch_size))).get_next()
  model = PretrainingModel(config, features, True)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(model.total_loss))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-dir", required=True,
                      help="Location of data files (model weights, etc).")
  parser.add_argument("--data-file-list", required=True,
                      help="Location of data files (model weights, etc).")
  parser.add_argument("--model-name", required=True,
                      help="The name of the model being fine-tuned.")
  parser.add_argument("--generator-ckpt", required=False,
                      help="The name of the model being fine-tuned.")
  parser.add_argument("--discriminator-ckpt", required=False,
                      help="The name of the model being fine-tuned.")
  parser.add_argument("--hparams", default="{}",
                      help="JSON dict of model hyperparameters.")
  parser.add_argument("--vocab_file", default="{}",
                      help="JSON dict of model hyperparameters.")
  args = parser.parse_args()
  if args.hparams.endswith(".json"):
    hparams = utils.load_json(args.hparams)
  else:
    hparams = json.loads(args.hparams)
  hparams['vocab_file'] = FLAGS.vocab_file
  # tf.logging.set_verbosity(tf.logging.ERROR)
  tf.logging.set_verbosity(tf.logging.INFO)
  train_or_eval(configure_pretraining.PretrainingConfig(
      args.model_name, args.data_dir, args.data_file_list, 
      args.generator_ckpt, args.discriminator_ckpt,
      **hparams))


if __name__ == "__main__":
  main()