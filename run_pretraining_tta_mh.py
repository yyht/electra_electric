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
from model import modeling_tta_electra
from model import optimization
from pretrain import pretrain_data
from pretrain import pretrain_helpers, mh_sampling
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
        # config.untied_generator_embeddings = True
        print("==apply pretrained generator==")
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
        # config.untied_generator_embeddings = True
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
        # config.untied_generator_embeddings = True
        print("==apply pretrained generator==")
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
        tf.logging.info("** apply two-tower generator **")
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
        self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'cloze_predictions')
        if not config.untied_generator_embeddings:
          tf.logging.info("** add shared embeddings **")
          self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.discriminator_scope+"/embeddings")
        else:
          tf.logging.info("** without shared embeddings **")
        tf.logging.info(mlm_output)
      elif config.tta_generator:
        """
        tta with MLM-objective
        """
        tf.logging.info("** apply tta generator **")
        generator = build_tta_transformer(
            config, masked_inputs, is_training, self._generator_config,
            embedding_size=self.generator_embedding_size,
            untied_embeddings=self.untied_generator_embeddings,
            scope=self.generator_scope)
        cloze_output = self._get_cloze_outputs(unmasked_inputs, generator, self.generator_cls_scope)
        mlm_output = self._get_masked_lm_output(masked_inputs, generator, self.generator_cls_scope)

        self.gen_params = []
        self.gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_scope)
        self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_cls_scope)
        if not config.untied_generator_embeddings:
          tf.logging.info("** add shared embeddings **")
          self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.discriminator_scope+"/embeddings")
        else:
          tf.logging.info("** without shared embeddings **")
        tf.logging.info(mlm_output)
      else:
        # small masked language model generator
        tf.logging.info("** apply mlm generator **")
        generator = build_transformer(
            config, masked_inputs, is_training, self._generator_config,
            embedding_size=self.generator_embedding_size,
            untied_embeddings=self.untied_generator_embeddings,
            scope=self.generator_scope,
            reuse=tf.AUTO_REUSE)
        mlm_output = self._get_masked_lm_output(masked_inputs, generator, self.generator_cls_scope)

        self.gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_scope)
        self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_cls_scope)
        if not config.untied_generator_embeddings:
          tf.logging.info("** add shared embeddings **")
          self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.discriminator_scope+"/embeddings")
        else:
          tf.logging.info("** without shared embeddings **")
        tf.logging.info(mlm_output)
    else:
      # full-sized masked language model generator if using BERT objective or if
      # the generator and discriminator have tied weights
      tf.logging.info("** apply mlm generator shared with discriminator **")
      generator = build_transformer(
          config, masked_inputs, is_training, self._bert_config,
          embedding_size=self.discriminator_embedding_size,
          untied_embeddings=self.untied_generator_embeddings,
          scope=self.generator_scope)
      tf.logging.info("** share all parameters **")
      mlm_output = self._get_masked_lm_output(masked_inputs, generator, self.generator_cls_scope)
      tf.logging.info(mlm_output)

      self.gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_scope)
      self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_cls_scope)
    
    (greedy_inputs, 
    greedy_logprob) = self._get_greedy_data(masked_inputs, mlm_output.logits)
    
    tf.logging.info("** get tta greedy decoding **")
    tf.logging.info(greedy_inputs)
    tf.logging.info(greedy_logprob)

    (sampled_inputs, 
    sampled_logprob) = self._get_sampled_data(masked_inputs, mlm_output.logits,
                        disallow=config.disallow_correct)

    tf.logging.info("** get tta sampled decoding **")
    tf.logging.info(sampled_inputs)
    tf.logging.info(sampled_logprob)

    greedy_energy = self._get_generator_energy(
                          config, 
                          greedy_inputs, 
                          is_training,
                          self.generator_cls_scope)

    sampled_energy = self._get_generator_energy(
                          config, 
                          sampled_inputs, 
                          is_training,
                          self.generator_cls_scope)

    gen_true_energy = self._get_generator_energy(
                          config, 
                          unmasked_inputs, 
                          is_training,
                          self.generator_cls_scope)

    tf.logging.info("** get tta generator energy **")
    tf.logging.info(greedy_energy)
    tf.logging.info(sampled_energy)
    tf.logging.info(gen_true_energy)

    # MH-sampling
    # greedy-to-MH transition
    # exp(-energy)*p_transition
    new_data_logprob = -sampled_energy - sampled_logprob
    greedy_data_logprob = -greedy_energy - greedy_logprob
    transition_energy = new_data_logprob - greedy_data_logprob
    transition_logprob = tf.minimum(0., transition_energy)
    u = tf.random.uniform(modeling_tta_electra.get_shape_list(new_data_logprob), minval=1e-10, maxval=1)
    log_u = tf.log(u)

    accept_mask = tf.less(log_u, transition_logprob)
    accept_mask = tf.cast(accept_mask, tf.float32)
  
    accept_weights = tf.expand_dims(tf.cast(accept_mask, greedy_inputs.input_ids.dtype), axis=-1)
    
    tf.logging.info("** get mh accept_weights **")
    tf.logging.info(accept_weights)
    tf.logging.info(sampled_inputs)
    tf.logging.info(greedy_inputs)

    fake_input_ids = (accept_weights) * sampled_inputs.input_ids + (1-accept_weights)*greedy_inputs.input_ids

    gen_fake_energy = accept_mask * sampled_energy + (1.0-accept_mask)*greedy_energy
    
    fake_data_inputs = pretrain_data.get_updated_inputs(
        unmasked_inputs,
        input_ids=tf.stop_gradient(fake_input_ids),
      )

    self.mlm_output = mlm_output
    if config.two_tower_generator or config.tta_generator:
      self.total_loss = config.gen_weight * cloze_output.loss
      tf.logging.info("** apply cloze loss **")
    else:
      self.total_loss = config.gen_weight * mlm_output.loss
      tf.logging.info("** apply mlm loss **")

    if config.two_tower_generator or config.tta_generator:
      self.gen_loss = cloze_output.loss
    else:
      self.gen_loss = mlm_output.loss
    # Discriminator
      
    real_disc_energy = self._get_discriminator_energy(
                        config, 
                        unmasked_inputs, 
                        is_training,
                        self.discriminator_cls_scope)

    fake_disc_energy = self._get_discriminator_energy(
                        config, 
                        fake_data_inputs, 
                        is_training,
                        self.discriminator_cls_scope)

    self.disc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.discriminator_scope)
    self.disc_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.discriminator_cls_scope)
    
    nce_disc_output = self._get_nce_disc_output( 
                              gen_true_energy,
                              gen_fake_energy,
                              real_disc_energy,
                              fake_disc_energy,
                              scope="nce_logz")

    self.disc_loss = nce_disc_output.loss
    self.total_loss += config.disc_weight * nce_disc_output.loss
    tf.logging.info("** apply electra **")

    # Evaluation
    eval_fn_inputs = {
        "input_ids": masked_inputs.input_ids,
        "mlm_loss": mlm_output.per_example_loss,
        "masked_lm_ids": masked_inputs.masked_lm_ids,
        "masked_lm_weights": masked_inputs.masked_lm_weights,
        "input_mask": masked_inputs.input_mask,
        "masked_lm_preds":mlm_output.preds
    }
    
    eval_fn_inputs.update({
        "disc_loss": nce_disc_output.per_example_loss,
        "disc_labels": nce_disc_output.labels,
        "disc_probs": nce_disc_output.probs,
        "disc_preds": nce_disc_output.preds,
        "disc_real_loss":nce_disc_output.real_loss,
        "disc_fake_loss":nce_disc_output.fake_loss,
        "disc_real_labels":nce_disc_output.real_labels,
        "disc_real_preds":nce_disc_output.real_preds,
        "disc_fake_labels":nce_disc_output.fake_labels,
        "disc_fake_preds":nce_disc_output.fake_preds,
        'disc_real_energy':nce_disc_output.d_real_energy,
        'disc_fake_energy':nce_disc_output.d_fake_energy,
        "disc_noise_real_logprob":nce_disc_output.d_noise_real_logprob,
        "disc_noise_fake_logprob":nce_disc_output.d_noise_fake_logprob,
        "masked_mask": masked_inputs.masked_lm_weights,
        "mh_mask": accept_mask,
        "transition_logprob": transition_logprob,
        "sampled_energy": sampled_energy,
        "greedy_energy": greedy_energy,
        "sampled_logprob": sampled_logprob,
        "greedy_logprob": greedy_logprob,
        "transition_energy": transition_energy
      })
    
    eval_fn_keys = eval_fn_inputs.keys()
    eval_fn_values = [eval_fn_inputs[k] for k in eval_fn_keys]
    print(eval_fn_keys, "===eval_fn_keys===")
    print(eval_fn_values, "===eval_fn_values===")

    def electra_electric_monitor_fn(eval_fn_inputs, keys):
      d = {}
      for key in eval_fn_inputs:
        if key in keys:
          d[key] = eval_fn_inputs[key]
      monitor_dict = dict()
      masked_lm_ids = tf.reshape(d["masked_lm_ids"], [-1])
      masked_lm_preds = tf.reshape(d["masked_lm_preds"], [-1])
      masked_lm_weights = tf.reshape(d["masked_lm_weights"], [-1])

      print(masked_lm_preds, "===masked_lm_preds===")
      print(masked_lm_ids, "===masked_lm_ids===")
      print(masked_lm_weights, "===masked_lm_weights===")

      mlm_acc = tf.cast(tf.equal(masked_lm_preds, masked_lm_ids), dtype=tf.float32)
      mlm_acc = tf.reduce_sum(mlm_acc*tf.cast(masked_lm_weights, dtype=tf.float32))
      mlm_acc /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      mlm_loss = tf.reshape(d["mlm_loss"], [-1])
      mlm_loss = tf.reduce_sum(mlm_loss*tf.cast(masked_lm_weights, dtype=tf.float32))
      mlm_loss /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      monitor_dict['gen_mlm_loss'] = mlm_loss
      monitor_dict['gen_mlm_acc'] = mlm_acc
      sent_nce_pred_acc = tf.cast(tf.equal(d["disc_preds"], d['disc_labels']),
                                dtype=tf.float32)
      sent_nce_pred_acc = tf.reduce_mean(sent_nce_pred_acc)

      monitor_dict['disc_sent_pred_acc'] = sent_nce_pred_acc
      monitor_dict['disc_real_loss'] = tf.reduce_mean(d['disc_real_loss'])
      monitor_dict['disc_fake_loss'] = tf.reduce_mean(d['disc_fake_loss'])
      monitor_dict['disc_loss'] = tf.reduce_mean(d['disc_loss'])

      sent_nce_real_pred_acc = tf.cast(tf.equal(d["disc_real_preds"], d['disc_real_labels']),
                                dtype=tf.float32)
      sent_nce_real_pred_acc = tf.reduce_mean(sent_nce_real_pred_acc)
      monitor_dict['disc_sent_real_pred_acc'] = sent_nce_real_pred_acc

      sent_nce_fake_pred_acc = tf.cast(tf.equal(d["disc_fake_preds"], d['disc_fake_labels']),
                                dtype=tf.float32)
      sent_nce_fake_pred_acc = tf.reduce_mean(sent_nce_fake_pred_acc)
      monitor_dict['disc_sent_fake_pred_acc'] = sent_nce_fake_pred_acc
      monitor_dict['disc_real_energy'] = d["disc_real_energy"]
      monitor_dict['disc_fake_energy'] = d["disc_fake_energy"]
      monitor_dict['gen_real_logprob'] = d["disc_noise_real_logprob"]
      monitor_dict['gen_fake_logprob'] = d["disc_noise_fake_logprob"]

      monitor_dict['mh_accept_rate'] = tf.reduce_mean(d['mh_mask'])
      monitor_dict['gen_sampled_energy'] = tf.reduce_mean(d['sampled_energy'])
      monitor_dict['gen_greedy_energy'] = tf.reduce_mean(d['greedy_energy'])
      monitor_dict['gen_sampled_logprob'] = tf.reduce_mean(d['sampled_logprob'])
      monitor_dict['gen_greedy_logprob'] = tf.reduce_mean(d['greedy_logprob'])
      monitor_dict['transition_energy'] = tf.reduce_mean(d['transition_energy'])
      return monitor_dict

    self.monitor_dict = electra_electric_monitor_fn(eval_fn_inputs, eval_fn_keys)
    
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
            modeling_tta_electra.get_shape_list(inputs.masked_lm_ids) +
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
 
  def _get_generator_energy(self, config, inputs, is_training,
                          lm_scope):
    generator = build_tta_transformer(
              config, inputs, is_training, self._generator_config,
              embedding_size=self.generator_embedding_size,
              untied_embeddings=self.untied_generator_embeddings,
              scope=self.generator_scope)

    with tf.variable_scope(lm_scope if lm_scope else 'generator_predictions',
                reuse=tf.AUTO_REUSE):
      relevant_reprs = generator.get_sequence_output()
      # logtis: [batch_size, seq-length, vocab]
      sequence_logits = get_token_logits(
          relevant_reprs, generator.get_embedding_table(), self._bert_config)

    # logtis: [batch_size, seq-length]
    log_energy = tf.reduce_sum(
            sequence_logits * tf.one_hot(
                inputs.input_ids, 
                depth=self._bert_config.vocab_size,
                dtype=tf.float32), -1)

    tf.logging.info("** generator_energy **")
    tf.logging.info(log_energy)

    # weights = tf.cast(inputs.input_mask, tf.float32)
    
    weights = tf.cast(pretrain_helpers.get_candidates_mask(
        self._config, inputs), tf.float32)

    final_energy = -tf.reduce_sum(log_energy*weights, axis=-1)
    
    tf.logging.info(final_energy)

    return final_energy

  def _get_discriminator_energy(self, config, inputs, is_training,
                          lm_scope):
    discriminator = build_tta_transformer(
        config, inputs, is_training, self._bert_config, 
        embedding_size=self.discriminator_embedding_size,
        untied_embeddings=self.untied_discriminator_embeddings,
        scope=self.discriminator_scope)

    with tf.variable_scope(lm_scope if lm_scope else 'discriminator_predictions',
                reuse=tf.AUTO_REUSE):
      relevant_reprs = discriminator.get_sequence_output()
      # logtis: [batch_size, num_masked, vocab]
      sequence_logits = get_token_logits(
          relevant_reprs, discriminator.get_embedding_table(), 
          self._bert_config)

    # logtis: [batch_size, seq-length]
    log_energy = tf.reduce_sum(
            sequence_logits * tf.one_hot(
                inputs.input_ids, 
                depth=self._bert_config.vocab_size,
                dtype=tf.float32), -1)

    tf.logging.info("** discriminator_energy **")
    tf.logging.info(log_energy)

    # weights = tf.cast(inputs.input_mask, tf.float32)
    weights = tf.cast(pretrain_helpers.get_candidates_mask(
        self._config, inputs), tf.float32)

    final_energy = -tf.reduce_sum(log_energy*weights, axis=-1)

    tf.logging.info(final_energy)

    return final_energy

  def _get_discriminator_output(
      self, inputs, discriminator, labels, cloze_output=None):
    """Discriminator binary classifier."""
    with tf.variable_scope("discriminator_predictions"):
      hidden = tf.layers.dense(
          discriminator.get_sequence_output(),
          units=self._bert_config.hidden_size,
          activation=modeling_tta_electra.get_activation(self._bert_config.hidden_act),
          kernel_initializer=modeling_tta_electra.create_initializer(
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

  def _get_nce_disc_output(self, 
                          noise_real_logprobs,
                          noise_fake_logprobs,
                          discriminator_real_energy,
                          discriminator_fake_energy,
                          scope):

    d_out_real = discriminator_real_energy + tf.stop_gradient(-noise_real_logprobs)
    d_out_fake = discriminator_fake_energy + tf.stop_gradient(-noise_fake_logprobs)

    tf.logging.info("** d_out_real **")
    tf.logging.info(d_out_real)

    tf.logging.info("** d_out_fake **")
    tf.logging.info(d_out_fake)

    d_real_energy = tf.reduce_mean(-discriminator_real_energy)
    d_fake_energy = tf.reduce_mean(-discriminator_fake_energy)

    d_noise_real_logprob = tf.reduce_mean(-noise_real_logprobs)
    d_noise_fake_logprob = tf.reduce_mean(-noise_fake_logprobs)

    d_loss_real = (tf.nn.sigmoid_cross_entropy_with_logits(
          logits=d_out_real, labels=tf.zeros_like(d_out_real)
      ))
    d_loss_fake = (tf.nn.sigmoid_cross_entropy_with_logits(
          logits=d_out_fake, labels=tf.ones_like(d_out_fake)
    ))

    per_example_loss = d_loss_real + d_loss_fake
    d_loss = tf.reduce_mean(per_example_loss)

    d_real_probs = tf.nn.sigmoid(d_out_real)
    d_fake_probs = tf.nn.sigmoid(d_out_fake)

    d_real_labels = tf.zeros_like(d_out_real)
    d_fake_labels = tf.ones_like(d_out_fake)

    probs = tf.concat([d_fake_probs, d_real_probs], axis=0)
    preds = tf.cast(tf.greater(probs, 0.5), dtype=tf.int32)

    real_preds = tf.cast(tf.greater(d_real_probs, 0.5), dtype=tf.int32)
    real_labels = tf.cast(d_real_labels, dtype=tf.int32)

    fake_preds = tf.cast(tf.greater(d_fake_probs, 0.5), dtype=tf.int32)
    fake_labels = tf.cast(d_fake_labels, dtype=tf.int32)

    labels = tf.concat([d_fake_labels, d_real_labels], axis=0)
    labels = tf.cast(labels, dtype=tf.int32)

    DiscOutput = collections.namedtuple(
        "DiscOutput", ["loss", "per_example_loss", "probs", "preds",
                       "labels", "real_loss", 'fake_loss',
                       'real_preds', 'real_labels',
                       'fake_preds', 'fake_labels',
                       'd_real_energy', 'd_fake_energy',
                       'd_noise_real_logprob', 'd_noise_fake_logprob'])
    return DiscOutput(
        loss=d_loss, per_example_loss=per_example_loss, probs=probs,
        preds=preds, labels=labels, real_loss=d_loss_real,
        fake_loss=d_loss_fake,
        real_preds=real_preds, real_labels=real_labels,
        fake_preds=fake_preds, fake_labels=fake_labels,
        d_real_energy=d_real_energy, d_fake_energy=d_fake_energy,
        d_noise_real_logprob=d_noise_real_logprob, 
        d_noise_fake_logprob=d_noise_fake_logprob
    )

  def _get_greedy_data(self, inputs, mlm_logits):
    masked_lm_weights = inputs.masked_lm_weights
    inputs = pretrain_helpers.unmask(inputs)
    (sampled_tokens, 
    sampled_logprob) = mh_sampling.greedy_from_softmax(
                          mlm_logits, 
                          self._config.logits_temp, 
                          self._config.gumbel_temp)

    sampled_tokens = tf.stop_gradient(sampled_tokens)

    sampled_tokids = tf.argmax(sampled_tokens, -1, output_type=tf.int32)
    updated_input_ids, masked = pretrain_helpers.scatter_update(
        inputs.input_ids, sampled_tokids, inputs.masked_lm_positions)

    updated_inputs = pretrain_data.get_updated_inputs(
        inputs, input_ids=updated_input_ids)

    return updated_inputs, sampled_logprob

  def _get_sampled_data(self, inputs, mlm_logits, disallow):
    masked_lm_weights = inputs.masked_lm_weights
    inputs = pretrain_helpers.unmask(inputs)
    
    if self._config.fake_data_sample == 'sample_from_softmax':
      fn = mh_sampling.sample_from_softmax
      tf.logging.info("***** apply sample_from_softmax *****")
    elif self._config.fake_data_sample == 'sample_from_top_k':
      fn = mh_sampling.sample_from_top_k
      tf.logging.info("***** apply sample_from_top_k *****")
    elif self._config.fake_data_sample == 'sample_from_top_p':
      fn =  mh_sampling.sample_from_top_p
      tf.logging.info("***** apply sample_from_top_p *****")
    else:
      fn = mh_sampling.sample_from_softmax
      tf.logging.info("***** apply sample_from_softmax *****")

    (sampled_tokens, 
    sampled_logprob) = fn(
      mlm_logits, self._config.logits_temp, 
        self._config.gumbel_temp, disallow=disallow)

    sampled_tokens = tf.stop_gradient(sampled_tokens)

    tf.logging.info("** sampled from mlm logits ***")
    tf.logging.info(sampled_tokens)
    tf.logging.info(sampled_logprob)

    sampled_tokids = tf.argmax(sampled_tokens, -1, output_type=tf.int32)
    updated_input_ids, masked = pretrain_helpers.scatter_update(
        inputs.input_ids, sampled_tokids, inputs.masked_lm_positions)

    updated_inputs = pretrain_data.get_updated_inputs(
        inputs, input_ids=updated_input_ids)
    return updated_inputs, sampled_logprob

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
        units=modeling_tta_electra.get_shape_list(embedding_table)[-1],
        activation=modeling_tta_electra.get_activation(bert_config.hidden_act),
        kernel_initializer=modeling_tta_electra.create_initializer(
            bert_config.initializer_range))
    hidden = modeling_tta_electra.layer_norm(hidden)
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

def build_tta_transformer(config,
                      inputs, is_training,
                      bert_config, reuse=False, **kwargs):
  """Build a transformer encoder network."""
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
    return modeling_tta_electra.BertModel(
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
  gen_config = modeling_tta_electra.BertConfig.from_dict(bert_config.to_dict())
  gen_config.hidden_size = int(round(
      bert_config.hidden_size * config.generator_hidden_size))
  gen_config.num_hidden_layers = int(round(
      bert_config.num_hidden_layers * config.generator_layers))
  gen_config.intermediate_size = 4 * gen_config.hidden_size
  gen_config.num_attention_heads = max(1, gen_config.hidden_size // 64)
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
  args = parser.parse_args()
  if args.hparams.endswith(".json"):
    hparams = utils.load_json(args.hparams)
  else:
    hparams = json.loads(args.hparams)
  # tf.logging.set_verbosity(tf.logging.ERROR)
  tf.logging.set_verbosity(tf.logging.INFO)
  train_or_eval(configure_pretraining.PretrainingConfig(
      args.model_name, args.data_dir, args.data_file_list, 
      args.generator_ckpt, args.discriminator_ckpt,
      **hparams))


if __name__ == "__main__":
  main()