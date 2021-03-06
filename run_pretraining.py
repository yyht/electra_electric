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
    cloze_output = None
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
            self._bert_config.vocab_size,
            self._config.logprob_avg)
        print("==two_tower_generator==")

        self.gen_params = []
        self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator_ltr')
        self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator_rtl')
        self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'cloze_predictions')
        if not config.untied_generator_embeddings:
          print("==add shared embeddings==")
          self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.discriminator_scope+"/embeddings")
      elif config.tta_generator:
        generator = build_tta_transformer(
            config, unmasked_inputs, is_training, self._generator_config,
            embedding_size=self.generator_embedding_size,
            untied_embeddings=self.untied_generator_embeddings,
            scope=self.generator_scope)
        mlm_output = self._get_masked_lm_output(masked_inputs, generator, self.generator_cls_scope)
        print("==tta_generator==")
        self.gen_params = []
        self.gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_scope)
        self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_cls_scope)
        if not config.untied_generator_embeddings:
          print("==add shared embeddings==")
          self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.discriminator_scope+"/embeddings")
      else:
        # small masked language model generator
        generator = build_transformer(
            config, masked_inputs, is_training, self._generator_config,
            embedding_size=self.generator_embedding_size,
            untied_embeddings=self.untied_generator_embeddings,
            scope=self.generator_scope,
            reuse=tf.AUTO_REUSE)
        mlm_output = self._get_masked_lm_output(masked_inputs, generator, self.generator_cls_scope)

        print("==mlm share embeddings==")
        self.gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_scope)
        self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_cls_scope)
        print("==untied_generator_embeddings==", config.untied_generator_embeddings)
        if not config.untied_generator_embeddings:
          print("==add shared embeddings==")
          self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.discriminator_scope+"/embeddings")
        print(mlm_output, "===mlm_output using tied embedding mlm generator===")
    else:
      # full-sized masked language model generator if using BERT objective or if
      # the generator and discriminator have tied weights
      generator = build_transformer(
          config, masked_inputs, is_training, self._bert_config,
          embedding_size=self.discriminator_embedding_size,
          untied_embeddings=self.untied_generator_embeddings,
          scope=self.generator_scope)
      print("==share all params==")
      mlm_output = self._get_masked_lm_output(masked_inputs, generator, self.generator_cls_scope)

      self.gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_scope)
      self.gen_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator_cls_scope)
    
    fake_data = self._get_fake_data(masked_inputs, mlm_output.logits)
    self.mlm_output = mlm_output
    self.total_loss = config.gen_weight * (
        cloze_output.loss if config.two_tower_generator else mlm_output.loss)
    if config.two_tower_generator:
      self.gen_loss = cloze_output.loss
    else:
      self.gen_loss = mlm_output.loss
    # Discriminator
    disc_output = None
    if config.electra_objective or config.electric_objective:
      discriminator = build_transformer(
          config, fake_data.inputs, is_training, self._bert_config,
          reuse=tf.AUTO_REUSE, 
          embedding_size=self.discriminator_embedding_size,
          untied_embeddings=self.untied_discriminator_embeddings,
          scope=self.discriminator_scope)
      disc_output = self._get_discriminator_output(
          fake_data.inputs, discriminator, fake_data.is_fake_tokens,
          cloze_output)
      self.disc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.discriminator_scope)
      self.disc_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator_predictions')
      self.disc_loss = disc_output.loss
      self.total_loss += config.disc_weight * disc_output.loss
    elif config.electra_nce_objective:
      disc_fake = build_transformer(
          config, fake_data.inputs, is_training, self._bert_config,
          reuse=tf.AUTO_REUSE, 
          embedding_size=self.discriminator_embedding_size,
          untied_embeddings=self.untied_discriminator_embeddings,
          scope=self.discriminator_scope)

      input_ids_shape = modeling.get_shape_list(unmasked_inputs.input_ids, expected_rank=[2,3])

      global_step = tf.train.get_or_create_global_step()

      random_prob = tf.random.uniform(shape=[input_ids_shape[0]], minval=0.0, maxval=1.0)
      threhold_prob = tf.train.polynomial_decay(
                    0.85,
                    global_step,
                    config.num_train_steps,
                    end_learning_rate=0.9,
                    power=1,
                    cycle=True) 
      random_prob_mask = tf.greater_equal(random_prob, threhold_prob)
      random_prob_mask = tf.cast(random_prob_mask, unmasked_inputs.input_ids.dtype)
      # [batch_size, 1]
      tf.logging.info("** * unmasked_inputs **")
      tf.logging.info(unmasked_inputs.input_ids)
      tf.logging.info("** * fake_data **")
      tf.logging.info(fake_data.inputs.input_ids)
      random_prob_mask = tf.expand_dims(random_prob_mask, axis=-1)
      real_fake_mixture = (1-random_prob_mask) * unmasked_inputs.input_ids + (random_prob_mask)*fake_data.inputs.input_ids

      unmasked_inputs = pretrain_data.get_updated_inputs(
        unmasked_inputs,
        input_ids=tf.stop_gradient(real_fake_mixture),
      )

      disc_real = build_transformer(
          config, unmasked_inputs, is_training, self._bert_config,
          reuse=tf.AUTO_REUSE, 
          embedding_size=self.discriminator_embedding_size,
          untied_embeddings=self.untied_discriminator_embeddings,
          scope=self.discriminator_scope)
      print(disc_real, "===disc_real using for conditional real data energy function===")

      self.disc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.discriminator_scope)
      if config.nce_mlm:
        disc_mlm_output = self._get_masked_lm_output(fake_data.inputs, disc_fake, self.discriminator_cls_scope)
        print(disc_fake, "===disc_fake using for mlm===")
        self.total_loss += disc_mlm_output.loss
        self.disc_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.discriminator_cls_scope)
      if config.nce_electra:

        disc_output = self._get_discriminator_output(
          fake_data.inputs, disc_fake, fake_data.is_fake_tokens,
          mlm_output)
        self.disc_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator_predictions')
        self.total_loss += config.electra_disc_weight * disc_output.loss
      if config.nce == 'nce':
        disc_real_energy = self._get_nce_disc_energy(unmasked_inputs, 
                                              disc_real)
        print(disc_real_energy, "===disc_real_energy using for conditional real data energy function===")

        disc_fake_energy = self._get_nce_disc_energy(fake_data.inputs, 
                                                disc_fake)
        print(disc_fake_energy, "===disc_fake_energy using for conditional real data energy function===")
        nce_disc_output = self._get_nce_disc_output( 
                                mlm_output.pseudo_logprob,
                                fake_data.pseudo_logprob,
                                disc_real_energy,
                                disc_fake_energy)

        self.disc_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'nce/discriminator_predictions')

      elif config.nce == 'gan':
        disc_real_energy = self._get_gan_output(unmasked_inputs, 
                                              disc_real)
        print(disc_real_energy, "===disc_real_energy using for conditional real data energy function===")

        disc_fake_energy = self._get_gan_output(fake_data.inputs, 
                                                disc_fake)
        print(disc_fake_energy, "===disc_fake_energy using for conditional real data energy function===")
        nce_disc_output = self._get_gan_disc_output( 
                                mlm_output.pseudo_logprob,
                                fake_data.pseudo_logprob,
                                disc_real_energy,
                                disc_fake_energy)
        print("==not using mlm as bias==")
        self.disc_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gan/discriminator_predictions')
      
      self.total_loss += config.nce_disc_weight * nce_disc_output.loss
      
      self.disc_loss = nce_disc_output.loss
      if config.nce_mlm:
        print("==discriminator using mlm loss==")
        self.disc_loss += disc_mlm_output.loss
      if config.nce_electra:
        print("==discriminator using electra loss==")
        self.disc_loss += config.electra_disc_weight * disc_output.loss

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
    elif config.electra_nce_objective:
      eval_fn_inputs.update({
          "disc_loss": nce_disc_output.per_example_loss,
          "disc_labels": nce_disc_output.labels,
          "disc_probs": nce_disc_output.probs,
          "disc_preds": nce_disc_output.preds,
          "sampled_tokids": tf.argmax(fake_data.sampled_tokens, -1,
                                      output_type=tf.int32),
          "disc_real_loss":nce_disc_output.real_loss,
          "disc_fake_loss":nce_disc_output.fake_loss,
          "disc_real_labels":nce_disc_output.real_labels,
          "disc_real_preds":nce_disc_output.real_preds,
          "disc_fake_labels":nce_disc_output.fake_labels,
          "disc_fake_preds":nce_disc_output.fake_preds,
          'disc_real_energy':nce_disc_output.d_real_energy,
          'disc_fake_energy':nce_disc_output.d_fake_energy,
          "disc_noise_real_logprob":nce_disc_output.d_noise_real_logprob,
          "disc_noise_fake_logprob":nce_disc_output.d_noise_fake_logprob

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

      sampled_masked_lm_ids = tf.reshape(d["sampled_masked_lm_ids"], [-1])
      sampled_masked_lm_preds = tf.reshape(d["sampled_masked_lm_preds"], [-1])
      sampled_masked_lm_weights = tf.reshape(d["sampled_masked_lm_weights"], [-1])

      print(sampled_masked_lm_preds, "===sampled_masked_lm_preds===")
      print(sampled_masked_lm_ids, "===sampled_masked_lm_ids===")
      print(sampled_masked_lm_weights, "===sampled_masked_lm_weights===")

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

      sent_nce_pred_acc = tf.cast(tf.equal(d["disc_preds"], d['disc_labels']),
                                dtype=tf.float32)
      sent_nce_pred_acc = tf.reduce_mean(sent_nce_pred_acc)

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
      return monitor_dict

    def electra_nce_monitor_fn(eval_fn_inputs, keys):
      # d = {k: arg for k, arg in zip(eval_fn_keys, args)}
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

      sampled_masked_lm_ids = tf.reshape(d["sampled_masked_lm_ids"], [-1])
      sampled_masked_lm_preds = tf.reshape(d["sampled_masked_lm_preds"], [-1])
      sampled_masked_lm_weights = tf.reshape(d["sampled_masked_lm_weights"], [-1])

      print(sampled_masked_lm_preds, "===sampled_masked_lm_preds===")
      print(sampled_masked_lm_ids, "===sampled_masked_lm_ids===")
      print(sampled_masked_lm_weights, "===sampled_masked_lm_weights===")

      # masked_lm_pred_ids = tf.argmax(masked_lm_preds, axis=-1, 
      #                             output_type=tf.int32)
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
      sampeld_mlm_acc = tf.reduce_sum(sampeld_mlm_acc*tf.cast(masked_lm_weights, dtype=tf.float32))
      sampeld_mlm_acc /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      monitor_dict['generator_sampled_mlm_acc'] = sampeld_mlm_acc

      sent_nce_pred_acc = tf.cast(tf.equal(d["disc_preds"], d['disc_labels']),
                                dtype=tf.float32)
      sent_nce_pred_acc = tf.reduce_mean(sent_nce_pred_acc)

      monitor_dict['discriminator_sent_nce_pred_acc'] = sent_nce_pred_acc
      monitor_dict['discriminator_sent_nce_real_loss'] = tf.reduce_mean(d['disc_real_loss'])
      monitor_dict['discriminator_sent_nce_fake_loss'] = tf.reduce_mean(d['disc_fake_loss'])
      monitor_dict['discriminator_sent_nce_loss'] = tf.reduce_mean(d['disc_loss'])

      sent_nce_real_pred_acc = tf.cast(tf.equal(d["disc_real_preds"], d['disc_real_labels']),
                                dtype=tf.float32)
      sent_nce_real_pred_acc = tf.reduce_mean(sent_nce_real_pred_acc)
      monitor_dict['discriminator_sent_nce_real_pred_acc'] = sent_nce_real_pred_acc

      sent_nce_fake_pred_acc = tf.cast(tf.equal(d["disc_fake_preds"], d['disc_fake_labels']),
                                dtype=tf.float32)
      sent_nce_fake_pred_acc = tf.reduce_mean(sent_nce_fake_pred_acc)
      monitor_dict['discriminator_sent_nce_fake_pred_acc'] = sent_nce_fake_pred_acc
      monitor_dict['discriminator_real_energy'] = d["disc_real_energy"]
      monitor_dict['discriminator_fake_energy'] = d["disc_fake_energy"]
      monitor_dict['generator_noise_real_logprob'] = d["disc_noise_real_logprob"]
      monitor_dict['generator_noise_fake_logprob'] = d["disc_noise_fake_logprob"]
      return monitor_dict

    if config.electra_objective or config.electric_objective:
      self.monitor_dict = electra_electric_monitor_fn(eval_fn_inputs, eval_fn_keys)
      print("==monitor dict construction electra_electric_objective==", self.monitor_dict)
    elif config.electra_nce_objective:
      self.monitor_dict = electra_nce_monitor_fn(eval_fn_inputs, eval_fn_keys)
      print("==monitor dict construction electra_nce_objective==", self.monitor_dict)

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
          self._bert_config.vocab_size,
          self._config.logprob_avg)

  def _get_nce_disc_energy(self, inputs,
                              discriminator):
    if self._config.spectral_regularization:
      print("==spectral_regularization==")
      custom_getter = spectural_utils.spectral_normalization_custom_getter(training=self.is_training)
    else:
      custom_getter = None
      print("==no spectral_regularization==")
    print(discriminator.get_sequence_output(), "==discriminator.get_sequence_output()==")
    with tf.variable_scope("nce/discriminator_predictions", reuse=tf.AUTO_REUSE,
          custom_getter=custom_getter):
      hidden = tf.layers.dense(
          discriminator.get_sequence_output(),
          units=self._bert_config.hidden_size,
          activation=modeling.get_activation(self._bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              self._bert_config.initializer_range))
      weights = tf.cast(inputs.input_mask, tf.float32)
      weights = tf.expand_dims(weights, axis=-1)
      print("==hidden==", hidden, weights)
      print("==hidden sum==", tf.reduce_sum(hidden*weights, axis=1))
      energy = tf.reduce_sum(hidden*weights, axis=1) / (1e-10+tf.reduce_sum(weights, axis=1))
      # enrergy:[batch_size, hidden_size]
      print("==energy output==", energy)
      energy = tf.squeeze(tf.layers.dense(energy, units=1), -1)
      return energy

  def _get_gan_output(self, inputs,
                              discriminator):

    with tf.variable_scope("gan/discriminator_predictions", reuse=tf.AUTO_REUSE):
      print(discriminator.get_sequence_output(), "==discriminator.get_sequence_output()==")
      hidden = tf.layers.dense(
          discriminator.get_sequence_output(),
          units=self._bert_config.hidden_size,
          activation=modeling.get_activation(self._bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              self._bert_config.initializer_range))
      weights = tf.cast(inputs.input_mask, tf.float32)
      my_weights = tf.expand_dims(weights, axis=-1)
      energy = tf.reduce_sum(hidden*my_weights, axis=1) / (1e-10+tf.reduce_sum(weights, axis=-1, keepdims=True))
      # enrergy:[batch_size, hidden_size]
      print("==energy output==", energy)
      energy = tf.squeeze(tf.layers.dense(energy, units=1), -1)
      return energy

  def _get_gan_disc_output(self, 
                          noise_real_logprobs,
                          noise_fake_logprobs,
                          discriminator_real_energy,
                          discriminator_fake_energy):
    d_out_real = tf.identity(discriminator_real_energy)
    d_loss_real = (tf.nn.sigmoid_cross_entropy_with_logits(
          logits=d_out_real, labels=tf.zeros_like(d_out_real)
      ))
    d_out_fake = tf.identity(discriminator_fake_energy)
    d_loss_fake = (tf.nn.sigmoid_cross_entropy_with_logits(
          logits=d_out_fake, labels=tf.ones_like(d_out_fake)
    ))

    print(d_out_real, "==d_out_real==")
    print(d_out_fake, "==d_out_fake==")

    print(d_loss_real, "==d_loss_real==")
    print(d_loss_fake, "==d_loss_fake==")

    d_real_energy = tf.reduce_mean(d_out_real)
    d_fake_energy = tf.reduce_mean(d_out_fake)

    d_noise_real_logprob = tf.reduce_mean(discriminator_real_energy)
    d_noise_fake_logprob = tf.reduce_mean(discriminator_fake_energy)

    per_example_loss = d_loss_real + d_loss_fake
    d_loss = tf.reduce_mean(per_example_loss)

    print(per_example_loss, "==per_example_loss==")
    print(d_loss, "==d_loss==")

    d_real_probs = tf.nn.sigmoid(d_out_real)
    d_fake_probs = tf.nn.sigmoid(d_out_fake)

    print(d_real_probs, "==d_real_probs==")
    print(d_fake_probs, "==d_fake_probs==")

    d_real_labels = tf.zeros_like(d_out_real)
    d_fake_labels = tf.ones_like(d_out_fake)

    probs = tf.concat([d_fake_probs, d_real_probs], axis=0)
    preds = tf.cast(tf.greater(probs, 0.5), dtype=tf.int32)

    print(probs, "==probs==")
    print(preds, "==preds==")

    real_preds = tf.cast(tf.greater(d_real_probs, 0.5), dtype=tf.int32)
    real_labels = tf.cast(d_real_labels, dtype=tf.int32)

    print(real_preds, "==real_preds==")
    print(real_labels, "==real_labels==")

    fake_preds = tf.cast(tf.greater(d_fake_probs, 0.5), dtype=tf.int32)
    fake_labels = tf.cast(d_fake_labels, dtype=tf.int32)

    print(fake_preds, "==fake_preds==")
    print(fake_labels, "==fake_labels==")

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

  def _get_nce_disc_output(self, 
                          noise_real_logprobs,
                          noise_fake_logprobs,
                          discriminator_real_energy,
                          discriminator_fake_energy):

    d_out_real = discriminator_real_energy+tf.stop_gradient(noise_real_logprobs)
    d_out_fake = discriminator_fake_energy+tf.stop_gradient(noise_fake_logprobs)

    print(d_out_real, "==d_out_real==")
    print(d_out_fake, "==d_out_fake==")

    d_real_energy = tf.reduce_mean(discriminator_real_energy)
    d_fake_energy = tf.reduce_mean(discriminator_fake_energy)

    d_noise_real_logprob = tf.reduce_mean(noise_real_logprobs)
    d_noise_fake_logprob = tf.reduce_mean(noise_fake_logprobs)

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
 
  def _get_discriminator_output(
      self, inputs, discriminator, labels, cloze_output=None):
    """Discriminator binary classifier."""
    with tf.variable_scope("discriminator_predictions"):
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
        print("==apply electric_objective==")

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

  def _get_fake_data(self, inputs, mlm_logits):
    """Sample from the generator to create corrupted input."""
    masked_lm_weights = inputs.masked_lm_weights
    inputs = pretrain_helpers.unmask(inputs)
    disallow = tf.one_hot(
        inputs.masked_lm_ids, depth=self._bert_config.vocab_size,
        dtype=tf.float32) if self._config.disallow_correct else None
    if self._config.fake_data_sample == 'sample_from_softmax':
      sampled_tokens = tf.stop_gradient(pretrain_helpers.sample_from_softmax(
        mlm_logits, self._config.logits_temp, 
        self._config.gumbel_temp, disallow=disallow))
      tf.logging.info("***** apply sample_from_softmax *****")
    elif self._config.fake_data_sample == 'sample_from_top_k':
      sampled_tokens = tf.stop_gradient(pretrain_helpers.sample_from_top_k(
        mlm_logits, self._config.logits_temp, 
        self._config.gumbel_temp, disallow=disallow, k=self._config.topk))
      tf.logging.info("***** apply sample_from_top_k *****")
    elif self._config.fake_data_sample == 'sample_from_top_p':
      sampled_tokens = tf.stop_gradient(pretrain_helpers.sample_from_top_p(
        mlm_logits, self._config.logits_temp, 
        self._config.gumbel_temp, disallow=disallow, p=self._config.topp))
      tf.logging.info("***** apply sample_from_top_p *****")
    else:
      sampled_tokens = tf.stop_gradient(pretrain_helpers.sample_from_softmax(
        mlm_logits, self._config.logits_temp, 
        self._config.gumbel_temp, disallow=disallow))
      tf.logging.info("***** apply sample_from_softmax *****")

    # sampled_tokens: [batch_size, n_pos, n_vocab]
    # mlm_logits: [batch_size, n_pos, n_vocab]
    sampled_tokens_fp32 = tf.cast(sampled_tokens, dtype=tf.float32)
    print(sampled_tokens_fp32, "===sampled_tokens_fp32===")
    # [batch_size, n_pos]
    # mlm_logprobs: [batch_size, n_pos. n_vocab]
    mlm_logprobs = tf.nn.log_softmax(mlm_logits, axis=-1)
    print(mlm_logprobs, "===mlm_logprobs===")
    pseudo_logprob = tf.reduce_sum(mlm_logprobs*sampled_tokens_fp32, axis=-1)
    print(pseudo_logprob, "===pseudo_logprob===")
    pseudo_logprob *= tf.cast(masked_lm_weights, dtype=tf.float32)
    print(pseudo_logprob, "===pseudo_logprob===")
    # [batch_size]
    pseudo_logprob = tf.reduce_sum(pseudo_logprob, axis=-1)
    # [batch_size]
    if self._config.logprob_avg:
      pseudo_logprob /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32), axis=-1))
      print("==apply averaging on fake logprob==")
    print("== _get_fake_data pseudo_logprob ==", pseudo_logprob)
    sampled_tokids = tf.argmax(sampled_tokens, -1, output_type=tf.int32)
    updated_input_ids, masked = pretrain_helpers.scatter_update(
        inputs.input_ids, sampled_tokids, inputs.masked_lm_positions)
    if self._config.electric_objective:
      labels = masked
    else:
      labels = masked * (1 - tf.cast(
          tf.equal(updated_input_ids, inputs.input_ids), tf.int32))
    updated_inputs = pretrain_data.get_updated_inputs(
        inputs, input_ids=updated_input_ids)
    FakedData = collections.namedtuple("FakedData", [
        "inputs", "is_fake_tokens", "sampled_tokens", "pseudo_logprob"])
    return FakedData(inputs=updated_inputs, is_fake_tokens=labels,
                     sampled_tokens=sampled_tokens,
                     pseudo_logprob=pseudo_logprob)

  # def _get_cloze_outputs(self, inputs, model):
  #   """Cloze model softmax layer."""
  #   weights = tf.cast(pretrain_helpers.get_candidates_mask(
  #       self._config, inputs), tf.float32)
  #   with tf.variable_scope("cloze_predictions"):
  #     logits = get_token_logits(model.get_sequence_output(),
  #                               model.get_embedding_table(), self._bert_config)
  #     return get_softmax_output(logits, inputs.input_ids, weights,
  #                               self._bert_config.vocab_size)

  def _get_cloze_outputs(self, inputs, model, scope=''):
    weights = tf.cast(pretrain_helpers.get_candidates_mask(
        self._config, inputs), tf.float32)
    # if scope:
    #   pretrained_scope = scope + "/cls/predictions"
    # else:
    #   pretrained_scope = 'cloze_predictions'
    with tf.variable_scope(scope if scope else 'cloze_predictions'):
      logits = get_token_logits(model.get_sequence_output(),
                                model.get_embedding_table(), self._bert_config)
      return get_softmax_output(logits, inputs.input_ids, weights,
                                self._bert_config.vocab_size,
                                self._config.logprob_avg)

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


def get_softmax_output(logits, targets, weights, vocab_size, 
                      logprob_avg=False):
  oh_labels = tf.one_hot(targets, depth=vocab_size, dtype=tf.float32)
  preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
  probs = tf.nn.softmax(logits)
  log_probs = tf.nn.log_softmax(logits)
  # [batch_size, num_masked]
  label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)
  print(label_log_probs, "===label_log_probs===")
  numerator = tf.reduce_sum(weights * label_log_probs, axis=-1)
  print(numerator, "===numerator===")
  # [batch_size, num_masked]
  denominator = tf.reduce_sum(weights, axis=-1)
  pseudo_logprob = -numerator
  if logprob_avg:
    pseudo_logprob /= (denominator + 1e-6)
    print("==apply averaging on mlm==")
  # pseudo_logprob = -numerator
  print("== get_softmax_output ==", pseudo_logprob)
  loss = tf.reduce_sum(numerator) / (tf.reduce_sum(denominator) + 1e-6)
  SoftmaxOutput = collections.namedtuple(
      "SoftmaxOutput", ["logits", "probs", "loss", "per_example_loss", "preds",
                        "weights", "pseudo_logprob",
                        "targets"])
  return SoftmaxOutput(
      logits=logits, probs=probs, per_example_loss=label_log_probs,
      loss=loss, preds=preds, weights=weights,
      pseudo_logprob=pseudo_logprob,
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
  gen_config = modeling.BertConfig.from_dict(bert_config.to_dict())
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
    tf.logging.info("Running training")

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
