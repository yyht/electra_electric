# coding=utf-8

"""Pre-trains an ELECTRA/ConvBERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json

# import tensorflow.compat.v1 as tf
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
import numpy as np
import configure_pretraining_electra as configure_pretraining
from model import modeling_electra
from model import modeling_electra
from model import optimization
from pretrain import pretrain_data
from pretrain import pretrain_helpers
from util import training_utils_electra as training_utils
from util import utils, log_utils


class PretrainingModel(object):
  """Transformer pre-training using the replaced-token-detection task."""

  def __init__(self, config,
               features, is_training):
    # Set up model config
    self._config = config
    self._bert_config = training_utils.get_bert_config(config)
    if config.debug:
      self._bert_config.num_hidden_layers = 3
      self._bert_config.hidden_size = 144
      self._bert_config.intermediate_size = 144 * 4
      self._bert_config.num_attention_heads = 4

    self.monitor_dict = {}

    # Mask the input
    masked_inputs = pretrain_helpers.mask(
        config, pretrain_data.features_to_inputs(features), config.mask_prob)

    # Generator
    embedding_size = (
        self._bert_config.hidden_size if config.embedding_size is None else
        config.embedding_size)
    if config.disc_mlm:
      tf.logging.info("** add disc mlm **")
      config.untied_generator_embeddings = True

    if config.uniform_generator:
      mlm_output = self._get_masked_lm_output(masked_inputs, None)
    elif config.electra_objective and config.untied_generator:
      generator = self._build_transformer(
          masked_inputs, is_training,
          bert_config=get_generator_config(config, self._bert_config),
          embedding_size=(None if config.untied_generator_embeddings
                          else embedding_size),
          untied_embeddings=config.untied_generator_embeddings,
          name="generator")
      mlm_output = self._get_masked_lm_output(masked_inputs, generator)
    else:
      generator = self._build_transformer(
          masked_inputs, is_training, embedding_size=embedding_size)
      mlm_output = self._get_masked_lm_output(masked_inputs, generator)
    fake_data = self._get_fake_data(masked_inputs, mlm_output.logits)
    self.mlm_output = mlm_output
    self.total_loss = config.gen_weight * mlm_output.loss

    # Discriminator
    disc_output = None
    if config.electra_objective:
      discriminator = self._build_transformer(
          fake_data.inputs, is_training, reuse=not config.untied_generator,
          embedding_size=embedding_size)
      disc_output = self._get_discriminator_output(
          fake_data.inputs, discriminator, fake_data.is_fake_tokens)
      
      if config.disc_mlm:
        tf.logging.info("** add disc mlm **")
        tf.logging.info("** fake data **")
        tf.logging.info(fake_data)
        tf.logging.info("** masked data **")
        tf.logging.info(masked_inputs)

        disc_mlm_output = self._get_disc_masked_lm_output(fake_data.inputs,
                                      discriminator)
        self.total_loss += config.disc_mlm_weight * disc_mlm_output.loss

      self.total_loss += config.disc_weight * disc_output.loss

    # Evaluation
    eval_fn_inputs = {
        "input_ids": masked_inputs.input_ids,
        "masked_lm_preds": mlm_output.preds,
        "mlm_loss": mlm_output.per_example_loss,
        "masked_lm_ids": masked_inputs.masked_lm_ids,
        "masked_lm_weights": masked_inputs.masked_lm_weights,
        "input_mask": masked_inputs.input_mask
    }
    if config.electra_objective:
      eval_fn_inputs.update({
          "disc_loss": disc_output.per_example_loss,
          "disc_labels": disc_output.labels,
          "disc_probs": disc_output.probs,
          "disc_preds": disc_output.preds,
          "sampled_tokids": tf.argmax(fake_data.sampled_tokens, -1,
                                      output_type=tf.int32)
      })

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

      sampled_masked_lm_ids = tf.reshape(d["sampled_masked_lm_ids"], [-1])
      sampled_masked_lm_preds = tf.reshape(d["sampled_masked_lm_preds"], [-1])
      sampled_masked_lm_weights = tf.reshape(d["sampled_masked_lm_weights"], [-1])

      mlm_acc = tf.cast(tf.equal(masked_lm_preds, masked_lm_ids), dtype=tf.float32)
      mlm_acc = tf.reduce_sum(mlm_acc*tf.cast(masked_lm_weights, dtype=tf.float32))
      mlm_acc /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      mlm_loss = tf.reshape(d["mlm_loss"], [-1])
      mlm_loss = tf.reduce_sum(mlm_loss*tf.cast(masked_lm_weights, dtype=tf.float32))
      mlm_loss /= (1e-10+tf.reduce_sum(tf.cast(masked_lm_weights, dtype=tf.float32)))

      monitor_dict['mlm_loss'] = mlm_loss
      monitor_dict['mlm_acc'] = mlm_acc

      sampled_lm_ids = tf.reshape(d["sampled_masked_lm_ids"], [-1])
      sampled_lm_pred_ids = tf.reshape(d["sampled_tokids"], [-1])
      sampeld_mlm_acc = tf.cast(tf.equal(sampled_lm_pred_ids, sampled_lm_ids), dtype=tf.float32)
      sampeld_mlm_acc = tf.reduce_sum(sampeld_mlm_acc*tf.cast(sampled_masked_lm_weights, dtype=tf.float32))
      sampeld_mlm_acc /= (1e-10+tf.reduce_sum(tf.cast(sampled_masked_lm_weights, dtype=tf.float32)))
      monitor_dict['sampled_mlm_acc'] = sampeld_mlm_acc

      token_pred_acc = tf.cast(tf.equal(d["disc_preds"], d['disc_labels']),
                                dtype=tf.float32)
      label_weights = tf.cast(d["input_mask"], dtype=tf.float32)
      token_pred_acc = tf.reduce_sum(token_pred_acc*label_weights, axis=-1)
      token_pred_acc /= tf.reduce_sum(label_weights, axis=-1)

      monitor_dict['token_pred_acc'] = tf.reduce_mean(token_pred_acc)
      monitor_dict['disc_loss'] = tf.reduce_mean(d['disc_loss'])

      return monitor_dict

    eval_fn_keys = eval_fn_inputs.keys()
    self.monitor_dict = monitor_fn(eval_fn_inputs, eval_fn_keys)
    eval_fn_values = [eval_fn_inputs[k] for k in eval_fn_keys]

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
      if config.electra_objective:
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

  def _get_disc_masked_lm_output(self, inputs, model):
    """Masked language modeling_electra softmax layer."""
    masked_lm_weights = inputs.masked_lm_weights
    with tf.variable_scope("cls/predictions"):
      if self._config.uniform_generator:
        logits = tf.zeros(self._bert_config.vocab_size)
        logits_tiled = tf.zeros(
            modeling_electra.get_shape_list(inputs.masked_lm_ids) +
            [self._bert_config.vocab_size])
        logits_tiled += tf.reshape(logits, [1, 1, self._bert_config.vocab_size])
        logits = logits_tiled
      else:
        relevant_hidden = pretrain_helpers.gather_positions(
            model.get_sequence_output(), inputs.masked_lm_positions)
        hidden = tf.layers.dense(
            relevant_hidden,
            units=modeling_electra.get_shape_list(model.get_embedding_table())[-1],
            activation=modeling_electra.get_activation(self._bert_config.hidden_act),
            kernel_initializer=modeling_electra.create_initializer(
                self._bert_config.initializer_range))
        hidden = modeling_electra.layer_norm(hidden)
        output_bias = tf.get_variable(
            "output_bias",
            shape=[self._bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(hidden, model.get_embedding_table(),
                           transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

      oh_labels = tf.one_hot(
          inputs.masked_lm_ids, depth=self._bert_config.vocab_size,
          dtype=tf.float32)

      probs = tf.nn.softmax(logits)
      log_probs = tf.nn.log_softmax(logits)
      label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)

      numerator = tf.reduce_sum(inputs.masked_lm_weights * label_log_probs)
      denominator = tf.reduce_sum(masked_lm_weights) + 1e-6
      loss = numerator / denominator
      preds = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

      MLMOutput = collections.namedtuple(
          "MLMOutput", ["logits", "probs", "loss", "per_example_loss", "preds"])
      return MLMOutput(
          logits=logits, probs=probs, per_example_loss=label_log_probs,
          loss=loss, preds=preds)

  def _get_masked_lm_output(self, inputs, model):
    """Masked language modeling_electra softmax layer."""
    masked_lm_weights = inputs.masked_lm_weights
    with tf.variable_scope("generator_predictions"):
      if self._config.uniform_generator:
        logits = tf.zeros(self._bert_config.vocab_size)
        logits_tiled = tf.zeros(
            modeling_electra.get_shape_list(inputs.masked_lm_ids) +
            [self._bert_config.vocab_size])
        logits_tiled += tf.reshape(logits, [1, 1, self._bert_config.vocab_size])
        logits = logits_tiled
      else:
        relevant_hidden = pretrain_helpers.gather_positions(
            model.get_sequence_output(), inputs.masked_lm_positions)
        hidden = tf.layers.dense(
            relevant_hidden,
            units=modeling_electra.get_shape_list(model.get_embedding_table())[-1],
            activation=modeling_electra.get_activation(self._bert_config.hidden_act),
            kernel_initializer=modeling_electra.create_initializer(
                self._bert_config.initializer_range))
        hidden = modeling_electra.layer_norm(hidden)
        output_bias = tf.get_variable(
            "output_bias",
            shape=[self._bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(hidden, model.get_embedding_table(),
                           transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

      oh_labels = tf.one_hot(
          inputs.masked_lm_ids, depth=self._bert_config.vocab_size,
          dtype=tf.float32)

      probs = tf.nn.softmax(logits)
      log_probs = tf.nn.log_softmax(logits)
      label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)

      numerator = tf.reduce_sum(inputs.masked_lm_weights * label_log_probs)
      denominator = tf.reduce_sum(masked_lm_weights) + 1e-6
      loss = numerator / denominator
      preds = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

      MLMOutput = collections.namedtuple(
          "MLMOutput", ["logits", "probs", "loss", "per_example_loss", "preds"])
      return MLMOutput(
          logits=logits, probs=probs, per_example_loss=label_log_probs,
          loss=loss, preds=preds)

  def _get_discriminator_output(self, inputs, discriminator, labels):
    """Discriminator binary classifier."""
    with tf.variable_scope("discriminator_predictions"):
      hidden = tf.layers.dense(
          discriminator.get_sequence_output(),
          units=self._bert_config.hidden_size,
          activation=modeling_electra.get_activation(self._bert_config.hidden_act),
          kernel_initializer=modeling_electra.create_initializer(
              self._bert_config.initializer_range))
      logits = tf.squeeze(tf.layers.dense(hidden, units=1), -1)
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
    inputs = pretrain_helpers.unmask(inputs)
    disallow = tf.one_hot(
        inputs.masked_lm_ids, depth=self._bert_config.vocab_size,
        dtype=tf.float32) if self._config.disallow_correct else None
    sampled_tokens = tf.stop_gradient(pretrain_helpers.sample_from_softmax(
        mlm_logits / self._config.temperature, disallow=disallow))
    sampled_tokids = tf.argmax(sampled_tokens, -1, output_type=tf.int32)
    updated_input_ids, masked = pretrain_helpers.scatter_update(
        inputs.input_ids, sampled_tokids, inputs.masked_lm_positions)
    labels = masked * (1 - tf.cast(
        tf.equal(updated_input_ids, inputs.input_ids), tf.int32))
    updated_inputs = pretrain_data.get_updated_inputs(
        inputs, input_ids=updated_input_ids)
    FakedData = collections.namedtuple("FakedData", [
        "inputs", "is_fake_tokens", "sampled_tokens"])
    return FakedData(inputs=updated_inputs, is_fake_tokens=labels,
                     sampled_tokens=sampled_tokens)

  def _build_transformer(self, inputs, is_training,
                         bert_config=None, name="electra", reuse=False, **kwargs):
    """Build a transformer encoder network."""
    if bert_config is None:
      bert_config = self._bert_config
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
      return modeling_electra.BertModel(
          bert_config=bert_config,
          is_training=is_training,
          input_ids=inputs.input_ids,
          input_mask=inputs.input_mask,
          token_type_ids=inputs.segment_ids,
          use_one_hot_embeddings=self._config.use_tpu,
          scope=name,
          **kwargs)


def get_generator_config(config,
                         bert_config):
  """Get model config for the generator network."""
  gen_config = modeling_electra.BertConfig.from_dict(bert_config.to_dict())
  gen_config.hidden_size = int(round(
      bert_config.hidden_size * config.generator_hidden_size))
  gen_config.num_hidden_layers = int(round(
      bert_config.num_hidden_layers * config.generator_layers))
  gen_config.intermediate_size = 4 * gen_config.hidden_size
  # gen_config.num_attention_heads = max(1, gen_config.hidden_size // 64)
  gen_config.num_attention_heads = int(max(1, gen_config.num_attention_heads * config.generator_hidden_size))
  return gen_config


def model_fn_builder(config):
  """Build the model for training."""

  def model_fn(features, labels, mode, params):
    """Build the model for training."""
    model = PretrainingModel(config, features,
                             mode == tf.estimator.ModeKeys.TRAIN)
    # utils.log("Model is built!")
    tf.logging.info("Model is built!")
    def count_params():
      n = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
      # utils.log("Model size: %dK" % (n/1000))
      tf.logging.info("Model size: %dK" % (n/1000))
    count_params()
    tvars = tf.trainable_variables()
    for tvar in tvars:
      tf.logging.info(tvar)
    if mode == tf.estimator.ModeKeys.TRAIN:

      if config.monitoring:
        if model.monitor_dict:
          host_call = log_utils.construct_scalar_host_call_v1(
                                    monitor_dict=model.monitor_dict,
                                    model_dir=config.model_dir,
                                    prefix="train/")
        else:
          host_call = None
        print("==host_call==", host_call)

      train_op, _ = optimization.create_optimizer(
          model.total_loss, config.learning_rate, config.num_train_steps,
          weight_decay_rate=config.weight_decay_rate,
          use_tpu=config.use_tpu,
          warmup_steps=config.num_warmup_steps,
          lr_decay_power=config.lr_decay_power
      )
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=model.total_loss,
          train_op=train_op,
          # training_hooks=[training_utils.ETAHook(
          #     {} if config.use_tpu else dict(loss=model.total_loss),
          #     config.num_train_steps, config.iterations_per_loop,
          #     config.use_tpu,100)],
          host_call=host_call if config.monitoring else None
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
  if config.debug:
    utils.rmkdir(config.model_dir)
  # utils.heading("Config:")
  # utils.log_config(config)
  tf.logging.info("Config:")
  tf.logging.info(config)

  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  tpu_cluster_resolver = None
  if config.use_tpu and config.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        config.tpu_name, zone=config.tpu_zone, project=config.gcp_project)
  tpu_config = tf.estimator.tpu.TPUConfig(
      iterations_per_loop=config.iterations_per_loop,
      num_shards=(config.num_tpu_cores if config.do_train else
                  config.num_tpu_cores),
      # tpu_job_name=("train_tpu_worker" if config.do_train else
      #               "lm_eval_tpu_worker"),
      per_host_input_for_training=is_per_host)
  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=config.model_dir,
      save_checkpoints_steps=config.save_checkpoints_steps,
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
    estimator.train(input_fn=pretrain_data.get_input_fn(config, True),
                    max_steps=config.num_train_steps)
  if config.do_eval:
    # utils.heading("Running evaluation")
    tf.logging.info("Running evaluation")
    result = estimator.evaluate(
        input_fn=pretrain_data.get_input_fn(config, False),
        steps=config.num_eval_steps)
    for key in sorted(result.keys()):
      # utils.log("  {:} = {:}".format(key, str(result[key])))
      tf.logging.info("  {:} = {:}".format(key, str(result[key])))
    return result


def train_one_step(config):
  """Builds an ELECTRA/ConvBERT model an trains it for one step; useful for debugging."""
  train_input_fn = pretrain_data.get_input_fn(config, True)
  features = tf.data.make_one_shot_iterator(train_input_fn(dict(
      batch_size=config.train_batch_size))).get_next()
  model = PretrainingModel(config, features, True)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    utils.log(sess.run(model.total_loss))


def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-dir", required=True,
                      help="Location of data files (model weights, etc).")
  parser.add_argument("--data-file-list", required=True,
                      help="Location of data files (model weights, etc).")
  parser.add_argument("--model-name", required=True,
                      help="The name of the model being fine-tuned.")
  parser.add_argument("--hparams", default="{}",
                      help=" JSON dict of model hyperparameters.")
  args = parser.parse_args()
  if args.hparams.endswith(".json"):
    hparams = utils.load_json(args.hparams)
  else:
    hparams = json.loads(args.hparams)
  # tf.logging.set_verbosity(tf.logging.ERROR)
  train_or_eval(configure_pretraining.PretrainingConfig(
      args.model_name, args.data_dir, args.data_file_list, **hparams))


if __name__ == "__main__":
  main()
