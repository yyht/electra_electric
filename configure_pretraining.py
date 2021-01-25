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

"""Config controlling hyperparameters for pre-training ELECTRA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tensorflow as tf

class PretrainingConfig(object):
  """Defines pre-training hyperparameters."""

  def __init__(self, model_name, data_dir, data_file_list, **kwargs):
    self.model_name = model_name
    self.debug = False  # debug mode for quickly running things
    self.do_train = True  # pre-train ELECTRA
    self.do_eval = False  # evaluate generator/discriminator on unlabeled data
    self.mask_strategy = 'electra'

    # loss functions
    # train ELECTRA or Electric? if both are false, trains a masked LM like BERT
    self.electra_objective = True
    self.electric_objective = False
    self.electra_nce_objective = False
    self.gen_weight = 1.0  # masked language modeling / generator loss
    self.disc_weight = 50.0  # discriminator loss
    self.mask_prob = 0.15  # percent of input tokens to mask out / replace
    self.nce = 'gan'
    self.logprob_avg = False
    self.stage = 'one_stage'
    self.gen_learning_rate = 5e-5
    self.disc_learning_rate = 1e-3
    self.spectral_regularization = False
    # optimization
    self.learning_rate = 5e-4
    self.lr_decay_power = 1.0  # linear weight decay by default
    self.weight_decay_rate = 0.01
    self.num_warmup_steps = 10000
    self.initial_ratio = 0.2
    self.final_ratio = 0.2

    self.use_pretrained_generator = False
    self.use_pretrained_discriminator = False

    # training settings
    self.iterations_per_loop = 200
    self.save_checkpoints_steps = 1000
    self.num_train_steps = 1000000
    self.num_eval_steps = 100
    self.keep_checkpoint_max = 5 # maximum number of recent checkpoint files to keep;
                                 # change to 0 or None to keep all checkpoints

    self.model_scope = 'electra'
    # model settings
    self.model_size = "base"  # one of "small", "base", or "large"
    # override the default transformer hparams for the provided model size; see
    # modeling.BertConfig for the possible hparams and util.training_utils for
    # the defaults

    self.model_size_generator = 'tiny'

    self.model_hparam_overrides = (
        kwargs["model_hparam_overrides"]
        if "model_hparam_overrides" in kwargs else {})
    self.embedding_size = None  # bert hidden size by default
    self.vocab_size = 30522  # number of tokens in the vocabulary
    self.do_lower_case = True  # lowercase the input?
    self.monitoring = True

    # generator settings
    self.uniform_generator = False  # generator is uniform at random
    self.two_tower_generator = False  # generator is a two-tower cloze model
    self.untied_generator_embeddings = False  # tie generator/discriminator
                                              # token embeddings?
    self.untied_generator = True  # tie all generator/discriminator weights?
    self.generator_layers = 1.0  # frac of discriminator layers for generator
    self.generator_hidden_size = 0.25  # frac of discrim hidden size for gen
    self.disallow_correct = False  # force the generator to sample incorrect
                                   # tokens (so 15% of tokens are always
                                   # fake)
    self.temperature = 1.0  # temperature for sampling from generator

    # batch sizes
    self.max_seq_length = 128
    self.train_batch_size = 128
    self.eval_batch_size = 128

    # TPU settings
    self.use_tpu = False
    self.num_tpu_cores = 1
    self.tpu_job_name = None
    self.tpu_name = None  # cloud TPU to use for training
    self.tpu_zone = None  # GCE zone where the Cloud TPU is located in
    self.gcp_project = None  # project name for the Cloud TPU-enabled project

    # default locations of data files
    self.pretrain_file = os.path.join(data_dir, data_file_list)
    self.pretrain_tfrecords = []
    with tf.gfile.GFile(self.pretrain_file, "r") as reader:
        for index, line in enumerate(reader):
            content = line.strip()
            if 'tfrecord' in content:
                train_file_path = os.path.join(data_dir, content)
                print(train_file_path, "====train_file_path====")
                self.pretrain_tfrecords.append(train_file_path)
    random.shuffle(self.pretrain_tfrecords)
    tf.logging.info("** total pretrain tfrecords:%s **"%(str(len(self.pretrain_tfrecords))))
    
    self.pretrain_tfrecords = ",".join(self.pretrain_tfrecords)
    self.vocab_file = "./vocab/vocab_ch.txt"

    # print(train_file)
    # self.pretrain_tfrecords = os.path.join(
    #     data_dir, "pretrain_tfrecords/pretrain_data.tfrecord*")
    # self.vocab_file = os.path.join(data_dir, "vocab.txt")
    self.model_dir = os.path.join(data_dir, "models", model_name)
    results_dir = os.path.join(self.model_dir, "results")
    self.results_txt = os.path.join(results_dir, "unsup_results.txt")
    self.results_pkl = os.path.join(results_dir, "unsup_results.pkl")

    # update defaults with passed-in hyperparameters
    self.update(kwargs)

    self.max_predictions_per_seq = int((self.mask_prob + 0.005) *
                                       self.max_seq_length)

     # span-mask-config
    self.min_tok = 1
    self.max_tok = 10
    self.sep_id = 102
    self.pad_id = 0
    self.cls_id = 101
    self.mask_id = 103
    self.leak_ratio = 0.1
    self.rand_ratio = 0.1
    self.vocab_size = 30522
    self.mask_prob = 0.15
    self.sample_strategy = 'token_span'
    self.truncate_seq = False
    self.stride = 1
    self.use_bfloat16 = False

    # debug-mode settings
    if self.debug:
      self.train_batch_size = 8
      self.num_train_steps = 20
      self.eval_batch_size = 4
      self.iterations_per_loop = 1
      self.num_eval_steps = 2

    # defaults for different-sized model
    if self.model_size == "small":
      self.embedding_size = 128
    # Here are the hyperparameters we used for larger models; see Table 6 in the
    # paper for the full hyperparameters
    # else:
    #   self.max_seq_length = 512
    #   self.learning_rate = 2e-4
    #   if self.model_size == "base":
    #     self.embedding_size = 768
    #     self.generator_hidden_size = 0.33333
    #     self.train_batch_size = 256
    #   else:
    #     self.embedding_size = 1024
    #     self.mask_prob = 0.25
    #     self.train_batch_size = 2048
    if self.electric_objective:
      self.two_tower_generator = True  # electric requires a two-tower generator

    # passed-in-arguments override (for example) debug-mode defaults
    self.update(kwargs)

  def update(self, kwargs):
    for k, v in kwargs.items():
      if k not in self.__dict__:
        raise ValueError("Unknown hparam " + k)
      self.__dict__[k] = v
