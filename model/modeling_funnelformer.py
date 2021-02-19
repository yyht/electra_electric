import numpy as np
import tensorflow as tf
import copy
from model.funnel_transformer_utils import funnel_transformer_modules_v1 as funnel_transformer_modules
from model.funnel_transformer_utils import funnel_transformer_utils_v1 as funnel_transformer_utils
from model.funnel_transformer_utils import funnel_transformer_ops_v1 as funnel_transformer_ops

import numpy as np
import tensorflow as tf
import copy
import os, json
from bunch import Bunch
import collections
import copy
import json
import math
import re

import numpy as np
import six

def parse_depth_string(depth_str):
  depth_config = depth_str.split("x")
  if len(depth_config) == 1:
    depth_config.append(1)
  assert len(depth_config) == 2, "Require two-element depth config."

  return list(map(int, depth_config))

def get_initializer(net_config):
  """Get variable intializer."""
  net_config = net_config
  if net_config.init == "uniform":
    initializer = tf.initializers.random_uniform(
        minval=-net_config.init_range,
        maxval=net_config.init_range,
        seed=None)
  elif net_config.init == "normal":
    initializer = tf.initializers.random_normal(
        stddev=net_config.init_std,
        seed=None)
  elif net_config.init == "truncated_normal":
    initializer = tf.initializers.truncated_normal(
        stddev=net_config.init_std,
        seed=None)
  else:
    raise ValueError("Initializer {} not supported".format(net_config.init))
  return initializer

def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

def get_activation(activation_type):
  """Get the corresponding activation function from string."""
  if activation_type == "relu":
    activation = tf.nn.relu
  elif activation_type == "gelu":
    activation = gelu
  elif activation_type == "tanh":
    activation = tf.tanh
  else:
    raise ValueError("Unsupported activation type {}".format(activation_type))

  return activation

class ModelConfig(object):
  """ModelConfig contains fixed hyperparameters of a Funnel-Transformer."""

  keys = ["block_size", "vocab_size", "d_embed", "d_model", "n_head", "d_head",
          "d_inner", "ff_activation", "dropout", "dropatt", "dropact",
          "init", "init_std", "init_range", "rel_attn_type", "separate_cls",
          "pooling_type", "pooling_size", "pool_q_only", "decoder_size",
          "if_skip_connetion", "corrupted",
          'use_bfloat16', "verbose", "truncate_seq",
          "seg_id_cls", "scope", "hidden_size", 'initializer_range',
          "hidden_act", "max_position_embeddings", "num_hidden_layers"]

  def __init__(self, block_size, vocab_size, d_embed, d_model, n_head,
        d_head, d_inner, dropout, dropatt, dropact, ff_activation,
        init="truncated_normal", init_std=0.02, init_range=0.1,
        rel_attn_type="factorized", separate_cls=True,
        pooling_type="mean", pooling_size=2, pool_q_only=True,
        decoder_size="0",
        if_skip_connetion=False,
        corrupted=False,
        use_bfloat16=False,
        verbose=False,
        truncate_seq=True,
        seg_id_cls=2,
        scope="model", 
        hidden_size=768,
        initializer_range=0.01,
        hidden_act='gelu',
        max_position_embeddings=512,
        num_hidden_layers=4):

    """Initialize model config."""
    self.vocab_size = vocab_size
    self.if_skip_connetion = if_skip_connetion
    self.corrupted = corrupted
    self.use_bfloat16 = use_bfloat16
    self.truncate_seq = truncate_seq
    self.scope = scope
    self.verbose = verbose
    self.d_embed = d_embed
    self.d_model = d_model
    self.n_head = n_head
    self.d_head = d_head
    self.d_inner = d_inner
    self.hidden_size = self.d_model
    self.seg_id_cls = seg_id_cls
    self.dropout = dropout
    self.dropatt = dropatt
    self.dropact = dropact
    self.ff_activation = ff_activation
    self.hidden_act = ff_activation
    self.init = init
    self.init_std = init_std
    self.init_range = init_range
    self.initializer_range = initializer_range
    self.rel_attn_type = rel_attn_type
    self.block_size = block_size
    self.decoder_size = decoder_size
    self.pooling_type = pooling_type
    self.pooling_size = pooling_size
    self.pool_q_only = pool_q_only
    self.separate_cls = separate_cls
    self.max_position_embeddings = max_position_embeddings
    self.num_hidden_layers = num_hidden_layers

  @staticmethod
  def init_from_text(file_path, sep_symbol=None):
    """Initialize ModelConfig from a text file."""
    tf.logging.info("Initialize ModelConfig from text file %s.", file_path)
    args = {}
    with tf.io.gfile.GFile(file_path) as f:
      for line in f:
        k, v = line.strip().split(sep_symbol)
        if k in ModelConfig.keys:
          args[k] = v
        else:
          tf.logging.warning("Unused key %s", k)

    args = ModelConfig.overwrite_args(args)
    net_config = ModelConfig(**args)

    return net_config

  @staticmethod
  def init_from_json(file_path):
    """Initialize ModelConfig from a json file."""
    tf.logging.info("Initialize ModelConfig from json file %s.", file_path)
    with tf.io.gfile.GFile(file_path) as f:
      json_data = json.load(f)
      json_data["rel_attn_type"] = "factorized"
      tf.logging.info("Change rel_attn_type to `rel_shift` for non-TPU env.")
    net_config = ModelConfig(**json_data)
    return net_config

  def to_json(self, json_path):
    """Save ModelConfig to a json file."""
    tf.logging.info("Save ModelConfig to json file %s.", json_path)
    json_data = {}
    for key in ModelConfig.keys:
      json_data[key] = getattr(self, key)

    json_dir = os.path.dirname(json_path)
    if not tf.io.gfile.exists(json_dir):
      tf.io.gfile.makedirs(json_dir)
    with tf.io.gfile.GFile(json_path, "w") as f:
      json.dump(json_data, f, indent=4, sort_keys=True)


class FunnelTFM(object):
  def __init__(self,
                 bert_config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=True,
                 scope=None,
                 embedding_size=None,
                 input_embeddings=None,
                 input_reprs=None,
                 update_embeddings=True,
                 untied_embeddings=False,
                 compute_type=tf.float32,
                 use_tpu=False,
                 use_bfloat16=False):

    self.config = bert_config

    self.ret_dict = {}
    self.block_size = self.config.block_size
    self.block_depth = []
    self.block_param_size = []
    self.block_repeat_size = []
    for cur_block_size in self.block_size.split("_"):
      cur_block_size = parse_depth_string(cur_block_size)
      self.block_depth.append(cur_block_size[0] * cur_block_size[1])
      self.block_param_size.append(cur_block_size[0])
      self.block_repeat_size.append(cur_block_size[1])
    self.n_block = len(self.block_depth)
    self.config.initializer_range = self.config.init_std

    # assert not (self.n_block == 1 and decoder_size != "0"), \
    #     "Models with only 1 block does NOT need a decoder."
    self.decoder_size = self.config.decoder_size
    decoder_size = parse_depth_string(self.decoder_size)
    self.decoder_depth = decoder_size[0] * decoder_size[1]
    self.decoder_param_size = decoder_size[0]
    self.decoder_repeat_size = decoder_size[1]

    self.config.n_block = self.n_block
    self.config.block_depth = self.block_depth
    self.config.block_param_size = self.block_param_size
    self.config.block_repeat_size = self.block_repeat_size

    self.config.decoder_depth = decoder_size[0] * decoder_size[1]
    self.config.decoder_param_size = decoder_size[0]
    self.config.decoder_repeat_size = decoder_size[1]
    self.attn_structures = None
    self.initializer_range = self.config.init_std

    initializer = get_initializer(self.config)
    embedding_scope = self.config.scope
    tf.logging.info("==using embedding scope of original model_config.embedding_scope: %s ==", embedding_scope)

    cls_token_type = self.config.seg_id_cls * tf.ones_like(token_type_ids)[:, 0:1]
    token_type_ids = tf.concat([cls_token_type, token_type_ids[:, 1:]], axis=1)

    input_mask = 1.0 - tf.cast(input_mask, dtype=tf.float32)

    dtype = tf.float32 if not use_bfloat16 else tf.bfloat16
    with tf.variable_scope(embedding_scope, reuse=tf.AUTO_REUSE):
      embed_name = os.path.join(embedding_scope, 'embed')
      [self.input_embed, 
      self.word_embed_table, 
      self.emb_dict] = funnel_transformer_modules.input_embedding(
        self.config,
        initializer,
        input_ids, 
        is_training, 
        seg_id=token_type_ids, 
        use_tpu=use_tpu, 
        dtype=dtype,
        name=embed_name)

    scope = self.config.scope
    self.attn_structures = None

    embedding_seq_output = self.input_embed
    tf.logging.info("****** self-embedding_seq_output *******")

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      [self.encoder_output, 
        self.encoder_hiddens, 
        self.enc_dict,
        self.attn_structures] = funnel_transformer_modules.encoder(
          self.config,
          embedding_seq_output,
          is_training,
          initializer,
          seg_id=token_type_ids,
          input_mask=input_mask,
          attn_structures=self.attn_structures)
      print(self.attn_structures, "==attention structures==")

    # decoder
    if self.config.n_block > 1:
      decoder_initializer = get_initializer(self.config)
      scope = self.config.get("scope", "model")
      with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        [self.decoder_output, 
        self.dec_dict] = funnel_transformer_modules.decoder(
            self.config,
            self.encoder_output,
            input_mask=input_mask,
            seg_id=token_type_ids,
            is_training=is_training,
            initializer=decoder_initializer,
            attn_structures=self.attn_structures)
      funnel_transformer_ops.update_ret_dict(self.ret_dict, 
                                            self.dec_dict, 
                                            "dec")

    scope = self.config.scope
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      self.sequence_output = self.encoder_hiddens[-1]

      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        initializer = get_initializer(self.config)
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            # first_token_tensor,
            self.encoder_output[:, 0],
            self.config.d_model,
            activation=tf.tanh,
            kernel_initializer=initializer,
            use_bias=True)

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    if self.config.n_block > 1:
      return self.decoder_output
    else:
      return self.encoder_output

  def get_all_encoder_layers(self):
    return self.encoder_hiddens

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.input_embed

  def get_embedding_table(self):
    return self.word_embed_table
  
def get_assignment_map_from_checkpoint(tvars, init_checkpoint, prefix=""):
  """Compute the union of the current variables and checkpoint variables."""
  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  initialized_variable_names = {}
  assignment_map = collections.OrderedDict()
  for x in tf.train.list_variables(init_checkpoint):
    (name, var) = (x[0], x[1])
    if prefix + name not in name_to_variable:
      continue
    assignment_map[name] = prefix + name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return assignment_map, initialized_variable_names

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if isinstance(tensor, np.ndarray) or isinstance(tensor, list):
    shape = np.array(tensor).shape
    if isinstance(expected_rank, six.integer_types):
      assert len(shape) == expected_rank
    elif expected_rank is not None:
      assert len(shape) in expected_rank
    return shape

  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
