
import tensorflow as tf
from model.vqvae_utils import tfidf_utils
import collections
import copy
import json
import math
import re

import numpy as np
import six
import tensorflow as tf

class ETMConfig(object):
  def __init__(self,
               vocab_size,
               topic_size=768,
               hidden_size=768,
               embedding_size=768,
               num_hidden_layers=12,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               initializer_range=0.02,
               apply_bn_vae_mean=True,
               apply_bn_vae_var=True):
    
    self.vocab_size = vocab_size
    self.topic_size = topic_size
    self.hidden_size = hidden_size
    self.embedding_size = embedding_size
    self.num_hidden_layers = num_hidden_layers
    self.hidden_act = hidden_act
    self.hidden_dropout_prob = hidden_dropout_prob
    self.initializer_range = initializer_range
    self.apply_bn_vae_mean = apply_bn_vae_mean
    self.apply_bn_vae_var = apply_bn_vae_var

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = ETMConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class ETM(object):
  def __init__(self,
                etm_config,
               input_term_count,
               input_term_binary,
               input_term_freq,
               is_training=False,
               embedding_matrix=None,
               hidden_vector=None,
               scope=None):
    """
    https://github.com/linkstrife/NVDM-GSM/blob/master/GSM.py
    https://github.com/adjidieng/ETM/blob/master/etm.py
    compared to NVDM, GSM or ETM add topic-word-matrix-alignment
    """
    etm_config = copy.deepcopy(etm_config)
    if not is_training:
      etm_config.hidden_dropout_prob = 0.0

    with tf.variable_scope("etm", scope):
      with tf.variable_scope("bow_embeddings"):
       self.term_count = input_term_count
       self.term_binary = input_term_binary
       self.term_freq = input_term_freq

       tf.logging.info(self.term_count)
       tf.logging.info(self.term_binary)
       tf.logging.info(self.term_freq)

    with tf.variable_scope("etm", scope):
      with tf.variable_scope("encoder"):
        self.q_theta = mlp(input_tensor=self.term_count, 
                          num_hidden_layers=etm_config.num_hidden_layers, 
                          hidden_size=etm_config.hidden_size,
                          is_training=is_training,
                          dropout_prob=etm_config.hidden_dropout_prob,
                          intermediate_act_fn=get_activation(etm_config.hidden_act),
                          initializer_range=etm_config.initializer_range,
                          scope="bow_mlp")

    if hidden_vector is not None:
      with tf.variable_scope("etm", scope):
        if hidden_vector.shape[-1] != etm_config.hidden_size:
          self.hidden_vector = tf.layers.dense(
              hidden_vector, etm_config.hidden_size,
              name="hidden_vector_project")
      self.q_theta = tf.concat([self.q_theta, self.hidden_vector], axis=-1)

    with tf.variable_scope("etm", scope):
      with tf.variable_scope("bridge"):
        self.mu_q_theta = mlp(
                              input_tensor=self.q_theta, 
                              num_hidden_layers=1,
                              hidden_size=etm_config.topic_size,
                              is_training=is_training,
                              dropout_prob=etm_config.hidden_dropout_prob,
                              intermediate_act_fn=None,
                              initializer_range=etm_config.initializer_range,
                              scope="mu_theta_mlp")
        if etm_config.apply_bn_vae_mean:
          with tf.variable_scope("vae_mu_bn"): 
            self.mu_q_theta = tf.layers.batch_normalization(
                      self.mu_q_theta,
                      training=is_training,
                      scale=False,
                      center=False
              )
            self.mu_q_theta = scalar_layer(self.mu_q_theta, tau=0.5, 
              mode='positive', initializer_range=0.02)

        self.sigma_std_q_theta = mlp(
                              input_tensor=self.q_theta, 
                              num_hidden_layers=1,
                              hidden_size=etm_config.topic_size,
                              is_training=is_training,
                              dropout_prob=etm_config.hidden_dropout_prob,
                              intermediate_act_fn=tf.nn.softplus,
                              initializer_range=etm_config.initializer_range,
                              scope="sigma_std_mlp"
                              )
        if etm_config.apply_bn_vae_var:
          with tf.variable_scope("vae_sigma_std_bn"): 
            self.sigma_std_q_theta = tf.layers.batch_normalization(
                      self.sigma_std_q_theta,
                      training=is_training,
                      scale=False,
                      center=False
              )
            self.sigma_std_q_theta = scalar_layer(self.sigma_std_q_theta, tau=0.5, 
              mode='negative', initializer_range=0.02)

    with tf.variable_scope("etm", scope):
      with tf.variable_scope("reparameterize"):
        self.z = reparameterize(self.mu_q_theta, 
                        self.sigma_std_q_theta, is_training)

        # [batch_size, topic_size]
        self.theta = tf.nn.softmax(self.z, dim=-1)

        with tf.variable_scope("embeddings"):
          if embedding_matrix is None:
           self.embedding_table = tf.get_variable(
                  name="vocab_word_embeddings",
                  shape=[etm_config.vocab_size, etm_config.embedding_size],
                  initializer=create_initializer(initializer_range))
          else:
            self.embedding_table = embedding_matrix

        with tf.variable_scope("embeddings"):
          self.topic_embedding_table = tf.get_variable(
                  name="topic_word_embeddings",
                  shape=[etm_config.topic_size, etm_config.embedding_size],
                  initializer=create_initializer(initializer_range))

        self.topic_word_align = tf.matmul(self.topic_embedding_table,
                                        self.embedding_table,
                                        transpose_b=True)
        # [topic_size, vocab_size]
        self.beta = tf.softmax(self.topic_word_align, axis=-1)

        # theta: [batch_size, topic_size]
        # beta : [topic_size, vocab_size]
        # preds: [batch_size, vocab_size]
        self.preds = tf.log(tf.matmul(self.theta, self.beta)+1e-10)
        
  def get_hidden_vector(self):
    return self.z

  def get_vocab_word_embeddings(self):
    return self.embedding_table

  def get_topic_embedding_table(self):
    return self.topic_embedding_table

  def get_recon_loss(self):
    self.per_example_recon_loss = -tf.reduce_sum(self.preds * self.term_count, axis=-1)
    self.recon_loss = tf.reduce_mean(self.per_example_recon_loss)
    return self.recon_loss

  def get_kl_loss(self):
    self.sigma_q_theta = tf.pow(self.sigma_std_q_theta, 2.0)
    self.logsigma_theta = tf.log(self.sigma_q_theta+1e-10)
    self.per_example_kl_theta_loss = -0.5 * tf.reduce_sum(1 + self.logsigma_theta - tf.pow(self.mu_theta, 2) - tf.exp(self.logsigma_theta), dim=-1)
    self.kl_theta_loss = tf.reduce_mean(self.per_example_kl_theta_loss)
    return self.kl_theta_loss


def scalar_layer(input_tensor, tau=0.5, 
              mode='positive', initializer_range=0.02):
  with tf.variable_scope("vae_bn_scale"):
    scale = tf.get_variable(
                  name="scale",
                  shape=[input_tensor.shape[-1]],
                  initializer=create_initializer(initializer_range))
  if mode == 'positive':
      scale = tau + (1 - tau) * tf.nn.sigmoid(scale)
  else:
      scale = (1 - tau) * K.sigmoid(-scale)
  return input_tensor * tf.sqrt(scale)

def gelu(input_tensor):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415

  Args:
    input_tensor: float Tensor to perform activation.

  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf

def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)

def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)

def tokenid2bow(input_ids, vocab_size):
  [term_count, 
  term_binary, 
  term_freq] = tfidf_utils.tokenid2tf(input_ids, vocab_size)

  return term_count, term_binary, term_freq

def mlp(input_tensor, 
        num_hidden_layers, 
        hidden_size,
        is_training,
        dropout_prob,
        intermediate_act_fn,
        initializer_range,
        scope=None
        ):
  prev_output = input_tensor
  with tf.variable_scope(scope, default_name="mlp"):
    for layer_idx in range(num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer_idx):
        layer_input = prev_output
        layer_output = tf.layers.dense(
                layer_input,
                hidden_size,
                kernel_initializer=create_initializer(initializer_range),
                activation=intermediate_act_fn)
        prev_output = layer_output
    final_outputs = prev_output
    return final_outputs

def reparameterize(mu_q_theta, sigma_std_q_theta, is_training):
  if is_training:
    sigma_q_theta = sigma_std_q_theta
    eps = tf.random.normal(get_shape_list(sigma_q_theta), 
                            mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
    return eps*sigma_q_theta+mu_q_theta
  else:
    return mu_q_theta


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
