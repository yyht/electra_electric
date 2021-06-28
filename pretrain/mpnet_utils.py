import tensorflow as tf
tf.disable_v2_behavior()

# import tensorflow as tf
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

import numpy as np

def shape_list(x, out_type=tf.int32):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x, out_type=out_type)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = shape_list(from_tensor)
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = shape_list(to_mask,)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask

def recurrence_mask(input_tensor, pad_mask, is_target):
  """
  array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0.]],
      dtype=float32)
  that for each masked position, we use current all data to predict,
  just like the auto-regressive model that first predicted tokens are conditioned
  on current-time all unmasked input
  which could be used for masked token relationship modeling
  unlike text-infilling that modify the input sequence order,
  we could just use ordinary sequence input
  """
  mask_sequence = tf.cast(is_target, tf.float32)

  attention_mask = 1.0 - create_attention_mask_from_input_mask(input_tensor, mask_sequence)

  seq_shape = shape_list(mask_sequence)
  seq_len = seq_shape[1]
  ones = tf.ones((1, seq_len, seq_len))
  a_mask = tf.matrix_band_part(ones, -1, 0)
  a_mask = tf.roll(a_mask, -1, axis=-1)
  s_ex12 = tf.expand_dims(tf.expand_dims(mask_sequence, 1), 2)
  s_ex13 = tf.expand_dims(tf.expand_dims(mask_sequence, 1), 3)
  a_mask = (1 - s_ex13) * (1 - s_ex12) + s_ex13 * a_mask
  # generate mask of batch x seq_len x seq_len
  a_mask = tf.reshape(a_mask, (-1, seq_len, seq_len))
  out_mask = attention_mask + a_mask
  out_mask = tf.cast(out_mask, tf.float32)
  out_mask = tf.cast(tf.greater_equal(out_mask, 1), dtype=tf.float32)
  return out_mask