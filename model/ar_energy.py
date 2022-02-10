
import tensorflow as tf
import numpy as np

def shape_list(x, out_type=tf.int32):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x, out_type=out_type)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def autoregressive_energy(logits, onehot_labels, input_mask, **kargs):
  """
  Hybrid Discriminative-Generative Training via Contrastive Learning
  """
  [batch_size, seq_len, vocab_size] = shape_list(logits)
  with tf.variable_scope("ar_energy"):

    mask = tf.cast(onehot_labels, dtype=tf.float32)

    # # [1, seq_len, vocab_size]
    # vocab_mask = tf.reduce_sum(onehot_labels, axis=0, keep_dims=True)
    # vocab_mask = tf.not_equal(tf.cast(vocab_mask, dtype=tf.int32), 0)
    # vocab_mask = tf.cast(vocab_mask, dtype=tf.float32)

    # [batch_size, seq_len]
    seq_mask = tf.cast(input_mask, dtype=tf.float32)
    # [batch_size, seq_len, 1]
    seq_mask = tf.expand_dims(seq_mask, axis=-1)

    total_mask = mask * seq_mask

    logits /= 0.1 # for contrastive-learning

    # [seq_len, vocab_size]
    # only get negative logits
    Z = tf.reduce_logsumexp(logits, axis=0)
    # [1, seq_len, vocab_size]
    Z = tf.expand_dims(Z, axis=0)

    # [batch_size, seq_len, vocab_size]
    per_example_loss = -(logits - Z) * total_mask

    numerator = tf.reduce_sum(per_example_loss)
    denominator = tf.reduce_sum(total_mask) + 1e-10

    loss = numerator / denominator

    return per_example_loss, loss
