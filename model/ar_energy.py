
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

    queue = tf.get_variable('queue', 
              [4, seq_len, vocab_size], 
              dtype=tf.float32,
              initializer=tf.random_normal_initializer(
              mean=0.0, stddev=1.0, seed=None, dtype=tf.dtypes.float32),
              trainable=False)

    mask = tf.cast(onehot_labels, dtype=tf.float32)

    # [batch_size, seq_len]   abc
    seq_mask = tf.cast(input_mask, dtype=tf.float32)
    # [batch_size, seq_len, 1]
    seq_mask = tf.expand_dims(seq_mask, axis=-1)

    total_mask = mask * seq_mask

    # [batch_size, seq_len, vocab_size]
    # logits_norm = tf.nn.l2_normalize(logits, axis=-1)
    # logits_norm /= 0.1

    # using raw logits
    logits_norm = logits

    # current negative logits
    Z = tf.reduce_logsumexp(logits_norm-(1-mask)*1e10, axis=0)
    # [1, seq_len, vocab_size]
    Z = tf.expand_dims(Z, axis=0)
 
    queue_buffer = tf.reduce_logsumexp(tf.stop_gradient(queue), axis=0)
    queue_buffer = tf.expand_dims(queue_buffer, axis=0)

    Z_all = tf.reduce_logsumexp(tf.concat([Z, tf.stop_gradient(queue_buffer)], axis=0), axis=0)
    Z_all = tf.expand_dims(Z_all, axis=0)

    tf.logging.info("*** Z_all ***")
    tf.logging.info(Z_all)

    # [batch_size, seq_len, vocab_size]
    per_example_loss =  tf.nn.sigmoid_cross_entropy_with_logits(
        labels=total_mask, 
        logits=logits_norm-Z_all
    )

    numerator = tf.reduce_sum(per_example_loss * total_mask)
    denominator = tf.reduce_sum(total_mask) + 1e-10

    loss = numerator / denominator

    # reference from https://github.com/EncodeTS/TensorFlow_Center_Loss/blob/master/mnist_sample_code/mnist_with_center_loss.ipynb
    # get loss then update
    
    queue_op = queue.assign(tf.concat([Z, queue[:-1, :, :]], axis=0))
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, queue_op)

    return per_example_loss, loss, queue_buffer
