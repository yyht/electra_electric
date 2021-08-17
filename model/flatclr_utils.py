
import tensorflow as tf
import numpy as np

def shape_list(x, out_type=tf.int32):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x, out_type=out_type)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def flatclr(anchor_features, 
            target_features,
            labels,
            temperature=0.1):

  """
  https://github.com/yyht/FlatCLR/blob/main/flatclr.py
  """

  # anchor_features = tf.nn.l2_normalize(anchor_features, axis=-1)
  # target_features = tf.nn.l2_normalize(target_features, axis=-1)

  anchor_shape = shape_list(anchor_features)
  target_shape = shape_list(target_features)
  # mask = tf.eye(anchor_shape[0])

  mask = tf.one_hot(tf.range(anchor_shape[0]), target_shape[0])
  mask = tf.cast(mask, dtype=tf.float32)

  similarity_matrix = tf.matmul(anchor_features, target_features, transpose_b=True)
  similarity_matrix /= temperature

  # [batch_size, num_classes]
  mask = tf.one_hot(labels, depth=target_shape[0])
  positives = tf.reduce_sum(similarity_matrix * mask, axis=-1, keep_dims=True)

  # [batch_size, batch_size]
  negatives = similarity_matrix * (1-mask) - mask * 1e20

  # # [batch_size, 1]
  # positives = tf.gather_nd(
  #     similarity_matrix,
  #     tf.where(mask))
  # positives = tf.expand_dims(positives, axis=-1)

  # # [batch_size, batch_size-1]
  # negatives = tf.gather_nd(
  #     similarity_matrix,
  #     tf.where(1-mask)).reshape([anchor_shape[0], -1])

  # [batch_size, batch_size]
  logits = negatives - positives

  # [batch_size, batch_size-1]
  v = tf.reduce_logsumexp(logits, axis=-1, keep_dims=True) #(512,1)
  per_example_flatnce_loss = tf.exp(v-tf.stop_gradient(v))
  flatnce_loss = tf.reduce_mean(per_example_flatnce_loss)

  # since flatnce loss will always be 1, so we remove it
  flatnce_loss = flatnce_loss - 1
  per_example_infonce_loss = tf.nn.softmax_cross_entropy_with_logits(
                        logits=similarity_matrix, 
                        labels=tf.stop_gradient(labels))

  infonce_loss = tf.reduce_mean(per_example_infonce_loss)

  total_loss = flatnce_loss + tf.stop_gradient(infonce_loss)

  return (per_example_flatnce_loss, 
          flatnce_loss,
          per_example_infonce_loss,
          infonce_loss,
          total_loss
          )


