
import tensorflow as tf

def iterative_inv(mat, n_iter=6):

  """
  https://downloads.hindawi.com/journals/aaa/2014/563787.pdf
  A New Iterative Method for Finding Approximate Inverses of
  Complex Matrices
  """

  mat_shape = bert_utils.get_shape_list(mat, expected_rank=[2,3,4])
  I = tf.cast(tf.eye(mat_shape[-1]), dtype=tf.float32)
  K = tf.identity(mat) 
  # [B, N, n-landmarks, n-landmarks]
  V = 1 / (tf.reduce_max(tf.reduce_sum(tf.abs(K), axis=-2)) * tf.reduce_max(tf.reduce_sum(tf.abs(K), axis=-1))) * tf.transpose(K, [0,1,3,2])

  for _ in range(n_iter):
      KV = tf.matmul(K, V)
      V = tf.matmul(0.25 * V, 13 * I - tf.matmul(KV, 15 * I - tf.matmul(KV, 7 * I - KV)))
  # [B, N, n-landmarks, n-landmarks]
  V = tf.stop_gradient(V)
  return V

