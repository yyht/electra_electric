
import numpy as np
import tensorflow as tf

"""
Adapted from https://github.com/gpeyre/SinkhornAutoDiff
and from https://github.com/dfdazac/wassdistance/blob/master/layers.py
and from https://github.com/michaelsdr/sinkformers/blob/main/nlp-tutorial/text-classification-transformer/sinkhorn.py
"""

def shape_list(x, out_type=tf.int32):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x, out_type=out_type)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def sinkhorn_distance(input_tensor, 
                  eps, 
                  max_iter, 
                  reduction='none',
                  stopThr=1e-2):
  
  C = input_tensor
  C_shape = shape_list(C)

  x_points = C_shape[-2]
  y_points = C_shape[-1]
  batch_size = C_shape[0]

  # both marginals are fixed with equal weights
  mu = 1.0 / x_points * tf.ones((batch_size, x_points))
  nu = 1.0 / y_points * tf.ones((batch_size, y_points))

  u = tf.zeros_like(mu)
  v = tf.zeros_like(nu)

  cpt = tf.constant(0)
  err = tf.constant(1.0)

  c = lambda cpt, u, v, err: tf.logical_and(cpt < max_iter, err > stopThr)

  def M( C, u, v):
    "Modified cost for logarithmic updates"
    "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
    return (-C + tf.expand_dims(u, -1) + tf.expand_dims(v, -2) )/eps

  def loop_func(cpt, u, v, err):
    u1 = tf.identity(u)  # useful to check the update

    cpt = cpt + 1

    u = eps * (tf.log(mu+1e-8) - tf.reduce_logsumexp(M(C, u, v), axis=-1)) + u
    v = eps * (tf.log(nu+1e-8) - tf.reduce_logsumexp(tf.transpose(M(C, u, v), [0, 2, 1]), axis=-1)) + v

    err = tf.reduce_mean(tf.reduce_sum(tf.abs(u - u1), axis=-1))

    return cpt, u, v, err

  _, u_final, v_final, _ = tf.while_loop(c, loop_func, loop_vars=[cpt, u, v, err])
  U, V = tf.identity(u_final), tf.identity(v_final)

  # Transport plan pi = diag(a)*K*diag(b)
  pi = tf.exp(M(C, U, V))

  return pi, C, U, V

