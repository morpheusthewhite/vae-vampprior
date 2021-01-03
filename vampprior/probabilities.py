import numpy as np
import tensorflow as tf

min_epsilon = 1e-5
max_epsilon = 1.-1e-5


def log_bernoulli(x, mean, reduce_dim=None, name=None):
    """
    Log bernoulli. Rec-error between output reconstruction x and ground-truth mean
    @param reduce_dim: dimension of the data attributes, along which to sum the log-prob.
        If tensor has shape (N, sample_size) then provide reduce_dim=1
        If tensor has shape (N, L, sample_size) then provide reduce_dim=2
    """
    probs = tf.clip_by_value(mean, min_epsilon, max_epsilon)
    log_b = x * tf.math.log(probs) + (1. - x) * tf.math.log(1. - probs)
    return tf.reduce_sum(log_b, axis=reduce_dim, name=name)


def log_logistic256(x, mean, logvar, reduce_dim=None, name=None):
    """
    Discretized log-logistic. Similar to log-normal, but with heavier tails.
    @param reduce_dim: dimension of the data attributes, along which to sum the log-prob.
        If tensor has shape (N, sample_size) then provide reduce_dim=1
        If tensor has shape (N, L, sample_size) then provide reduce_dim=2
    """
    binsize = 1. / 256.
    scale = tf.math.exp(logvar)
    x_std = (tf.math.floor(x / binsize) * binsize - mean) / scale
    logp = tf.math.log(tf.sigmoid(x_std + binsize / scale) - tf.sigmoid(x_std) + 1e-7)

    return tf.reduce_sum(logp, axis=reduce_dim, name=name)


def log_normal_standard(x, reduce_dim=None, name=None):
    log2pi = np.log(2 * np.pi)
    log_normal = -.5 * (log2pi + tf.math.pow(x, 2))
    return tf.reduce_sum(log_normal, axis=reduce_dim, name=name)


def log_normal_diag(x, mean, logvar, reduce_dim=None, name=None):
    """
    Multivariate log normal
    @param reduce_dim: dimension of the data attributes, along which to sum the log-prob.
        If tensor has shape (minibatch, sample_size) then provide reduce_dim=1
        If tensor has shape (N, L, sample_size) then provide reduce_dim=2
    """
    log2pi = np.log(2 * np.pi)
    log_normal = -.5 * (log2pi + logvar + tf.math.pow(x - mean, 2) / tf.math.exp(logvar))
    return tf.reduce_sum(log_normal, axis=reduce_dim, name=name)
