import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def _dot_similarity_dim1(x, y):
    # (N, 1, C), (N, C, 1) -> (N, 1, 1)
    v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
    return v


def _dot_similarity_dim2(x, y):
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
    # (N, 1, C), (1, C, 2N) -> (N, 2N)
    return v


def get_negative_mask(batch_size):
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)


def _nan_to_zero(x):
    return tf.cond(
        tf.math.is_finite(x), lambda: x, lambda: tf.constant(0, dtype=tf.float32))


def pearson_correlation(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    corr = tfp.stats.correlation(y_true, y_pred, sample_axis=-1, event_axis=None)
    corr = tf.map_fn(_nan_to_zero, corr)
    return corr
