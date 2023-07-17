"""
Minimal SimPer implementation & example training loops.
"""
import tensorflow as tf
from networks import Featurizer, Classifier


@tf.function
def _max_cross_corr(feats_1, feats_2):
    # feats_1: 1 x T(# time stamp)
    # feats_2: M(# aug) x T(# time stamp)
    feats_2 = tf.cast(feats_2, feats_1.dtype)
    feats_1 = feats_1 - tf.math.reduce_mean(feats_1, axis=-1, keepdims=True)
    feats_2 = feats_2 - tf.math.reduce_mean(feats_2, axis=-1, keepdims=True)

    min_N = min(feats_1.shape[-1], feats_2.shape[-1])
    padded_N = max(feats_1.shape[-1], feats_2.shape[-1]) * 2
    feats_1_pad = tf.pad(feats_1, tf.constant([[0, 0], [0, padded_N - feats_1.shape[-1]]]))
    feats_2_pad = tf.pad(feats_2, tf.constant([[0, 0], [0, padded_N - feats_2.shape[-1]]]))

    feats_1_fft = tf.signal.rfft(feats_1_pad)
    feats_2_fft = tf.signal.rfft(feats_2_pad)
    X = feats_1_fft * tf.math.conj(feats_2_fft)

    power_norm = tf.cast(
        tf.math.reduce_std(feats_1, axis=-1, keepdims=True) *
        tf.math.reduce_std(feats_2, axis=-1, keepdims=True),
        X.dtype)
    power_norm = tf.where(
        tf.equal(power_norm, 0), tf.ones_like(power_norm), power_norm)
    X = X / power_norm

    cc = tf.signal.irfft(X) / (min_N - 1)
    max_cc = tf.math.reduce_max(cc, axis=-1)

    return max_cc


@tf.function
def batched_max_cross_corr(x, y):
    """
    x: M(# aug) x T(# time stamp)
    y: M(# aug) x T(# time stamp)
    """
    # Calculate distance for a single row of x.
    per_x_dist = lambda i: _max_cross_corr(x[i:(i + 1), :], y)
    # Compute and stack distances for all rows of x.
    dist = tf.map_fn(fn=per_x_dist,
                     elems=tf.range(tf.shape(x)[0], dtype=tf.int64),
                     fn_output_signature=x.dtype)
    return dist


@tf.function
def normed_psd(x, fps, zero_pad=0, high_pass=0.25, low_pass=15):
    """ x: M(# aug) x T(# time stamp) """
    x = x - tf.math.reduce_mean(x, axis=-1, keepdims=True)
    if zero_pad > 0:
        L = x.shape[-1]
        x = tf.pad(x, tf.constant([[int(zero_pad / 2 * L), int(zero_pad / 2 * L)]]))

    x = tf.abs(tf.signal.rfft(x)) ** 2

    Fn = fps / 2
    freqs = tf.linspace(0., Fn, x.shape[-1])
    use_freqs = tf.math.logical_and(freqs >= high_pass, freqs <= low_pass)
    use_freqs = tf.repeat(tf.expand_dims(use_freqs, 0), x.shape[0], axis=0)
    x = tf.reshape(x[use_freqs], (x.shape[0], -1))

    # Normalize PSD
    denom = tf.math.reduce_euclidean_norm(x, axis=-1, keepdims=True)
    denom = tf.where(tf.equal(denom, 0), tf.ones_like(denom), denom)
    x = x / denom
    return x


@tf.function
def batched_normed_psd(x, y):
    """
    x: M(# aug) x T(# time stamp)
    y: M(# aug) x T(# time stamp)
    """
    return tf.matmul(normed_psd(x), normed_psd(y), transpose_b=True)


def label_distance(labels_1, labels_2, dist_fn='l1', label_temperature=0.1):
    # labels: bsz x M(#augs)
    # output: bsz x M(#augs) x M(#augs)
    if dist_fn == 'l1':
        dist_mat = - tf.math.abs(labels_1[:, :, None] - labels_2[:, None, :])
    elif dist_fn == 'l2':
        dist_mat = - tf.math.abs(labels_1[:, :, None] - labels_2[:, None, :]) ** 2
    elif dist_fn == 'sqrt':
        dist_mat = - tf.math.abs(labels_1[:, :, None] - labels_2[:, None, :]) ** 0.5
    else:
        raise NotImplementedError(f"`{dist_fn}` not implemented.")

    prob_mat = tf.nn.softmax(dist_mat / label_temperature, axis=-1)
    return prob_mat


class SimPer(tf.keras.Model):

    def __init__(self, hparams):
        super(SimPer, self).__init__()
        self.hparams = hparams
        self.featurizer = Featurizer(self.hparams["n_frames"])
        self.regressor = Classifier(self.featurizer.n_outputs, 1, False)
        self.network = tf.keras.Sequential([self.featurizer, self.regressor])
        self.optimizer = tf.keras.optimizers.Adam(lr=self.hparams["lr"])

    def update(self, minibatches):
        all_x, all_speed = minibatches

        # all_x: [bsz, 2*M, SSL_FRAMES, H, W, C]
        batch_size, num_augments = all_x.shape[0], all_x.shape[1]
        all_x = tf.reshape(all_x, [batch_size * num_augments] + all_x.shape[2:].as_list())

        # [bsz, 2*M] -> [bsz, M, M]
        all_labels = label_distance(all_speed[:, :num_augments // 2],
                                    all_speed[:, num_augments // 2:],
                                    self.hparams["label_dist_fn"],
                                    self.hparams["label_temperature"])

        criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        with tf.GradientTape() as tape:
            all_z = self.featurizer(all_x)
            all_z = tf.reshape(all_z, [batch_size, num_augments, -1])

            loss = 0
            for feats, labels in zip(all_z, all_labels):
                feat_dist = globals()[self.hparams["feat_dist_fn"]](
                    feats[:num_augments // 2], feats[num_augments // 2:])
                loss += criterion(y_pred=feat_dist, y_true=labels)
            loss /= batch_size

        gradients = tape.gradient(loss, self.featurizer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.featurizer.trainable_variables))

        return loss

    def predict(self, x, training: bool):
        return self.featurizer(x, training=training)
