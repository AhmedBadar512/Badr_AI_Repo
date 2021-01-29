import tensorflow.keras as K
import tensorflow_addons as tfa
import tensorflow as tf


def get_loss(name='cross_entropy'):
    if name == 'cross_entropy':
        loss_func = K.losses.CategoricalCrossentropy(from_logits=True, reduction=K.losses.Reduction.NONE)
    elif name == 'focal_loss':
        loss_func = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=K.losses.Reduction.NONE)
    elif name == 'binary_crossentropy':
        loss_func = K.losses.BinaryCrossentropy(from_logits=True, reduction=K.losses.Reduction.NONE)
    elif name == "MSE":
        loss_func = K.losses.MSE
    elif name == "RMI":
        loss_func = RMI(rmi_radius=3, reduction=K.losses.Reduction.NONE, from_logits=True)
    elif name == "Wasserstein":
        loss_func = WasserSteinLoss(reduction=K.losses.Reduction.NONE)
    else:
        loss_func = K.losses.MAE
    return loss_func


# ================== Region Mutual Information ===================== #

class RMI(K.losses.Loss):
    def __init__(self, pool_stride=3, method=K.layers.MaxPool2D, rmi_radius=3, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.pooling = method(pool_stride, pool_stride,
                              padding="same")  # TODO: Add option to pool using bi-linear interpolation
        self.stride = pool_stride
        self.rmi_radius = rmi_radius
        self.diag_matrix = tf.eye(self.rmi_radius ** 2, dtype=tf.float64)[tf.newaxis, tf.newaxis] * 5e-4
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        assert y_true.shape == y_pred.shape
        y_true, y_pred = tf.cast(y_true, dtype=tf.float64), tf.cast(y_pred, dtype=tf.float64)
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        if self.stride > 1:
            y_true = self.pooling(y_true)
            y_pred = self.pooling(y_pred)
        y_true_vectors, y_pred_vectors = self.extract_pairs(y_pred, y_true)
        y_true_vectors, y_pred_vectors = self.dim_set(y_pred_vectors, y_true_vectors)
        y_true_vectors, y_pred_vectors = self.flatten_features(y_pred_vectors, y_true_vectors)
        y_true_mean, y_pred_mean = tf.reduce_mean(y_true_vectors, axis=-1, keepdims=True), tf.reduce_mean(
            y_pred_vectors, axis=-1, keepdims=True)
        y_true_vectors, y_pred_vectors = y_true_vectors - y_true_mean, y_pred_vectors - y_pred_mean
        y_true_cov, y_pred_cov = tf.linalg.matmul(y_true_vectors, y_true_vectors, transpose_b=True), tf.linalg.matmul(
            y_pred_vectors, y_pred_vectors, transpose_b=True)
        y_pred_cov_inv = tf.linalg.inv(y_pred_cov + self.diag_matrix)
        y_tp_cov = tf.matmul(y_true_vectors, y_pred_vectors, transpose_b=True)
        A_tp = tf.linalg.matmul(y_tp_cov, y_pred_cov_inv)
        est_variance = y_true_cov - tf.linalg.matmul(A_tp, y_tp_cov, transpose_b=True)
        rmi = 0.5 * self.get_log_det_cholesky(est_variance + self.diag_matrix)
        rmi_per_class = rmi / (self.rmi_radius ** 2)
        return rmi_per_class

    def get_log_det_cholesky(self, est_variance):
        return 2. * tf.linalg.trace(tf.math.log(tf.linalg.cholesky(est_variance) + 1e-8))

    def flatten_features(self, y_pred_vectors, y_true_vectors):
        return tf.transpose(
            tf.reshape(y_true_vectors, (y_true_vectors.shape[0], -1, y_true_vectors.shape[3], y_true_vectors.shape[4])),
            perm=[0, 3, 2, 1]), \
               tf.transpose(tf.reshape(y_pred_vectors,
                                       (y_pred_vectors.shape[0], -1, y_pred_vectors.shape[3], y_pred_vectors.shape[4])),
                            perm=[0, 3, 2, 1])

    def dim_set(self, y_pred_vectors, y_true_vectors):
        return tf.reshape(y_true_vectors, (y_true_vectors.shape[0], y_true_vectors.shape[1], y_true_vectors.shape[2],
                                           self.rmi_radius ** 2, y_true_vectors.shape[-1] // self.rmi_radius ** 2)), \
               tf.reshape(y_pred_vectors, (y_true_vectors.shape[0], y_true_vectors.shape[1], y_true_vectors.shape[2],
                                           self.rmi_radius ** 2, y_true_vectors.shape[-1] // self.rmi_radius ** 2))

    def extract_pairs(self, y_pred, y_true):
        return tf.cast(tf.image.extract_patches(y_true,
                                                sizes=[1, self.rmi_radius, self.rmi_radius, 1],
                                                strides=[1, 1, 1, 1],
                                                rates=[1, 1, 1, 1],
                                                padding='VALID'), dtype=tf.float64), \
               tf.cast(tf.image.extract_patches(y_pred,
                                                sizes=[1, self.rmi_radius, self.rmi_radius, 1],
                                                strides=[1, 1, 1, 1],
                                                rates=[1, 1, 1, 1],
                                                padding='VALID'), dtype=tf.float64)


class WasserSteinLoss(K.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        return y_true * y_pred


if __name__ == "__main__":
    from model_provider import get_model

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    disc_model = get_model("cyclegan_disc", type="gan")
    y1 = tf.random.uniform((4, 512, 512, 3))
    y2 = tf.random.uniform((4, 512, 512, 3))
    wloss = WasserSteinLoss()
    w = wloss(y2, y1)
    print(w)
