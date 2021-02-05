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
    elif name == "PatchNCELoss":
        loss_func = PatchNCELoss(nce_temp=0.07, nce_lambda=1.0, reduction=K.losses.Reduction.NONE)
    else:
        loss_func = lambda real_image, cycled_image: tf.reduce_mean(tf.abs(real_image - cycled_image))
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


def gradient_penalty(img, f_img, m):
    a = tf.random.uniform((img.shape[0], 1, 1, 1), 0, 1, dtype=tf.float32)
    interpolated_img = a * img + (1 - a) * f_img
    with tf.GradientTape() as tape:
        tape.watch(interpolated_img)
        x = m(interpolated_img)
    grads = tape.gradient(x, [interpolated_img])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    grad_l2 = tf.reduce_mean(tf.square(1 - slopes))
    return grad_l2


class PatchNCELoss(K.losses.Loss):
    def __init__(self, nce_temp=0.07, nce_lambda=1.0, **kwargs):
        # Potential: only supports for batch_size=1 now.
        super().__init__(**kwargs)
        self.nce_temp = nce_temp
        self.nce_lambda = nce_lambda
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(
                                        reduction=tf.keras.losses.Reduction.NONE,
                                        from_logits=True)

    def call(self, source, target, netE, netF):
        feat_source = netE(source, training=True)
        feat_target = netE(target, training=True)

        feat_source_pool, sample_ids = netF(feat_source, patch_ids=None, training=True)
        feat_target_pool, _ = netF(feat_target, patch_ids=sample_ids, training=True)

        total_nce_loss = 0.0
        for feat_s, feat_t in zip(feat_source_pool, feat_target_pool):
            n_patches, dim = feat_s.shape

            logit = tf.matmul(feat_s, tf.transpose(feat_t)) / self.nce_temp

            diagonal = tf.eye(n_patches, dtype=tf.bool)
            target = tf.where(diagonal, 1.0, 0.0)

            loss = self.cross_entropy_loss(target, logit) * self.nce_lambda
            total_nce_loss += tf.reduce_mean(loss)

        return total_nce_loss / len(feat_source_pool)


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
