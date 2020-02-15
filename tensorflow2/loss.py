import tensorflow as tf
import numpy as np


def custom_miou(y_true, y_pred):
    """
    Non sparse IoU calculation from logits
    :param y_true: actual labels
    :param y_pred: predicted labels
    :return:
    """
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    m = tf.keras.metrics.MeanIoU(30)
    m.update_state(y_true, y_pred)
    return y_true, y_pred, m


# x = np.random.rand(1, 4, 4, 2)
# y, _, miou = custom_miou(x, np.random.rand(1, 4, 4, 2))
# print(x.shape, y.shape)
# print(y)
# print(miou.result())