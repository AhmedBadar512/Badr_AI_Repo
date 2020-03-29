import tensorflow as tf


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
