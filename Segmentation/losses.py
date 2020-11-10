import tensorflow.keras as K
import tensorflow as tf


def get_loss(name='cross_entropy', from_logits=True):
    if name == 'cross_entropy':
        loss_func = lambda labels, logits: \
            tf.reduce_mean(
                K.losses.categorical_crossentropy
                (labels, tf.nn.softmax(logits, axis=-1), from_logits=from_logits))
        return loss_func