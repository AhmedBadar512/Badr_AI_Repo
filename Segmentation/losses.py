import tensorflow.keras as K
import tensorflow as tf


def get_loss(name='cross_entropy'):
    if name == 'cross_entropy':
        # loss_func = lambda labels, logits: \
        #     tf.reduce_mean(
        #         K.losses.categorical_crossentropy
        #         (labels, tf.nn.softmax(logits, axis=-1)))
        loss_func = K.losses.CategoricalCrossentropy(from_logits=True, reduction=K.losses.Reduction.NONE)
        return loss_func
