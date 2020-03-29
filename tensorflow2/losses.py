import tensorflow.keras as K
import tensorflow as tf


def get_loss(logits, labels, name='cross_entropy', from_logits=True):
    if name ==  'cross_entropy':
        loss = K.losses.categorical_crossentropy(labels, logits, from_logits=from_logits)
        return tf.reduce_mean(loss)