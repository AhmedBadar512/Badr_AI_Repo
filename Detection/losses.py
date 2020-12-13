import tensorflow.keras as K
import tensorflow as tf
# TODO: Adapt for detection

def get_loss(name='cross_entropy'):
    if name == 'cross_entropy':
        loss_func = lambda labels, logits: \
            tf.reduce_mean(
                K.losses.categorical_crossentropy
                (labels, tf.nn.softmax(logits, axis=-1)))
        return loss_func