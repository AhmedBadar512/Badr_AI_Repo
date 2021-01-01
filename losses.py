import tensorflow.keras as K
import tensorflow_addons as tfa


def get_loss(name='cross_entropy'):
    if name == 'cross_entropy':
        loss_func = K.losses.CategoricalCrossentropy(from_logits=True, reduction=K.losses.Reduction.NONE)
    elif name == 'focal_loss':
        loss_func = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=K.losses.Reduction.NONE)
    elif name == 'binary_crossentropy':
        loss_func = K.losses.BinaryCrossentropy(from_logits=True, reduction=K.losses.Reduction.NONE)
    elif name == "MSE":
        loss_func = K.losses.MSE
    else:
        loss_func = K.losses.MAE
    return loss_func
