from models.sinet import *
from models.bisenet import *
from models.unet import *
from models.faceswap_gan import *

__all__ = ['get_model']


_models = {
    'faceswap': FaceSwap,
    'unet': UNet,
    'vunet': VUNet,
    'sinet_nie': get_nie_sinet,
    'bisenet_resnet18_celebamaskhq': bisenet_resnet18_celebamaskhq,
}


def get_model(name, **kwargs):
    """
    Get supported model.

    Parameters:
    ----------
    name : str
        Name of model.

    Returns
    -------
    Module
        Resulted model.
    """
    name = name.lower()
    if name not in _models:
        raise ValueError("Unsupported model: {}".format(name))
    net = _models[name](**kwargs)
    return net
