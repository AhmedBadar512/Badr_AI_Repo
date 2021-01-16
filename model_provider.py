from models.sinet import *
from models.bisenet import *
from models.unet import *
from models.faceswap_gan import *

__all__ = ['get_model']


_models = {
    'unet': UNet,
    'vunet': VUNet,
    'sinet_nie': get_nie_sinet,
    'bisenet_resnet18_celebamaskhq': bisenet_resnet18_celebamaskhq,
}

_ganmodels = {
    'faceswap_gen': FaceSwapGenerator,
    'faceswap_disc': FaceSwapDiscriminator,
}


def get_model(name, type="seg", **kwargs):
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
    if name not in _models and name not in _ganmodels:
        raise ValueError("Unsupported model: {}".format(name))
    if "seg" in type.lower():
        net = _models[name](**kwargs)
    elif "gan" in type.lower():
        net = _ganmodels[name](**kwargs)
    else:
        raise TypeError("{} type of networks do not exist".format(type))
    return net
