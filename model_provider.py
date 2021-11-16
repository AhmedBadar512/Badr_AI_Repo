from models.sinet import *
from models.bisenet import *
from models.unet import *
from models.deeplabv3 import *
from models.unet_expanded import *
from models.faceswap_gan import *
from models.cycle_gan import *
from models.cut import *
from models.gaugan import GAUGenerator, GAUDiscriminator, GAUEncoder
from models.oasis import OASISGenerator, OASISDiscriminator

__all__ = ['get_model']


_models = {
    'unet': UNet,
    'unet_exp': UNet_Expanded,
    'sinet_nie': get_nie_sinet,
    'bisenet_resnet18_celebamaskhq': bisenet_resnet18_celebamaskhq,
    'deeplabv3': Deeplabv3plus,
}

_ganmodels = {
    'cyclegan_gen': CycleGANGenerator,
    'cyclegan_disc': CycleGANDiscriminator,
    'faceswap_gen': FaceSwapGenerator,
    'faceswap_disc': FaceSwapDiscriminator,
    'cut_gen': CUTGenerator,
    'cut_disc': CUTDiscriminator,
    'cut_enc': CUTEncoder,
    'cut_mlp': PatchSampleMLP,
    'gaugan_gen': GAUGenerator,
    'gaugan_disc': GAUDiscriminator,
    'gaugan_enc': GAUEncoder,
    'oasis_gen': OASISGenerator,
    'oasis_disc': OASISDiscriminator,
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
