from models.centernet import *

__all__ = ['get_model']


_models = {

    'centernet_resnet18_voc': centernet_resnet18_voc,
    'centernet_resnet18_coco': centernet_resnet18_coco,
    'centernet_resnet50b_voc': centernet_resnet50b_voc,
    'centernet_resnet50b_coco': centernet_resnet50b_coco,
    'centernet_resnet101b_voc': centernet_resnet101b_voc,
    'centernet_resnet101b_coco': centernet_resnet101b_coco,
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
