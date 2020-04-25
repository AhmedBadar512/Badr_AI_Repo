from hardnet import *
from resnet import *


def get_model(name, **kwargs):
    if name == 'HarDNet':
        return HarDNet(**kwargs)
    if name == "resnet18":
        return resnet_18(**kwargs)
    if name == "resnet34":
        return resnet_34(**kwargs)
    if name == "resnet50":
        return resnet_50(**kwargs)
    if name == "resnet101":
        return resnet_101(**kwargs)
    if name == "resnet152":
        return resnet_152(**kwargs)
