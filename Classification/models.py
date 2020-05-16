from hardnet import *
from resnet import *
from probs_resnet import *


def get_model(name, **kwargs):
    if name == 'HarDNet':
        return HarDNet(**kwargs)
    if name == "resnet18":
        return resnet_18(**kwargs)
    if name == "resnet20":
        return resnet_20(**kwargs)
    if name == "resnet32":
        return resnet_32(**kwargs)
    if name == "resnet44":
        return resnet_44(**kwargs)
    if name == "resnet56":
        return resnet_56(**kwargs)
    if name == "resnet34":
        return resnet_34(**kwargs)
    if name == "resnet50":
        return resnet_50(**kwargs)
    if name == "resnet101":
        return resnet_101(**kwargs)
    if name == "resnet152":
        return resnet_152(**kwargs)
    if name == "pr_resnet20":
        return pr_resnet_20(**kwargs)
    if name == "pr_resnet56":
        return pr_resnet_56(**kwargs)