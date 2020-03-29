from hardnet import *


def get_model(name, **kwargs):
    if name == 'HarDNet':
        return HarDNet(**kwargs)