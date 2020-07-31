from .unet import build_model as unet_model
from .bisenet import bisenet_resnet18
MODELS = ["unet", "bisenet"]


def get_model(name="unet", in_channels=3, n_classes=19, shp=(512, 1024)):
    assert name in ["unet", "bisenet"], "Model {} not available, please select from {}".format(name, MODELS)
    if name == "unet":
        if shp is not None:
            return unet_model(nx=shp[0], ny=shp[1], channels=in_channels, num_classes=n_classes)
        else:
            return unet_model(channels=in_channels, num_classes=n_classes)
    if name == "bisenet":
        return bisenet_resnet18(in_size=shp, classes=n_classes, aux=False)
