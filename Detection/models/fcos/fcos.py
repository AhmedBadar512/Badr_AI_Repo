from fcos_backbone import get_backbone_outputs, P6P7_level, LastLevelMaxPool
from fcos_rpn import FCOSModule

if __name__ == "__main__":
    n_classes = 10
    backbone_outputs = get_backbone_outputs()
    fcos_model = FCOSModule(n_classes)(backbone_outputs, training=True)
