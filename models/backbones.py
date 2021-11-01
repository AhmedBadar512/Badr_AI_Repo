import tensorflow as tf


def get_backbone(name="resnet50", **kwargs):
    if name.lower() == "resnet50":
        return tf.keras.applications.ResNet50(include_top=False, weights='imagenet', **kwargs)
    elif name.lower() == "resnet50v2":
        return tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', **kwargs)
    elif name.lower() == "resnet101":
        return tf.keras.applications.ResNet101(include_top=False, weights='imagenet', **kwargs)
    elif name.lower() == "resnet101v2":
        return tf.keras.applications.ResNet101V2(include_top=False, weights='imagenet', **kwargs)
    elif name.lower() == "resnet152":
        return tf.keras.applications.ResNet152(include_top=False, weights='imagenet', **kwargs)
    elif name.lower() == "xception":
        return tf.keras.applications.Xception(include_top=False, weights='imagenet', **kwargs)
    else:
        raise ModuleNotFoundError(f'Model "{name}" not yet implemented')
