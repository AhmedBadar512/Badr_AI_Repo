from collections import namedtuple
import tensorflow as tf


def get_cityscapes():
    Label = namedtuple('Label', ['name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color'])

    labels = [
        # name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        Label('license plate', -1, -1, 'vehicle', 7, False, True, (255, 234, 142)),
    ]
    return labels


def generate_random_colors(n=256, seed=0, bg_class=0):
    cmap_mask = 1 - tf.one_hot(bg_class, depth=256, dtype=tf.int32)[..., tf.newaxis]
    cmp = tf.random.uniform((n, 3), minval=0, maxval=255, dtype=tf.int32, seed=seed) * cmap_mask
    return cmp


def convert_cs_19(segmentation):
    cs_19_map = [tf.where(segmentation == label[1], label[2] + 1, 0)
                 for label in get_cityscapes()
                 if (label[2] != 255 and label[2] != -1)]
    cs_19_map = sum(cs_19_map) - 1
    cs_19_map = tf.cast(cs_19_map, tf.int32)
    return cs_19_map


def gpu_cs_labels(segmentation_maps):
    """
    segmentation_map: (b, h, w, 1) or (b, h, w)
    """
    ncmap = [label[-1] for label in get_cityscapes() if (label[2] != 255 and label[2] != -1)]
    color_imgs = tf.gather(params=ncmap, indices=tf.cast(segmentation_maps, dtype=tf.int32))
    return color_imgs


def gpu_random_labels(segmentation_maps, cmp):
    """
    segmentation_map: (b, h, w, 1) or (b, h, w)
    """
    if len(segmentation_maps.shape) == 4:
        segmentation_maps = segmentation_maps[..., 0]
    color_imgs = tf.gather(params=cmp, indices=tf.cast(segmentation_maps, dtype=tf.int32))
    return color_imgs


if __name__ == "__main__":
    cs_dict = get_cityscapes()
    ncmap = [label[-1] if (label[2] != 255 and label[2] != -1) else (0, 0, 0) for label in cs_dict]
    print("a")
