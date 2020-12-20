from collections import namedtuple
import tensorflow as tf


def get_cityscapes():
    # --------------------------------------------------------------------------------
    # Definitions
    # --------------------------------------------------------------------------------

    # a label and all meta information
    Label = namedtuple('Label', [

        'name',  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class

        'id',  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.

        'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!

        'category',  # The name of the category that this label belongs to

        'categoryId',  # The ID of this category. Used to create ground truth images
        # on category level.

        'hasInstances',  # Whether this label distinguishes between single instances or not

        'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not

        'color',  # The color of this label
    ])

    # --------------------------------------------------------------------------------
    # A list of all labels
    # --------------------------------------------------------------------------------

    # Please adapt the train IDs as appropriate for your approach.
    # Note that you might want to ignore labels with ID 255 during training.
    # Further note that the current train IDs are only a suggestion. You can use whatever you like.
    # Make sure to provide your results using the original IDs and not the training IDs.
    # Note that many IDs are ignored in evaluation and thus you never need to predict these!

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


def generate_random_colors(n=256):
    cmp = tf.random.uniform((n, 3), minval=0, maxval=255, dtype=tf.int32, seed=0)
    return cmp


def convert_cs_19(segmentation):
    cs_dict = get_cityscapes()
    cs_19_map = [tf.where(segmentation == label[1], label[2] + 1, 0)
                 for label in cs_dict
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
    # new_imgs = []
    # for segmentation_map in segmentation_maps:
    #     new_img = tf.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), tf.uint8)
    #     color_named_tuple = get_cityscapes()
    #     for label in color_named_tuple:
    #         if with_train_ids:
    #             if label[2] == 255 or label[2] == -1:
    #                 continue
    #             tmp = [segmentation_map == label[2]] * 3
    #         else:
    #             if label[2] == 255:
    #                 continue
    #             tmp = [segmentation_map == label[1]] * 3
    #         tmp = tf.cast(tf.squeeze(tf.stack(tmp, axis=-1)), tf.uint8)
    #         new_img = new_img + tmp * label[-1]
    #     new_imgs.append(new_img)
    # return tf.stack(new_imgs)


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
