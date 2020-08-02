#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import visualization_dicts as vis
import numpy as np


def display(img_list, seg_list, pred_list=None, cs_19=True):
    seg_list = vis.gpu_cs_labels(seg_list, cs_19)
    img_list, seg_list = img_list.numpy(), seg_list.numpy()
    new_img = cv2.hconcat(img_list[..., ::-1]) / 255
    new_seg = cv2.hconcat(seg_list[..., ::-1]) / 255
    if pred_list is not None:
        pred_list = vis.gpu_cs_labels(pred_list, cs_19)
        pred_list = pred_list.numpy()
        new_pred = cv2.hconcat(pred_list[..., ::-1]) / 255
        final_img = np.concatenate([new_img, new_seg, new_pred])
    else:
        final_img = np.concatenate([new_img, new_seg])
    print(final_img.shape)
    cv2.namedWindow("My_Window", 0)
    cv2.imshow("My_Window", final_img)
    cv2.waitKey(3000)


def get_images(features, shp=(256, 512), cs_19=True):
    """
    Get images from segmentation tfrecords
    :param features: the tf.data.Dataset object loaded with tf_datasets
    :param shp: reshape the images to this tuple
    :return:
    """
    if shp is not None:
        image = tf.image.resize(features["image_left"], shp)
        label = tf.image.resize(features["segmentation_label"], shp, method='nearest')
    else:
        image, label = features["image_left"], features["segmentation_label"]
    if cs_19:
        label = vis.convert_cs_19(label)
    return image, label


def get_images_custom(features, shp=(256, 512), cs_19=True):
    """
    Get images from segmentation tfrecords
    :param features: the tf.data.Dataset object loaded with tf_datasets
    :param shp: reshape the images to this tuple
    :return:
    """
    if shp is not None:
        image = tf.image.resize(features[0], shp)
        label = tf.image.resize(features[1], shp, method='nearest')
    else:
        image, label = features[0], features[1]
    if cs_19:
        label = vis.convert_cs_19(label)
    return image, label[..., 0:1]


if __name__ == "__main__":
    from utils.create_cityscapes_tfrecords import TFRecordsSeg
    # ds_train = tfds.load(name="cityscapes", split='train', data_dir="/datasets/")
    custom = False
    if custom:
        ds_train = TFRecordsSeg(data_dir="/datasets/custom/cityscapes/", tfrecord_path="/datasets/custom/train.tfrecords",
                                split='train').read_tfrecords()
        ds_val = TFRecordsSeg(data_dir="/datasets/custom/cityscapes/", tfrecord_path="/datasets/custom/val.tfrecords",
                                split='val').read_tfrecords()
    else:
        ds_train = tfds.load(name="cityscapes", split='train', data_dir="/datasets/")
        ds_val = tfds.load(name="cityscapes", split='validation', data_dir="/datasets/")
    # ds_val = tfds.load(name="cityscapes", split='validation', data_dir="/datasets/")
    ds_train = ds_train.shuffle(100).batch(4)
    ds_val = ds_val.batch(4)
    cs_19 = True
    # process = lambda features: get_images_custom(features, shp=(512, 1024), cs_19=cs_19)
    if custom:
        process = lambda *features: get_images_custom(features, shp=None, cs_19=cs_19)
    else:
        process = lambda features: get_images(features, shp=None, cs_19=cs_19)

    ds_train_new = ds_train.map(process).repeat()
    # ds_val_new = ds_val.map(get_images_custom)

    for features in ds_train_new:
        image, segmentation = features
        # segmentation = tf.one_hot(segmentation[..., 0], depth=19)
        # segmentation_19 = vis.convert_cs_19(segmentation)
        print(image.shape, segmentation.shape)
        display(image, segmentation, cs_19=cs_19)
    cv2.destroyAllWindows()
