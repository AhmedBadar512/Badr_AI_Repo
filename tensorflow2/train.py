#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import visualization_dicts as vis
import numpy as np

ds_train = tfds.load(name="cityscapes", split='train')
ds_train = ds_train.shuffle(100).batch(4)
ds_val = tfds.load(name="cityscapes", split='validation')
ds_val = ds_val.batch(4)


def display(img_list, seg_list, pred_list=None):
    seg_list = vis.gpu_cs_labels(seg_list, False)
    img_list, seg_list = img_list.numpy(), seg_list.numpy()
    new_img = cv2.hconcat(img_list[..., ::-1]) / 255
    new_seg = cv2.hconcat(seg_list[..., ::-1]) / 255
    if pred_list is not None:
        pred_list = vis.gpu_cs_labels(pred_list, False)
        pred_list = pred_list.numpy()
        new_pred = cv2.hconcat(pred_list[..., ::-1]) / 255
        final_img = np.concatenate([new_img, new_seg, new_pred])
    else:
        final_img = np.concatenate([new_img, new_seg])
    print(final_img.shape)
    cv2.namedWindow("My_Window", 0)
    cv2.imshow("My_Window", final_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_images(features, shp=(256, 512)):
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
    return image, label


get_images_new = lambda features: get_images(features, (512, 1024))

ds_train_new = ds_train.map(get_images_new).repeat()
ds_val_new = ds_val.map(get_images)

for features in ds_train_new.take(1):
    image, segmentation = features

display(image, segmentation)
