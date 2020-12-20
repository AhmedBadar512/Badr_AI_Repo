#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import cv2
import visualization_dicts as vis
import numpy as np
import utils.augment_images as aug
import os

tf.random.set_seed(0)


def display(img_list, seg_list, pred_list=None, cs_19=False, save_dir=None, img_path=None, cmap=None):
    new_seg_tmp = cv2.hconcat(seg_list.numpy())
    if cs_19:
        seg_list = vis.gpu_cs_labels(seg_list)
    else:
        seg_list = vis.gpu_random_labels(seg_list, cmp=cmap)
    seg_list = tf.squeeze(seg_list)
    img_list, seg_list = img_list.numpy(), seg_list.numpy()
    new_img = cv2.hconcat(img_list[..., ::-1])
    new_seg = cv2.hconcat(seg_list[..., ::-1])
    if pred_list is not None:
        pred_list = vis.gpu_cs_labels(pred_list)
        pred_list = pred_list.numpy()
        new_pred = cv2.hconcat(pred_list[..., ::-1]) / 255
        final_img = np.concatenate([new_img, new_seg, new_pred])
    else:
        final_img = np.concatenate([new_img, new_seg])
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if img_path is None:
            cv2.imwrite("{}/test_{}.jpg".format(save_dir, np.random.randint(0, 10000)), final_img)
        else:
            cv2.imwrite("{}/{}.jpg".format(save_dir, os.path.basename(img_path)), final_img)
    else:
        cv2.namedWindow("My_Window", 0)
        cv2.namedWindow("test", 0)
        cv2.imshow("My_Window", final_img/255)
        cv2.imshow("test", new_seg_tmp)
        cv2.waitKey()


def get_images_custom(image, label, shp=(256, 512), cs_19=False):
    """
    Get images from segmentation tfrecords
    :param features: the tf.data.Dataset object loaded with tf_datasets
    :param shp: reshape the images to this tuple
    :return:
    """
    if shp is not None:
        image = tf.image.resize(image, shp)
        label = tf.image.resize(label, shp, method='nearest')
    if cs_19:
        label = tf.cast(label, dtype=tf.int32)
        label = tf.where(label == 6, 7, label)
        label = tf.where(label == 9, 8, label)
        label = tf.where(label == 4, 11, label)
        label = tf.where(label == 5, 11, label)
        label = vis.convert_cs_19(label)
    return image, label[..., 0:1]


if __name__ == "__main__":
    from utils.create_seg_tfrecords import TFRecordsSeg
    ds_train = TFRecordsSeg(tfrecord_path="/data/input/datasets/tf2_segmentation_tfrecords/cityscapes_train.tfrecords").read_tfrecords()
    ds_val = TFRecordsSeg(tfrecord_path="/data/input/datasets/tf2_segmentation_tfrecords/cityscapes_val.tfrecords").read_tfrecords()
    augmentor = lambda image, label: aug.augment(image, label, True, False, None, True, True, True, True, True)
    ds_train = ds_train.map(augmentor)
    ds_train = ds_train.batch(4)
    ds_val = ds_val.batch(4)
    cs_19 = False
    process = lambda image, label: get_images_custom(image, label, shp=(256, 512), cs_19=cs_19)

    ds_train_new = ds_train.map(process).repeat()
    cmap = vis.generate_random_colors()
    for image, segmentation in ds_train_new:
        display(image, segmentation, cs_19=cs_19, save_dir=None, cmap=cmap)
    cv2.destroyAllWindows()
