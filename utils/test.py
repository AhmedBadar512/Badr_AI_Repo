import cv2
import pathlib
import numpy as np
import os
import tensorflow_addons as tfa
import tensorflow as tf


def return_inst_cnts(inst_ex):
    inst_cnt = np.zeros(inst_ex.shape)
    for unique_class in np.unique(inst_ex):
        inst_img = (inst_ex == unique_class) / 1
        cnts, _ = cv2.findContours(inst_img.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        inst_cnt = cv2.drawContours(inst_cnt, cnts, -1, (1., 1., 1.), thickness=1)
    return inst_cnt


data_dir = "/datasets/custom/cityscapes"
split = 'val'
labels_dir = os.path.join(data_dir, "gtFine/{}".format(split))
image_dir = os.path.join(data_dir, "leftImg8bit/{}".format(split))
img_paths = sorted(pathlib.Path(image_dir).rglob("*leftImg8bit.png"))
instance_paths = sorted(pathlib.Path(labels_dir).rglob("*instanceIds.png"))
label_paths = sorted(pathlib.Path(labels_dir).rglob("*labelIds.png"))
color_paths = sorted(pathlib.Path(labels_dir).rglob("*color.png"))
cv2.namedWindow("label", 0)
cv2.namedWindow("final_blobs", 0)
cv2.namedWindow("inst_cnts", 0)
cv2.namedWindow("bnd_processed", 0)
for img_path, label_path, instance_path, color_path in zip(img_paths, label_paths, instance_paths, color_paths):
    inst_ex = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
    id_label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    color_label = cv2.imread(str(color_path), cv2.IMREAD_UNCHANGED)
    onehot_label = tf.one_hot(id_label, depth=31, axis=-1)
    for x in range(onehot_label.shape[-1]):
        print(onehot_label[..., 0].shape)
    cv2.imshow("label", color_label)
    final_blobs = tfa.image.connected_components(id_label)
    inst_cnt = return_inst_cnts(inst_ex)
    print(np.unique(tf.reshape(final_blobs, (-1))))
    visual = final_blobs / tf.math.reduce_max(final_blobs)
    cv2.imshow("final_blobs", visual.numpy())
    cv2.imshow("inst_cnts", inst_cnt/1)
    cv2.imshow("bnd_processed", (1 - inst_cnt) * id_label/31)
    cv2.waitKey()