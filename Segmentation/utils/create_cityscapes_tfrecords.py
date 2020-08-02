import tensorflow as tf
import pathlib
import os
import cv2
from PIL import Image
import numpy as np
import tensorflow_addons as tfa


class TFRecordsSeg:
    def __init__(self, data_dir="/datasets/custom/cityscapes", tfrecord_path="data.tfrecords", split='train'):
        """
        :param data_dir: the path to iam directory containing the subdirectories of xml and lines from iam dataset
        :param tfrecord_path:
        """
        assert split in ['train', 'val', 'test'], "Split value must be either train, test or val"
        self.data_dir = data_dir
        self.labels_dir = os.path.join(data_dir, "gtFine/{}".format(split))
        self.image_dir = os.path.join(data_dir, "leftImg8bit/{}".format(split))
        self.tfrecord_path = tfrecord_path
        self.labels = []
        self.image_feature_description = \
            {
                'label': tf.io.FixedLenFeature([], tf.string),
                'image': tf.io.FixedLenFeature([], tf.string),
                'instance_labels': tf.io.FixedLenFeature([], tf.string)
            }

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _parse_example_function(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_example(example_proto, self.image_feature_description)

    def image_example(self, image_string, label, instance_label):
        feature = {
            'label': self._bytes_feature(label),
            'image': self._bytes_feature(image_string),
            'instance_labels': self._bytes_feature(instance_label)
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def return_inst_cnts(self, inst_ex):
        inst_cnt = np.zeros(inst_ex.shape)
        for unique_class in np.unique(inst_ex):
            inst_img = (inst_ex == unique_class) / 1
            cnts, _ = cv2.findContours(inst_img.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            inst_cnt = cv2.drawContours(inst_cnt, cnts, -1, (1., 1., 1.), thickness=1)
        return inst_cnt

    def write_tfrecords(self):
        img_paths = sorted(pathlib.Path(self.image_dir).rglob("*leftImg8bit.png"))
        instance_paths = sorted(pathlib.Path(self.labels_dir).rglob("*instanceIds.png"))
        label_paths = sorted(pathlib.Path(self.labels_dir).rglob("*labelIds.png"))
        color_paths = sorted(pathlib.Path(self.labels_dir).rglob("*color.png"))
        with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
            for img_path, label_path, instance_path, color_path in zip(img_paths, label_paths, instance_paths,
                                                                       color_paths):
                img_string = open(str(img_path), 'rb').read()
                label_string = open(str(label_path), 'rb').read()
                instance_string = open(str(instance_path), 'rb').read()
                tf_example = self.image_example(img_string, label_string, instance_string)
                writer.write(tf_example.SerializeToString())

    @staticmethod
    def return_inst_cnts(inst_ex):
        inst_cnt = np.zeros(inst_ex.shape)
        for unique_class in np.unique(inst_ex):
            inst_img = (inst_ex == unique_class) / 1
            cnts, _ = cv2.findContours(inst_img.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            inst_cnt = cv2.drawContours(inst_cnt, cnts, -1, (1., 1., 1.), thickness=2)
        return inst_cnt

    def decode_strings(self, record):
        images = tf.io.decode_jpeg(record['image'], 3)
        labels = tf.io.decode_jpeg(record['label'], 3)
        instance_labels = tf.io.decode_jpeg(record['instance_labels'], 3)
        return images, labels, instance_labels

    def read_tfrecords(self):
        """
        Read iam tfrecords
        :return: Returns a tuple of images and their label (images, labels)
        """
        raw_dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        parsed_dataset = raw_dataset.map(self._parse_example_function)
        decoded_dataset = parsed_dataset.map(self.decode_strings)
        return decoded_dataset


# train = TFRecordsSeg(data_dir="/datasets/custom/cityscapes/", tfrecord_path="/datasets/custom/train.tfrecords", split='train')
# val = TFRecordsSeg(data_dir="/datasets/custom/cityscapes/", tfrecord_path="/datasets/custom/val.tfrecords", split='val')
# train.write_tfrecords()
# val.write_tfrecords()
# example = TFRecordsSeg(data_dir="", tfrecord_path="./test.tfrecords", split='val')
# image_dataset = example.read_tfrecords().repeat(10)
# cv2.namedWindow("img", 0)
# cv2.namedWindow("label", 0)
# cv2.namedWindow("insts", 0)
# for image_features in image_dataset:
#     img = image_features[0][..., ::-1]
#     label = image_features[1]
#     insts = image_features[2]
#     cv2.imshow("img", img.numpy())
#     cv2.imshow("label", label.numpy()/34)
#     cv2.imshow("insts", insts.numpy())
#     cv2.waitKey()
#
#     print(image_features[0].shape, image_features[1].shape, image_features[2].shape)
# example.write_tfrecords()
# image_dataset = example.read_tfrecords().shuffle(10000)
#
# for image_features in image_dataset.take(10):
#     print(image_features[0].shape, image_features[1].numpy())
