import tensorflow as tf
import pathlib
import os
import cv2
import numpy as np
import tqdm
import argparse


class TFRecordsSeg:
    def __init__(self,
                 image_dir="/datasets/custom/cityscapes",
                 label_dir="/datasets/custom/cityscapes",
                 tfrecord_path="data.tfrecords",
                 classes=34,
                 img_pattern="*.png",
                 label_pattern="*.png"):
        """
        :param data_dir: the path to iam directory containing the subdirectories of xml and lines from iam dataset
        :param tfrecord_path:
        """
        # self.data_dir = data_dir
        # self.labels_dir = os.path.join(data_dir, "gtFine/{}".format(split))
        # self.image_dir = os.path.join(data_dir, "leftImg8bit/{}".format(split))
        self.image_dir = image_dir
        self.labels_dir = label_dir
        self.tfrecord_path = tfrecord_path
        self.labels = []
        self.classes = classes
        self.img_pattern = img_pattern
        self.label_pattern = label_pattern
        self.image_feature_description = \
            {
                'label': tf.io.FixedLenFeature([], tf.string),
                'image': tf.io.FixedLenFeature([], tf.string)
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

    def image_example(self, image_string, label):
        feature = {
            'label': self._bytes_feature(label),
            'image': self._bytes_feature(image_string)
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def return_inst_cnts(self, inst_ex):
        inst_cnt = np.zeros(inst_ex.shape)
        for unique_class in np.unique(inst_ex):
            inst_img = (inst_ex == unique_class) / 1
            cnts, _ = cv2.findContours(inst_img.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            inst_cnt = cv2.drawContours(inst_cnt, cnts, -1, (1., 1., 1.), thickness=1)
        return inst_cnt

    def write_tfrecords(self, training=False, dataset_name=""):
        img_paths = sorted(pathlib.Path(self.image_dir).rglob(self.img_pattern))
        label_paths = sorted(pathlib.Path(self.labels_dir).rglob(self.label_pattern))
        with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
            for img_path, label_path in tqdm.tqdm(zip(img_paths, label_paths)):
                img_string = open(str(img_path), 'rb').read()
                label_string = open(str(label_path), 'rb').read()
                tf_example = self.image_example(img_string, label_string)
                writer.write(tf_example.SerializeToString())
            if training:
                import json
                if os.path.exists('{}/data_samples.json'.format(os.path.dirname(self.tfrecord_path))):
                    with open('{}/data_samples.json'.format(os.path.dirname(self.tfrecord_path))) as f:
                        data = json.load(f)
                    if dataset_name in list(data.keys()):
                        print("Dataset {} value was already present but value was updated".format(dataset_name))
                else:
                    data = {}
                data[dataset_name] = len(img_paths)
                with open('{}/data_samples.json'.format(os.path.dirname(self.tfrecord_path)), 'w') as json_file:
                    json.dump(data, json_file)

    def decode_strings(self, record):
        images = tf.io.decode_jpeg(record['image'], 3)
        labels = tf.io.decode_jpeg(record['label'], 3)
        return images, labels

    def read_tfrecords(self):
        """
        Read iam tfrecords
        :return: Returns a tuple of images and their label (images, labels)
        """
        raw_dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        parsed_dataset = raw_dataset.map(self._parse_example_function)
        decoded_dataset = parsed_dataset.map(self.decode_strings)
        return decoded_dataset


if __name__ == "__main__":
    classes = 34
    dataset_name = "cityscapes"
    os.makedirs("/data/input-ai/datasets/tf2_segmentation_tfrecords/", exist_ok=True)
    train = TFRecordsSeg(image_dir="/volumes2/datasets/Generic/cityscapes/leftImg8bit/train",
                         label_dir="/volumes2/datasets/Generic/cityscapes/gtFine/train",
                         tfrecord_path="/data/input-ai/datasets/tf2_segmentation_tfrecords/{}_train.tfrecords".format(dataset_name),
                         classes=classes, img_pattern="*.png",
                         label_pattern="*labelIds.png")
    val = TFRecordsSeg(image_dir="/volumes2/datasets/Generic/cityscapes/leftImg8bit/val",
                       label_dir="/volumes2/datasets/Generic/cityscapes/gtFine/val",
                       tfrecord_path="/data/input-ai/datasets/tf2_segmentation_tfrecords/{}_val.tfrecords".format(dataset_name),
                       classes=classes, img_pattern="*.png",
                       label_pattern="*labelIds.png")
    # train.write_tfrecords(training=True, dataset_name=dataset_name)
    # val.write_tfrecords()


    example = train
    image_dataset = example.read_tfrecords().repeat(10)
    cv2.namedWindow("img", 0)
    cv2.namedWindow("label", 0)
    for image_features in image_dataset:
        img = image_features[0][..., ::-1]
        label = image_features[1]
        print(np.unique(label.numpy()))
        # insts = image_features[2]
        cv2.imshow("img", img.numpy())
        cv2.imshow("label", label.numpy()/classes)
        cv2.waitKey()

        print(image_features[0].shape, image_features[1].shape)
    example.write_tfrecords()
    image_dataset = example.read_tfrecords().shuffle(10000)

    for image_features in image_dataset.take(10):
        print(image_features[0].shape, image_features[1].numpy())
