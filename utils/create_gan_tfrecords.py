import tensorflow as tf
import pathlib
import os
import cv2
import numpy as np
import tqdm

class TFRecordsGAN:
    def __init__(self,
                 image_dir="/volumes2/datasets/horse2zebra/trainA",
                 tfrecord_path="data.tfrecords",
                 img_pattern="*.jpgg"):
        """
        :param data_dir: the path to iam directory containing the subdirectories of xml and lines from iam dataset
        :param tfrecord_path:
        """
        self.image_dir = image_dir
        self.tfrecord_path = tfrecord_path
        self.img_pattern = img_pattern
        self.image_feature_description = \
            {
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

    def image_example(self, image_string):
        feature = {
            'image': self._bytes_feature(image_string)
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def write_tfrecords(self, training=False, dataset_name=""):
        img_paths = sorted(pathlib.Path(self.image_dir).rglob(self.img_pattern))
        with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
            for img_path in tqdm.tqdm(img_paths):
                img_string = open(str(img_path), 'rb').read()
                tf_example = self.image_example(img_string)
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
        return images

    def read_tfrecords(self):
        """
        Read iam tfrecords
        :return: Returns an image
        """
        raw_dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        parsed_dataset = raw_dataset.map(self._parse_example_function)
        decoded_dataset = parsed_dataset.map(self.decode_strings)
        return decoded_dataset


if __name__ == "__main__":
    dataset_name = "zebra2horse_b"
    os.makedirs("/data/input-ai/datasets/tf2_gan_tfrecords", exist_ok=True)
    train = TFRecordsGAN(image_dir="/volumes2/datasets/horse2zebra/trainB/",
                         tfrecord_path="/data/input-ai/datasets/tf2_gan_tfrecords/{}_train.tfrecords".format(dataset_name), img_pattern="*.jpg")
    val = TFRecordsGAN(image_dir="/volumes2/datasets/horse2zebra/testB/",
                       tfrecord_path="/data/input-ai/datasets/tf2_gan_tfrecords/{}_val.tfrecords".format(dataset_name), img_pattern="*.jpg")
    train.write_tfrecords(training=True, dataset_name=dataset_name)
    val.write_tfrecords()
    # image_dataset = train.read_tfrecords().repeat(10).batch(4)
    # cv2.namedWindow("img", 0)
    # for image_features in image_dataset:
    #     img = image_features[0, ..., ::-1]
    #     cv2.imshow("img", img.numpy())
    #     cv2.waitKey()
