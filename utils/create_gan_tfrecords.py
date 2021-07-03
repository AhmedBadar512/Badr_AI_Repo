import tensorflow as tf
import pathlib
import os
import cv2
import numpy as np
import tqdm
import argparse


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
    args = argparse.ArgumentParser(description="Create tfrecords with the following settings")
    args.add_argument("-d", "--dataset", type=str, default=f"no_name_{str(np.random.randint(0, 20000))}",
                      help="Name a dataset to be later used with seg_train script, highly recommended to have one")
    args.add_argument("--img_dir", "-i", type=str, required=True, help="Directory containing the dataset images")
    args.add_argument("--save_dir", "-s", type=str, required=True, help="Directory to save the tfrecords")
    args.add_argument("--img_pat", "-i_p", type=str, default="*.jpg", help="Image pattern/extension in directory, "
                                                                           "glob regex convention")
    args.add_argument("--visualize", "-v", action="store_true", help="Show 4 samples after creation. As visual check.")
    args.add_argument("--eval", "-e", action="store_true", help="Set to true in case the records are for evaluation")
    args = args.parse_args()
    dataset_name = args.dataset
    os.makedirs(args.save_dir, exist_ok=True)
    record_type = "train" if not args.eval else "val"
    records = TFRecordsGAN(image_dir=f"{args.img_dir}",
                           tfrecord_path=f"{args.save_dir}/{dataset_name}_{record_type}.tfrecords",
                           img_pattern=args.img_pat)
    records.write_tfrecords(training=True, dataset_name=dataset_name) if not args.eval else records.write_tfrecords()
    if args.visualize:
        image_dataset = records.read_tfrecords().batch(1).take(4)
        cv2.namedWindow("img", 0)
        for image_features in image_dataset:
            img = image_features[0, ..., ::-1]
            cv2.imshow("img", img.numpy())
            cv2.waitKey()
