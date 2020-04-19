import tensorflow as tf
import numpy as np
import pathlib
import untangle
import cv2

# def _bytes_feature(value):
#     """Returns a bytes_list from a string / byte."""
#     if isinstance(value, type(tf.constant(0))):
#         value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#
# def _float_feature(value):
#     """Returns a float_list from a float / double."""
#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
#
#
# def _int64_feature(value):
#     """Returns an int64_list from a bool / enum / int / uint."""
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
#
# def image_example(image_string, label):
#     image_shape = tf.image.decode_jpeg(image_string).shape
#
#     feature = {
#         'height': _int64_feature(image_shape[0]),
#         'width': _int64_feature(image_shape[1]),
#         'depth': _int64_feature(image_shape[2]),
#         'label': _int64_feature(label),
#         'image_raw': _bytes_feature(image_string),
#     }
#
#     return tf.train.Example(features=tf.train.Features(feature=feature))


#  TODO: Create a class that automatically maps values to tf.dataset
class TFRecordsOCR:
    def __init__(self, save_dir="C:/Code/tf-crnn/data/iam/"):
        self.save_dir = save_dir
        self.labels = []

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

    def xmltolist(self, xml_paths):
        """
        Convert iam dataset xmls to list of strings
        :param xml_paths:
        :return:
        """
        for xml_path in xml_paths:
            # print(xml_path)
            xml_file = untangle.parse(str(xml_path))
            lines = [line['text'] for line in xml_file.form.handwritten_part.line]
            self.labels += lines

    def collect_iam_line_list(self):
        img_paths = sorted(pathlib.Path(self.save_dir + "lines").rglob("*png"))
        xml_paths = sorted(pathlib.Path(self.save_dir + "xml").rglob("*xml"))
        self.xmltolist(xml_paths)
        self.labels = self.labels[:len(img_paths)]
        # print(len(img_paths), len(self.labels))
        # cv2.namedWindow("test", 0)
        # for img_path, label in zip(img_paths, self.labels):
        #     # image_string = open(img_path, 'rb').read()
        #     # print(img_path)
        #     img = cv2.imread(str(img_path))
        #     try:
        #         cv2.imshow("test", img)
        #         print(label)
        #         cv2.waitKey(1)
        #     except:
        #         print(img_path)
        #         print(label)
        #         cv2.waitKey()
        #         continue


example = TFRecordsOCR()
# # Image data writing
# cat_in_snow = tf.keras.utils.get_file('C:/Code/Badr_AI_Repo/utils/320px-Felis_catus-cat_on_snow.jpg',
#                                       'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
# williamsburg_bridge = tf.keras.utils.get_file(
#     'C:/Code/Badr_AI_Repo/utils/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
#     'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')
#
# image_labels = {
#     cat_in_snow: 0,
#     williamsburg_bridge: 1,
# }
#
# image_string = open(cat_in_snow, 'rb').read()
#
# label = image_labels[cat_in_snow]
#
# # Create a dictionary with features that may be relevant.
#
# record_file = 'images.tfrecords'
# with tf.io.TFRecordWriter(record_file) as writer:
#     for filename, label in image_labels.items():
#         image_string = open(filename, 'rb').read()
#         tf_example = image_example(image_string, label)
#         writer.write(tf_example.SerializeToString())
#
# raw_image_dataset = tf.data.TFRecordDataset(record_file)
#
# # Create a dictionary describing the features.
# image_feature_description = {
#     # 'height': tf.io.FixedLenFeature([], tf.int64),
#     # 'width': tf.io.FixedLenFeature([], tf.int64),
#     # 'depth': tf.io.FixedLenFeature([], tf.int64),
#     # 'label': tf.io.FixedLenFeature([], tf.int64),
#     'image_raw': tf.io.FixedLenFeature([], tf.string),
# }
#
#
# def _parse_image_function(example_proto):
#     # Parse the input tf.Example proto using the dictionary above.
#     return tf.io.parse_example(example_proto, image_feature_description)
#
#
# parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
#
# for image_features in parsed_image_dataset:
#     image_raw = image_features['image_raw'].numpy()  # the image as a tensor
#     x = tf.io.decode_jpeg(image_raw, 3)
#     print(x.shape)
