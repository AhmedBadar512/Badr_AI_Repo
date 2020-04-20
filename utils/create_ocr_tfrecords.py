import tensorflow as tf
import pathlib
import untangle


class TFRecordsOCR:
    def __init__(self, data_dir="C:/Code/tf-crnn/data/iam/", tfrecord_path="data.tfrecords"):
        """
        :param data_dir: the path to iam directory containing the subdirectories of xml and lines from iam dataset
        :param tfrecord_path:
        """
        self.data_dir = data_dir
        self.tfrecord_path = tfrecord_path
        self.labels = []
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

    def iam_xml_to_list(self, xml_paths):
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

    def image_example(self, image_string, label):
        feature = {
            'label': self._bytes_feature(label),
            'image': self._bytes_feature(image_string),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def write_tfrecords(self):
        img_paths = sorted(pathlib.Path(self.data_dir + "lines").rglob("*png"))
        xml_paths = sorted(pathlib.Path(self.data_dir + "xml").rglob("*xml"))
        self.iam_xml_to_list(xml_paths)
        with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
            for img_path, label in zip(img_paths, self.labels):
                img_string = open(str(img_path), 'rb').read()
                tf_example = self.image_example(img_string, label.encode())
                writer.write(tf_example.SerializeToString())

    def decode_strings(self, record):
        images = tf.io.decode_jpeg(record['image'], 3)
        return images, record['label']

    def read_tfrecords(self):
        """
        Read iam tfrecords
        :return: Returns a tuple of images and their label (images, labels)
        """
        raw_dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        parsed_dataset = raw_dataset.map(self._parse_example_function)
        decoded_dataset = parsed_dataset.map(self.decode_strings)
        return decoded_dataset


# example = TFRecordsOCR(data_dir="C:/Code/tf-crnn/data/iam/", tfrecord_path="C:/Code/data.tfrecords")
# example.write_tfrecords()
# image_dataset = example.read_tfrecords().shuffle(10000)
#
# for image_features in image_dataset.take(10):
#     print(image_features[0].shape, image_features[1].numpy())
