import tensorflow as tf
import numpy as np


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


#  Each return value from above can be serialized to protobuf
# feature = _float_feature(np.exp(1))
# feature.SerializeToString()


def serialize_example(feature0, feature1, feature2, feature3):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }

    # Create a Features message using tf.train.Example.

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


serialized_example = serialize_example(False, 4, b'goat', 0.9876)  # Encoded example
example_proto = tf.train.Example.FromString(serialized_example)  # Decoded example
"""Note: There is no requirement to use tf.Example in TFRecord files. tf.Example is just a method of serializing 
dictionaries to byte-strings. Lines of text, encoded image data, or serialized tensors (using tf.io.serialize_tensor, 
and tf.io.parse_tensor when loading). See the tf.io module for more options. """
"""=================================== Data =============================================="""
n_observations = int(1e4)
feature0 = np.random.choice([False, True], n_observations)
feature1 = np.random.randint(0, 5, n_observations)
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]
feature3 = np.random.randn(n_observations)

filename = 'test.tfrecord'

with tf.io.TFRecordWriter(filename) as writer:
    for i in range(n_observations):
        example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
        writer.write(example)

""" Reading TFRecords """

filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)

example = tf.train.Example()
for raw_record in raw_dataset.take(2):
    example.ParseFromString(raw_record.numpy())
    print(example.features.feature['feature0'], example.features.feature['feature1'],
          example.features.feature['feature2'], example.features.feature['feature3'])

#  TODO: Create a class that automatically maps values to tf.dataset

# Image data writing
cat_in_snow = tf.keras.utils.get_file('C:/Code/Badr_AI_Repo/utils/320px-Felis_catus-cat_on_snow.jpg',
                                      'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
williamsburg_bridge = tf.keras.utils.get_file(
    'C:/Code/Badr_AI_Repo/utils/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')

image_labels = {
    cat_in_snow: 0,
    williamsburg_bridge: 1,
}

image_string = open(cat_in_snow, 'rb').read()

label = image_labels[cat_in_snow]


# Create a dictionary with features that may be relevant.
def image_example(image_string, label):
    image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_example(example_proto, image_feature_description)


parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

for image_features in parsed_image_dataset:
    image_raw = image_features['image_raw'].numpy()  # the image as a tensor
    x = tf.io.decode_jpeg(image_raw, 3)
    print(x.shape)