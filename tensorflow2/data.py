import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2


class ClassificationDataExtractor():
    def __init__(self, dataset_name,
                 random_crop=None,
                 random_flip=False,
                 random_brightness=False,
                 random_resize=False,
                 batch_size=1):
        """

        :param dataset_name: Name of the classification dataset to download e.g. cifar10, cifar100, imagenet from tfds.
        :param random_crop: bool value to randomly crop images
        :param random_flip: bool value to randomly flip images
        :param random_brightness: bool value to randomly change brightness
        :param random_resize: bool value to ranodomly resize the images
        :param batch_size: mini batch size as integer for the dataset
        """
        self.dataset_name = dataset_name
        self.random_crop = random_crop
        self.random_brightness = random_brightness
        self.batch_size = batch_size
        self.random_flip = random_flip
        self.random_resize =random_resize

    def pre_process_data(self,
                         in_dataset):
        """
        Pre processing mapping function
        :param in_dataset: the tf.dataset object as input
        :return:
        """
        image = in_dataset["image"]/255
        label = in_dataset["label"]
        if self.random_brightness:
            image = tf.image.random_brightness(image, 0.5)
        if self.random_flip:
            image = tf.image.random_flip_left_right(image)
        if self.random_crop is not None:
            image = tf.image.resize(image, (56, 56))
            image = tf.image.random_crop(image, (self.batch_size, 28, 28, 1))
        if self.random_resize:
            rand_val = tf.random.uniform(shape=[], minval=1, maxval=5, dtype=tf.int32)
            image = tf.image.resize(image, (rand_val * image.shape[1], rand_val * image.shape[2]))
        return image, label

    def get_processed_data(self):
        data_train = tfds.load(self.dataset_name, split='train')
        data_train = data_train.shuffle(1024).batch(self.batch_size)
        proc_data_train = data_train.map(self.pre_process_data)
        return proc_data_train


def visualize_processed_data(current_data):
    """
    Visualize the classification dataset
    :param current_data: input tf.Dataset object
    :return:
    """
    for x, y in current_data:
        # print(x.numpy().shape)
        cv2.imshow("Image", x[0].numpy())
        print(y[0])
        cv2.waitKey(5000)
    cv2.destroyAllWindows()


# test_data = ClassificationDataExtractor("cifar10", batch_size=8, random_resize=True).get_processed_data()
# visualize_processed_data(test_data)