"""
This Script is for GAN based trainings.
"""
import argparse
import datetime
import json
import os
import string

import cv2
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_datasets as tfds
import tqdm
import matplotlib.pyplot as plt

import losses
import utils.augment_images as aug
from seg_visualizer import get_images_custom
from model_provider import get_model
from utils.create_seg_tfrecords import TFRecordsSeg
from visualization_dicts import generate_random_colors, gpu_random_labels

physical_devices = tf.config.experimental.list_physical_devices("GPU")

if len(physical_devices) > 1:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    mirrored_strategy = tf.distribute.MirroredStrategy()
else:
    mirrored_strategy = tf.distribute.MirroredStrategy()

x_train = tfds.load("celeb_a", split='train', data_dir="tmp").batch(32)
x_train = mirrored_strategy.experimental_distribute_dataset(x_train)
x_test = tfds.load("celeb_a", split='test', data_dir="tmp").batch(1)


calc_loss = K.losses.BinaryCrossentropy(reduction=K.losses.Reduction.NONE)


def train_step(mini_batch, pick=None):
    img = tf.cast(mini_batch['image'], dtype=tf.float32)
    img = tf.image.resize(img, size=(64, 64))
    with tf.GradientTape() as tape:
        output = model(img, training=True)
        loss = calc_loss(img/255, output)
        loss = tf.reduce_mean(loss)
    if pick is not None:
        trainable_vars = [var for var in model.trainable_variables if pick in var.name]
    else:
        trainable_vars = model.trainable_variables
    grads = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))
    return loss


def distributed_train_step(dist_inputs):
    per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
    loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,
                                    axis=None)
    return loss


with mirrored_strategy.scope():
    model = get_model("attention_gan")
    optimizer = K.optimizers.Adam(learning_rate=1e-6)
img, new_img = None, None
loss = 10
cv2.namedWindow("img", 0)
cv2.namedWindow("syn_img", 0)
for epoch in range(999999):
    for img in tqdm.tqdm(x_train):
        loss = distributed_train_step(img).numpy()
        print("Loss: {}".format(loss))

        img = tf.cast(img['image'].values[0], dtype=tf.float32)
        img = tf.image.resize(img, size=(64, 64))
        new_img = model(img)
        img = [im for im in img]
        new_img = [im for im in new_img]
        img = tf.concat(img[:4], axis=0)
        new_img = tf.concat(new_img[:4], axis=0).numpy()
        new_img /= new_img.max()
        cv2.imshow("img", img.numpy()[..., ::-1]/255)
        cv2.imshow("syn_img", new_img[..., ::-1])
        cv2.waitKey(100)
        # plt.subplot(1, 2, 1)
        # plt.imshow(img.numpy())
        # plt.subplot(1, 2, 2)
        # plt.imshow(new_img.numpy())
        # plt.show()
