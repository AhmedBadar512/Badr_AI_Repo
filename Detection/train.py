import tensorflow.keras as K
import tensorflow_datasets as tfds
import argparse
import os
import tensorflow as tf
import cv2
import numpy as np
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


args = argparse.ArgumentParser(description="Train a network with specific settings")
args.add_argument("--dataset", type=str, default="cifar10", help="Name a dataset from the tf_dataset collection")
args.add_argument("--n_classes", type=int, default=100, help="Number of classes")
args.add_argument("--optimizer", type=str, default="Adam", help="Select optimizer", choices=["SGD", "RMSProp", "Adam"])
args.add_argument("--epochs", type=int, default=40, help="Number of epochs to train")
args.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
args.add_argument("--momentum", type=float, default=0.9, help="Momentum")
args.add_argument("--logging_freq", type=int, default=5, help="Add to tfrecords after this many steps")
args.add_argument("--batch_size", type=int, default=16, help="Size of mini-batch")
args.add_argument("--save_interval", type=int, default=1, help="Save interval for model")
args.add_argument("--model", type=str, default="pr_resnet56", help="Select model")
args.add_argument("--save_dir", type=str, default="./runs", help="Save directory for models and tensorboard")
parsed = args.parse_args()

dataset_name = parsed.dataset
epochs = parsed.epochs
batch_size = parsed.batch_size
n_classes = parsed.n_classes
optimizer_name = parsed.optimizer
lr = parsed.lr
momentum = parsed.momentum
model_name = parsed.model
log_freq = parsed.logging_freq
logdir = os.path.join(parsed.save_dir, "logs/{}_epochs-{}_bs-{}_{}_lr-{}_{}".format(dataset_name, epochs, batch_size,
                                                                        optimizer_name, lr, model_name))

dataset = tfds.load("voc/2012", data_dir="/datasets/")
dataset_train = dataset['train']
