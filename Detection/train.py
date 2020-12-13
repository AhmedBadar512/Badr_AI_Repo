import tensorflow.keras as K
import losses
import argparse
import os
import horovod.tensorflow as hvd
import tensorflow as tf
import cv2
import numpy as np
from model_provider import get_model
import utils.augment_images as aug
import tensorflow_datasets as tfds
import datetime
import tqdm
import string

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
hvd.init()
physical_devices = tf.config.experimental.list_physical_devices("GPU")
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)
print("Physical_Devices: {}".format(physical_devices))
if physical_devices:
    tf.config.experimental.set_visible_devices(physical_devices[hvd.local_rank()], 'GPU')

args = argparse.ArgumentParser(description="Train a network with specific settings")
args.add_argument("-d", "--dataset", type=str, default="wider_face",
                  help="Name a dataset from the tf_dataset collection",
                  choices=["wider_face"])
args.add_argument("-c", "--classes", type=int, default=19, help="Number of classes")
args.add_argument("-opt", "--optimizer", type=str, default="Adam", help="Select optimizer",
                  choices=["SGD", "RMSProp", "Adam"])
args.add_argument("-lrs", "--lr_scheduler", type=str, default="exp_decay", help="Select learning rate scheduler",
                  choices=["poly", "exp_decay"])
args.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs to train")
args.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
args.add_argument("--momentum", type=float, default=0.9, help="Momentum")
args.add_argument("-l", "--logging_freq", type=int, default=50, help="Add to tfrecords after this many steps")
args.add_argument("-bs", "--batch_size", type=int, default=4, help="Size of mini-batch")
args.add_argument("-si", "--save_interval", type=int, default=5, help="Save interval for model")
args.add_argument("-wis", "--write_image_summary_steps", type=int, default=5, help="Add images to tfrecords "
                                                                                   "after these many logging steps")
args.add_argument("-m", "--model", type=str, default="bisenet_resnet18_celebamaskhq", help="Select model")
args.add_argument("-l_m", "--load_model", type=str,
                  default=None,
                  help="Load model from path")
args.add_argument("-s", "--save_dir", type=str, default="./runs", help="Save directory for models and tensorboard")
args.add_argument("-tfrecs", "--tf_record_path", type=str, default="/data/input/datasets/tf2_segmentation_tfrecords/detection/",
                  help="Save directory that contains train and validation tfrecords")
args.add_argument("-sb", "--shuffle_buffer", type=int, default=128, help="Size of the shuffle buffer")
args.add_argument("--width", type=int, default=1024, help="Size of the shuffle buffer")
args.add_argument("--height", type=int, default=512, help="Size of the shuffle buffer")
args.add_argument("--aux", action="store_true", default=False, help="Auxiliary losses included if true")
args.add_argument("--aux_weight", type=float, default=0.25, help="Auxiliary losses included if true")
args.add_argument("--random_seed", type=int, default=1, help="Set random seed to this if true")
# ============ Augmentation Arguments ===================== #
args.add_argument("--flip_up_down", action="store_true", default=False, help="Randomly flip images up and down")
args.add_argument("--flip_left_right", action="store_true", default=False, help="Randomly flip images right left")
args.add_argument("--random_crop_height", type=int, default=None,
                  help="Height of random crop, random_crop_width must be given with this")
args.add_argument("--random_crop_width", type=int, default=None,
                  help="Width of random crop, random_crop_height must be given with this")
args.add_argument("--random_hue", action="store_true", default=False, help="Randomly change hue")
args.add_argument("--random_saturation", action="store_true", default=False, help="Randomly change saturation")
args.add_argument("--random_brightness", action="store_true", default=False, help="Randomly change brightness")
args.add_argument("--random_contrast", action="store_true", default=False, help="Randomly change contrast")
args.add_argument("--random_quality", action="store_true", default=False, help="Randomly change jpeg quality")
args = args.parse_args()

tf.random.set_seed(args.random_seed)
random_crop_size = (args.random_crop_width, args.random_crop_height) \
    if args.random_crop_width is not None and args.random_crop_height is not None \
    else None
dataset_name = args.dataset
aux = args.aux
aux_weight = args.aux_weight
epochs = args.epochs
batch_size = args.batch_size
classes = args.classes
optimizer_name = args.optimizer
lr = args.lr
momentum = args.momentum
model_name = args.model
log_freq = args.logging_freq
write_image_summary_steps = args.write_image_summary_steps
time = str(datetime.datetime.now())
time = time.translate(str.maketrans('', '', string.punctuation)).replace(" ", "-")[:-8]
logdir = os.path.join(args.save_dir, "{}_epochs-{}_bs-{}_{}_lr_{}-{}_{}_{}".format(dataset_name, epochs, batch_size,
                                                                                   optimizer_name, lr,
                                                                                   args.lr_scheduler,
                                                                                   model_name,
                                                                                   time))

dataset = tfds.load(dataset_name, data_dir=args.tf_record_path)
dataset_train = dataset['train']
total_samples = len(list(dataset_train))
dataset_train = dataset_train.shard(hvd.size(), hvd.local_rank())
print(total_samples)

# TODO: Add augmentation
# TODO: Start with an SSD and Centernet implementation
