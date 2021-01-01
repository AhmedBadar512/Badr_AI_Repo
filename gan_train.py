"""
This Script is for GAN based trainings.
"""
import argparse
import cv2
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_datasets as tfds
import tqdm
from model_provider import get_model
from losses import get_loss
from utils.augment_images import augment_autoencoder

physical_devices = tf.config.experimental.list_physical_devices("GPU")

if len(physical_devices) > 1:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    mirrored_strategy = tf.distribute.MirroredStrategy()
else:
    mirrored_strategy = tf.distribute.MirroredStrategy()
args = argparse.ArgumentParser(description="Train a network with specific settings")
args.add_argument("-d", "--dataset", type=str, default="celeb_a",
                  help="Name a dataset from the tf_dataset collection",
                  choices=["celeb_a"])
args.add_argument("-opt", "--optimizer", type=str, default="Adam", help="Select optimizer",
                  choices=["SGD", "RMSProp", "Adam"])
args.add_argument("-lrs", "--lr_scheduler", type=str, default="exp_decay", help="Select learning rate scheduler",
                  choices=["poly", "exp_decay"])
args.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs to train")
args.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate")
args.add_argument("--momentum", type=float, default=0.9, help="Momentum")
args.add_argument("-l", "--logging_freq", type=int, default=50, help="Add to tfrecords after this many steps")
args.add_argument("--loss", type=str, default="binary_crossentropy",
                  choices=["cross_entropy", "focal_loss", "binary_crossentropy"],
                  help="Loss function")
args.add_argument("-bs", "--batch_size", type=int, default=4, help="Size of mini-batch")
args.add_argument("-si", "--save_interval", type=int, default=5, help="Save interval for model")
args.add_argument("-wis", "--write_image_summary_steps", type=int, default=5, help="Add images to tfrecords "

                                                                                   "after these many logging steps")
args.add_argument("-m", "--model", type=str, default="faceswap", help="Select model")
args.add_argument("-l_m", "--load_model", type=str,
                  default=None,
                  help="Load model from path")
args.add_argument("-s", "--save_dir", type=str, default="./runs", help="Save directory for models and tensorboard")
args.add_argument("-tfrecs", "--tf_record_path", type=str, default="tmp",
                  help="Save directory that contains train and validation tfrecords")
args.add_argument("-sb", "--shuffle_buffer", type=int, default=128, help="Size of the shuffle buffer")
args.add_argument("--width", type=int, default=64, help="Size of the shuffle buffer")
args.add_argument("--height", type=int, default=64, help="Size of the shuffle buffer")
args.add_argument("--random_seed", type=int, default=1, help="Set random seed to this if true")
args = args.parse_args()


x_train = tfds.load(args.dataset, split='train', data_dir=args.tf_record_path).batch(args.batch_size).prefetch(
                    tf.data.experimental.AUTOTUNE).shuffle(args.shuffle_buffer)
total_samples = len(x_train)
x_test = tfds.load(args.dataset, split='test', data_dir=args.tf_record_path).batch(args.batch_size)
augmentor = lambda batch: augment_autoencoder(batch, size=(args.height, args.width))
x_train, x_test = x_train.map(augmentor), x_test.map(augmentor)
x_train = mirrored_strategy.experimental_distribute_dataset(x_train)
x_test = mirrored_strategy.experimental_distribute_dataset(x_test)


calc_loss = get_loss(args.loss)


def train_step(mini_batch, pick=None):
    img = tf.cast(mini_batch, dtype=tf.float32)
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
    model = get_model(args.model)
    if args.lr_scheduler:
        lrs = K.optimizers.schedules.PolynomialDecay(args.lr, decay_steps=args.epochs * total_samples // args.batch_size)
    if args.optimizer == "Adam":
        optimizer = K.optimizers.Adam(learning_rate=1e-6)

img, new_img = None, None
cv2.namedWindow("img", 0)
cv2.namedWindow("syn_img", 0)
for epoch in range(999999):
    for img in tqdm.tqdm(x_train):
        loss = distributed_train_step(img).numpy()
        print("Loss: {}".format(loss))

        img = tf.cast(img.values[0], dtype=tf.float32)
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
