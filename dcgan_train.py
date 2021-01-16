"""
This Script is for GAN based trainings.
"""
import argparse
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_datasets as tfds
import tqdm
from model_provider import get_model
from losses import get_loss
from utils.augment_images import augment_autoencoder
import datetime
import string
import os

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
mirrored_strategy = tf.distribute.MirroredStrategy()
args = argparse.ArgumentParser(description="Train a network with specific settings")
args.add_argument("-d", "--dataset", type=str, default="celeb_a",
                  help="Name a dataset from the tf_dataset collection",
                  choices=["celeb_a"])
args.add_argument("-opt", "--optimizer", type=str, default="Adam", help="Select optimizer",
                  choices=["SGD", "RMSProp", "Adam"])
args.add_argument("-fa", "--final_activation", type=str, default="tanh",
                  help="Select final activation layer for decoder",
                  choices=["tanh", "sigmoid", "softmax"])
args.add_argument("-lrs", "--lr_scheduler", type=str, default="exp_decay", help="Select learning rate scheduler",
                  choices=["poly", "exp_decay"])
args.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs to train")
args.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate")
args.add_argument("--momentum", type=float, default=0.9, help="Momentum")
args.add_argument("--loss", type=str, default="MSE",
                  choices=["cross_entropy", "focal_loss", "binary_crossentropy", "MSE", "MAE"],
                  help="Loss function, Hint: Use the tanh activation for model with MSE and MAE, "
                       "and sigmoid for others")
args.add_argument("-bs", "--batch_size", type=int, default=16, help="Size of mini-batch")
args.add_argument("-si", "--save_interval", type=int, default=5, help="Save interval for model")
args.add_argument("-wis", "--write_image_summary_steps", type=int, default=5, help="Add images to tfrecords "
                                                                                   "after these many logging steps")
args.add_argument("-m", "--model", type=str, default="faceswap", help="Select model")
args.add_argument("-l_m", "--load_model", type=str,
                  default=None,
                  help="Load model from path")
args.add_argument("-s", "--save_dir", type=str, default="./gan_runs", help="Save directory for models and tensorboard")
args.add_argument("-tfrecs", "--tf_record_path", type=str, default="tmp",
                  help="Save directory that contains train and validation tfrecords")
args.add_argument("-sb", "--shuffle_buffer", type=int, default=128, help="Size of the shuffle buffer")
args.add_argument("--width", type=int, default=64, help="Size of the shuffle buffer")
args.add_argument("--height", type=int, default=64, help="Size of the shuffle buffer")
args.add_argument("--random_seed", type=int, default=1, help="Set random seed to this if true")
args = args.parse_args()

tf.random.set_seed(args.random_seed)
time = str(datetime.datetime.now())
time = time.translate(str.maketrans('', '', string.punctuation)).replace(" ", "-")[:-8]
logdir = os.path.join(args.save_dir,
                      "{}_epochs-{}_{}_bs-{}_{}_lr_{}-{}_{}_{}_seed_{}".format(args.dataset, args.epochs,
                                                                               args.loss,
                                                                               args.batch_size,
                                                                               args.optimizer,
                                                                               args.lr,
                                                                               args.lr_scheduler,
                                                                               args.model,
                                                                               time, args.random_seed))

x_train = tfds.load(args.dataset, split='train', data_dir=args.tf_record_path).batch(args.batch_size).prefetch(
    tf.data.experimental.AUTOTUNE).shuffle(args.shuffle_buffer)
total_steps = len(x_train)
x_test = tfds.load(args.dataset, split='test', data_dir=args.tf_record_path).batch(args.batch_size)
test_total_steps = len(x_test)
augmentor = lambda batch: augment_autoencoder(batch, size=(args.height, args.width))
x_train, x_test = x_train.map(augmentor), x_test.map(augmentor)
x_train = mirrored_strategy.experimental_distribute_dataset(x_train)
x_test = mirrored_strategy.experimental_distribute_dataset(x_test)

calc_loss = get_loss(args.loss)

if args.final_activation == "tanh":
    act = tf.nn.tanh
elif args.final_activation == "sigmoid":
    act = tf.nn.sigmoid
else:
    act = tf.nn.softmax


def train_step(mini_batch, gen_loss_only=False, pick=None):
    img = tf.cast(mini_batch, dtype=tf.float32)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        output = generator(img, training=True)
        fake_output = discriminator(tf.concat([img, output], axis=-1))
        real_output = discriminator(tf.concat([img, img], axis=-1))
        gen_loss = calc_loss(tf.ones(shape=fake_output.shape), fake_output)
        gen_loss = tf.reduce_mean(gen_loss)
        disc_loss = calc_loss(tf.ones(shape=real_output.shape), real_output) + calc_loss(
            tf.zeros(shape=fake_output.shape), fake_output)
        disc_loss = tf.reduce_mean(disc_loss)
    if pick is not None:
        gen_trainable_vars = [var for var in generator.trainable_variables if pick in var.name]
        disc_trainable_vars = [var for var in discriminator.trainable_variables if pick in var.name]
    else:
        gen_trainable_vars = generator.trainable_variables
        disc_trainable_vars = discriminator.trainable_variables

    gen_grads = gen_tape.gradient(gen_loss, gen_trainable_vars)
    g_optimizer.apply_gradients(zip(gen_grads, gen_trainable_vars))
    if not gen_loss_only:
        disc_grads = disc_tape.gradient(disc_loss, disc_trainable_vars)
        d_optimizer.apply_gradients(zip(disc_grads, disc_trainable_vars))
    return gen_loss, disc_loss


@tf.function
def distributed_train_step(dist_inputs, balance_ratio=-1):
    per_replica_gen_losses, per_replica_disc_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
    reduced_gen_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_gen_losses,
                                                axis=None)
    reduced_disc_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_disc_losses,
                                                 axis=None)
    while reduced_gen_loss / reduced_disc_loss < balance_ratio:
        per_replica_gen_losses, per_replica_disc_losses = mirrored_strategy.run(train_step, args=(dist_inputs, True))
        reduced_gen_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_gen_losses,
                                                    axis=None)
        reduced_disc_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_disc_losses,
                                                     axis=None)
    return reduced_gen_loss, reduced_disc_loss


with mirrored_strategy.scope():
    generator = get_model(args.model + "_gen", activation=act, type="gan")
    discriminator = get_model(args.model + "_disc", type="gan")
    tmp = tf.cast(tf.random.uniform((1, args.height, args.width, 3), dtype=tf.float32, minval=0, maxval=1),
                  dtype=tf.float32)
    generator(tmp)
    tmp = tf.cast(tf.random.uniform((1, args.height, args.width, 6), dtype=tf.float32, minval=0, maxval=1),
                  dtype=tf.float32)
    discriminator(tmp)
    if args.lr_scheduler == "poly":
        lrs = K.optimizers.schedules.PolynomialDecay(args.lr,
                                                     decay_steps=args.epochs * total_steps,
                                                     end_learning_rate=1e-8, power=2.0)
    elif args.lr_scheduler == "exp_decay":
        lrs = K.optimizers.schedules.ExponentialDecay(args.lr,
                                                      decay_steps=args.epochs * total_steps,
                                                      decay_rate=0.5)
    else:
        lrs = args.lr
    if args.optimizer == "Adam":
        g_optimizer = K.optimizers.Adam(learning_rate=lrs)
        d_optimizer = K.optimizers.Adam(learning_rate=lrs)
    elif args.optimizer == "RMSProp":
        g_optimizer = K.optimizers.RMSprop(learning_rate=lrs, momentum=args.momentum)
        d_optimizer = K.optimizers.RMSprop(learning_rate=lrs, momentum=args.momentum)
    elif args.optimizer == "SGD":
        g_optimizer = K.optimizers.SGD(learning_rate=lrs, momentum=args.momentum)
        d_optimizer = K.optimizers.SGD(learning_rate=lrs, momentum=args.momentum)
    if args.load_model:
        if os.path.exists(os.path.join(args.load_model, "saved_model.pb")):
            pretrained_model = K.models.load_model(args.load_model)
            generator.set_weights(pretrained_model.get_weights())
            print("Model loaded from {} successfully".format(os.path.basename(args.load_model)))
        else:
            print("No file found at {}".format(os.path.join(args.load_model, "saved_model.pb")))

train_writer = tf.summary.create_file_writer(os.path.join(logdir, "train"))
val_writer = tf.summary.create_file_writer(os.path.join(logdir, "val"))

val_loss = 0
c_step = 0


def get_gen_disc_loss(im):
    output = generator(im, training=True)
    fake_output = discriminator(tf.concat([im, output], axis=-1))
    real_output = discriminator(tf.concat([im, im], axis=-1))
    gen_loss = calc_loss(tf.ones(shape=fake_output.shape), fake_output)
    gen_loss += tf.reduce_mean(gen_loss)
    disc_loss = calc_loss(tf.ones(shape=real_output.shape), real_output) + calc_loss(
        tf.zeros(shape=fake_output.shape), fake_output)
    disc_loss += tf.reduce_mean(disc_loss)
    return gen_loss, disc_loss, output


g_loss, d_loss = 0, 1

for epoch in range(args.epochs):
    print("\n\n-------------Epoch {}-----------".format(epoch))
    if epoch % args.save_interval == 0:
        K.models.save_model(generator, os.path.join(logdir, args.model, str(epoch)))
        print("Model at Epoch {}, saved at {}".format(epoch, os.path.join(logdir, args.model, str(epoch))))
    for s, img in enumerate(tqdm.tqdm(x_train, total=total_steps)):
        c_step = epoch * total_steps + s
        g_loss, d_loss = distributed_train_step(img, balance_ratio=(g_loss - d_loss))
        with train_writer.as_default():
            tmp = lrs(step=c_step)
            tf.summary.scalar("Learning Rate", tmp, c_step)
            tf.summary.scalar("G_Loss", g_loss.numpy(), c_step)
            tf.summary.scalar("D_Loss", d_loss.numpy(), c_step)
            if s % args.write_image_summary_steps == 0:
                if len(physical_devices) > 1:
                    img = tf.cast(img.values[0], dtype=tf.float32)
                new_img = generator(img)
                tf.summary.image("input", img, step=c_step)
                tf.summary.image("output", new_img, step=c_step)
    gen_loss, disc_loss = 0, 0
    for img in tqdm.tqdm(x_test, total=test_total_steps):
        if len(physical_devices) > 1:
            for im in img.values:
                gen_loss, disc_loss, output = get_gen_disc_loss(im)
            img = im
        else:
            gen_loss, disc_loss, output = get_gen_disc_loss(img)
    with val_writer.as_default():
        tf.summary.scalar("G_Loss", gen_loss / test_total_steps, c_step)
        tf.summary.scalar("D_Loss", disc_loss / test_total_steps, c_step)
        tf.summary.image("input", img, step=c_step)
        tf.summary.image("output", output, step=c_step)
