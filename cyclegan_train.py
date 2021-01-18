"""
This Script is for GAN based trainings.
"""
import argparse
import tensorflow as tf
import tensorflow.keras as K
import tqdm
import math
from model_provider import get_model
from losses import get_loss
from utils.augment_images import augment_autoencoder
import datetime
import string
import os
from utils.create_gan_tfrecords import TFRecordsGAN
import json

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
mirrored_strategy = tf.distribute.MirroredStrategy()
args = argparse.ArgumentParser(description="Train a network with specific settings")
args.add_argument("-d", "--dataset", type=str, default="celeb_a",
                  help="Name a dataset from the tf_dataset collection",
                  choices=["celeb_a", "zebra2horse"])
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
args.add_argument("-bs", "--batch_size", type=int, default=8, help="Size of mini-batch")
args.add_argument("-si", "--save_interval", type=int, default=5, help="Save interval for model")
args.add_argument("-wis", "--write_image_summary_steps", type=int, default=5, help="Add images to tfrecords "
                                                                                   "after these many logging steps")
args.add_argument("-m", "--model", type=str, default="cyclegan", help="Select model")
args.add_argument("-l_m", "--load_model", type=str,
                  default=None,
                  help="Load model from path")
args.add_argument("-s", "--save_dir", type=str, default="./gan_runs", help="Save directory for models and tensorboard")
args.add_argument("-tfrecs", "--tf_record_path", type=str, default="/data/input/datasets/tf2_gan_tfrecords",
                  help="Save directory that contains train and validation tfrecords")
args.add_argument("-sb", "--shuffle_buffer", type=int, default=128, help="Size of the shuffle buffer")
args.add_argument("--width", type=int, default=256, help="Size of the shuffle buffer")
args.add_argument("--height", type=int, default=256, help="Size of the shuffle buffer")
args.add_argument("--random_seed", type=int, default=1, help="Set random seed to this if true")
args = args.parse_args()

tf.random.set_seed(args.random_seed)
time = str(datetime.datetime.now())
time = time.translate(str.maketrans('', '', string.punctuation)).replace(" ", "-")[:-8]
logdir = os.path.join(args.save_dir,
                      "{}_epochs-{}_bs-{}_{}_lr_{}-{}_{}_{}_seed_{}".format(args.dataset, args.epochs,
                                                                            args.batch_size,
                                                                            args.optimizer,
                                                                            args.lr,
                                                                            args.lr_scheduler,
                                                                            args.model,
                                                                            time, args.random_seed))

x_train_A = TFRecordsGAN(
    tfrecord_path=
    "{}/{}_train.tfrecords".format(args.tf_record_path, args.dataset + "_a")).read_tfrecords()
x_train_B = TFRecordsGAN(
    tfrecord_path=
    "{}/{}_train.tfrecords".format(args.tf_record_path, args.dataset + "_b")).read_tfrecords()
with open("/data/input/datasets/tf2_gan_tfrecords/data_samples.json") as f:
    data = json.load(f)
num_samples_ab = [data[args.dataset + "_a"], data[args.dataset + "_b"]]
if num_samples_ab[0] > num_samples_ab[1]:
    total_samples = num_samples_ab[0]
    x_train_B = x_train_B.repeat()
else:
    total_samples = num_samples_ab[1]
    x_train_A = x_train_A.repeat()

x_test_A = TFRecordsGAN(
    tfrecord_path=
    "{}/{}_val.tfrecords".format(args.tf_record_path, args.dataset + "_a")).read_tfrecords().batch(args.batch_size,
                                                                                                   drop_remainder=True)
x_test_B = TFRecordsGAN(
    tfrecord_path=
    "{}/{}_val.tfrecords".format(args.tf_record_path, args.dataset + "_b")).read_tfrecords().batch(args.batch_size,
                                                                                                   drop_remainder=True)
augmentor = lambda batch: augment_autoencoder(batch, size=(args.height, args.width))
x_train_A, x_train_B = x_train_A.map(augmentor).batch(args.batch_size, drop_remainder=True).shuffle(args.shuffle_buffer).prefetch(
    tf.data.experimental.AUTOTUNE), x_train_B.map(augmentor).batch(args.batch_size, drop_remainder=True).shuffle(args.shuffle_buffer).prefetch(
    tf.data.experimental.AUTOTUNE)
x_train_A = mirrored_strategy.experimental_distribute_dataset(x_train_A)
x_train_B = mirrored_strategy.experimental_distribute_dataset(x_train_B)
x_test_A = mirrored_strategy.experimental_distribute_dataset(x_test_A)
x_test_B = mirrored_strategy.experimental_distribute_dataset(x_test_B)

calc_real_fake_loss = get_loss("MSE")
calc_cycle_loss = get_loss("MAE")
calc_id_loss = get_loss("MAE")
w_cycle = 5.0
w_real_fake = 10.0
w_id = 0.5

if args.final_activation == "tanh":
    act = tf.nn.tanh
elif args.final_activation == "sigmoid":
    act = tf.nn.sigmoid
else:
    act = tf.nn.softmax


def train_step(mini_batch_a, mini_batch_b, gen_loss_only=False):
    img_a = tf.cast(mini_batch_a, dtype=tf.float32)
    img_b = tf.cast(mini_batch_b, dtype=tf.float32)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_b = gen_AB(img_a, training=True)
        fake_a = gen_BA(img_b, training=True)
        recons_a = gen_BA(fake_b, training=True)
        recons_b = gen_AB(fake_a, training=True)
        valid_a = disc_A(fake_a, training=True)
        valid_b = disc_B(fake_b, training=True)
        id_a = gen_BA(img_a)
        id_b = gen_AB(img_b)
        # ====================== Generator Cycle ============================== #
        cycle_loss = tf.reduce_mean(calc_cycle_loss(img_a, recons_a) + calc_cycle_loss(img_b, recons_b))
        id_loss = tf.reduce_mean(calc_id_loss(img_a, id_a) + calc_id_loss(img_b, id_b))
        real_fake_loss_gen = tf.reduce_mean(
            calc_real_fake_loss(1, disc_A(fake_a)) + calc_real_fake_loss(1, disc_B(fake_b)))
        gen_loss = w_cycle * cycle_loss + w_id * id_loss + w_real_fake * real_fake_loss_gen
        # ====================== Discriminator Cycle ========================== #
        real_fake_loss_disc = calc_real_fake_loss(0, valid_a) + calc_real_fake_loss(0, valid_b) + \
                              calc_real_fake_loss(1, disc_A(img_a)) + calc_real_fake_loss(1, disc_B(img_b))
        disc_loss = tf.reduce_mean(real_fake_loss_disc)

    gens_grads = gen_tape.gradient(gen_loss, gen_AB.trainable_variables + gen_BA.trainable_variables)
    g_optimizer.apply_gradients(zip(gens_grads, gen_AB.trainable_variables + gen_BA.trainable_variables))
    if not gen_loss_only:
        disc_grads = disc_tape.gradient(disc_loss, disc_A.trainable_variables + disc_B.trainable_variables)
        d_optimizer.apply_gradients(zip(disc_grads, disc_A.trainable_variables + disc_B.trainable_variables))
    return gen_loss, disc_loss


@tf.function
def distributed_train_step(dist_inputs_a, dist_inputs_b, balance_ratio=-1.):
    per_replica_gen_losses, per_replica_disc_losses = mirrored_strategy.run(train_step,
                                                                            args=(dist_inputs_a, dist_inputs_b))
    reduced_gen_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_gen_losses,
                                                axis=None)
    reduced_disc_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_disc_losses,
                                                 axis=None)
    while reduced_gen_loss / reduced_disc_loss < balance_ratio:
        per_replica_gen_losses, per_replica_disc_losses = mirrored_strategy.run(train_step, args=(
            dist_inputs_a, dist_inputs_b, True))
        reduced_gen_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_gen_losses,
                                                    axis=None)
        reduced_disc_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_disc_losses,
                                                     axis=None)
    return reduced_gen_loss, reduced_disc_loss


total_steps = math.ceil(total_samples / args.batch_size)

with mirrored_strategy.scope():
    gen_AB = get_model(args.model + "_gen", final_activation=act, type="gan")
    gen_BA = get_model(args.model + "_gen", final_activation=act, type="gan")
    disc_A = get_model(args.model + "_disc", type="gan")
    disc_B = get_model(args.model + "_disc", type="gan")
    tmp = tf.cast(tf.random.uniform((1, args.height, args.width, 3), dtype=tf.float32, minval=0, maxval=1),
                  dtype=tf.float32)
    gen_AB(tmp), gen_BA(tmp), disc_A(tmp), disc_B(tmp)
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
        # TODO: Save and load 4 models. Write logic for them.
        if os.path.exists(os.path.join(args.load_model, "saved_model.pb")):
            pretrained_model = K.models.load_model(args.load_model)
            gen_AB.set_weights(pretrained_model.get_weights())
            print("Model loaded from {} successfully".format(os.path.basename(args.load_model)))
        else:
            print("No file found at {}".format(os.path.join(args.load_model, "saved_model.pb")))

train_writer = tf.summary.create_file_writer(os.path.join(logdir, "train"))
val_writer = tf.summary.create_file_writer(os.path.join(logdir, "val"))

val_loss = 0
c_step = 0


def get_gen_disc_loss(im_a, im_b):
    fake_b = gen_AB(im_a, training=True)
    fake_a = gen_BA(im_b, training=True)
    recons_a = gen_BA(fake_b, training=True)
    recons_b = gen_AB(fake_a, training=True)
    valid_a = disc_A(fake_a, training=True)
    valid_b = disc_B(fake_b, training=True)
    id_a = gen_BA(im_a)
    id_b = gen_AB(im_b)
    # ====================== Generator Cycle ============================== #
    cycle_loss = calc_cycle_loss(im_a, recons_a) + calc_cycle_loss(im_b, recons_b)
    id_loss = calc_id_loss(im_a, id_a) + calc_id_loss(im_b, id_b)
    real_fake_loss_gen = calc_real_fake_loss(1, disc_A(fake_a)) + calc_real_fake_loss(1, disc_B(fake_b))
    gen_loss = cycle_loss + id_loss + real_fake_loss_gen
    # ====================== Discriminator Cycle ========================== #
    real_fake_loss_disc = calc_real_fake_loss(0, disc_A(valid_a)) + calc_real_fake_loss(0, disc_B(valid_b)) + \
                          calc_real_fake_loss(1, disc_A(im_a)) + calc_real_fake_loss(1, disc_B(im_b))
    disc_loss = tf.reduce_mean(real_fake_loss_disc)
    # return gen_loss, disc_loss, output


g_loss, d_loss = 0, 1

for epoch in range(args.epochs):
    print("\n\n-------------Epoch {}-----------".format(epoch))
    if epoch % args.save_interval == 0:
        K.models.save_model(gen_AB, os.path.join(logdir, args.model, "gen_AB", str(epoch)))
        K.models.save_model(gen_BA, os.path.join(logdir, args.model, "gen_BA", str(epoch)))
        print("Model at Epoch {}, saved at {}".format(epoch, os.path.join(logdir, args.model, str(epoch))))
    for s, (img_a, img_b) in enumerate(tqdm.tqdm(zip(x_train_A, x_train_B), total=total_steps)):
        c_step = epoch * total_steps + s
        g_loss, d_loss = distributed_train_step(img_a, img_b)
        with train_writer.as_default():
            tmp = lrs(step=c_step)
            tf.summary.scalar("Learning Rate", tmp, c_step)
            tf.summary.scalar("G_Loss", g_loss.numpy(), c_step)
            tf.summary.scalar("D_Loss", d_loss.numpy(), c_step)
            if s % args.write_image_summary_steps == 0:
                if len(physical_devices) > 1:
                    img_a = tf.cast(img_a.values[0], dtype=tf.float32)
                    img_b = tf.cast(img_b.values[0], dtype=tf.float32)
                fake_img_b = gen_AB(img_a)
                fake_img_a = gen_BA(img_b)
                confidence_a = disc_A(fake_img_a)
                confidence_b = disc_B(fake_img_b)
                tf.summary.image("img_a", img_a, step=c_step)
                tf.summary.image("img_b", img_b, step=c_step)
                tf.summary.image("fake_img_a", fake_img_a, step=c_step)
                tf.summary.image("fake_img_b", fake_img_b, step=c_step)
                tf.summary.image("confidence_a", confidence_a, step=c_step)
                tf.summary.image("confidence_b", confidence_b, step=c_step)
    # gen_loss, disc_loss = 0, 0
    # for img in tqdm.tqdm(x_test, total=test_total_steps):
    #     if len(physical_devices) > 1:
    #         for im in img.values:
    #             gen_loss, disc_loss, output = get_gen_disc_loss(im)
    #         img = im
    #     else:
    #         gen_loss, disc_loss, output = get_gen_disc_loss(img)
    # with val_writer.as_default():
    #     tf.summary.scalar("G_Loss", gen_loss / test_total_steps, c_step)
    #     tf.summary.scalar("D_Loss", disc_loss / test_total_steps, c_step)
    #     tf.summary.image("input", img, step=c_step)
    #     tf.summary.image("output", output, step=c_step)
