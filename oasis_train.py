#!/usr/bin/python3
import tensorflow as tf
import json
from model_provider import get_model
from visualization_dicts import gpu_cs_labels, generate_random_colors, gpu_random_labels
from seg_visualizer import get_images_custom
from utils.create_seg_tfrecords import TFRecordsSeg
from utils.augment_images import augment_seg
import os
import tensorflow.keras as K
import datetime
import string
from losses import get_loss, gradient_penalty
import argparse
import psutil
import sys

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
mirrored_strategy = tf.distribute.MirroredStrategy()

args = argparse.ArgumentParser(description="Train a network with specific settings")
args.add_argument("-d", "--dataset", type=str, default="cityscapes",
                  help="Name a dataset from the tf_dataset collection",
                  choices=["cityscapes19"])
args.add_argument("-c", "--classes", type=int, default=32, help="Number of classes")
args.add_argument("-opt", "--optimizer", type=str, default="Adam", help="Select optimizer",
                  choices=["SGD", "RMSProp", "Adam"])
args.add_argument("-lrs", "--lr_scheduler", type=str, default="poly", help="Select learning rate scheduler",
                  choices=["poly", "exp_decay", "constant"])
args.add_argument("-gm", "--gan_mode", type=str, default="normal", help="Select training mode for GAN",
                  choices=["normal", "wgan_gp", "ls_gan", "hinge", "normal"])
args.add_argument("-cm", "--cut_mode", type=str, default="cut", help="Select training mode for GAN",
                  choices=["cut", "fastcut"])
args.add_argument("-e", "--epochs", type=int, default=1000, help="Number of epochs to train")
args.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
args.add_argument("--momentum", type=float, default=0.9, help="Momentum")
args.add_argument("-bs", "--batch_size", type=int, default=4, help="Size of mini-batch")
args.add_argument("-si", "--save_interval", type=int, default=5, help="Save interval for model")
args.add_argument("-m", "--model", type=str, default="oasis", help="Select model")
args.add_argument("-logs", "--logdir", type=str, default="./logs_oasis", help="Directory to save tensorboard logdir")
args.add_argument("-l_m", "--load_model", type=str,
                  default=None,
                  help="Load model from path")
args.add_argument("-s", "--save_dir", type=str, default="./oasis_runs",
                  help="Save directory for models and tensorboard")
args.add_argument("-tfrecs", "--tf_record_path", type=str, default="/data/input/datasets/tf2_segmentation_tfrecords",
                  help="Save directory that contains train and validation tfrecords")
args.add_argument("-sb", "--shuffle_buffer", type=int, default=128, help="Size of the shuffle buffer")
args.add_argument("--width", type=int, default=584, help="Size of the shuffle buffer")
args.add_argument("--height", type=int, default=286, help="Size of the shuffle buffer")
args.add_argument("--c_width", type=int, default=512, help="Crop width")
args.add_argument("--c_height", type=int, default=256, help="Crop height")
args.add_argument("-mem_lim", "--memory_limit", type=int, default=90, help="Restart if RAM exceeds this % of total")
# ============ Augmentation Arguments ===================== #
args.add_argument("--flip_up_down", action="store_true", default=False, help="Randomly flip images up and down")
args.add_argument("--flip_left_right", action="store_true", default=False, help="Randomly flip images right left")
args.add_argument("--random_hue", action="store_true", default=False, help="Randomly change hue")
args.add_argument("--random_saturation", action="store_true", default=False, help="Randomly change saturation")
args.add_argument("--random_brightness", action="store_true", default=False, help="Randomly change brightness")
args.add_argument("--random_contrast", action="store_true", default=False, help="Randomly change contrast")
args.add_argument("--random_quality", action="store_true", default=False, help="Randomly change jpeg quality")
args = args.parse_args()

tf_record_path = args.tf_record_path
dataset = args.dataset
BUFFER_SIZE = args.shuffle_buffer
BATCH_SIZE = args.batch_size
IMG_WIDTH = args.width
IMG_HEIGHT = args.height
CROP_HEIGHT = args.c_height if args.c_height < IMG_HEIGHT else IMG_HEIGHT
CROP_WIDTH = args.c_width if args.c_width < IMG_WIDTH else IMG_WIDTH
nce_identity = True if args.cut_mode == "cut" else False
LAMBDA_LM = 10
EPOCHS = args.epochs
LEARNING_RATE = args.lr
G_LEARNING_RATE, D_LEARNING_RATE = 1e-4, 4e-4
LEARNING_RATE_SCHEDULER = args.lr_scheduler
save_interval = args.save_interval
save_dir = args.save_dir
load_model_path = args.load_model
MODEL = args.model
gan_mode = args.gan_mode
time = str(datetime.datetime.now())
time = time.translate(str.maketrans('', '', string.punctuation)).replace(" ", "-")[:-8]
logdir = "{}_{}_e{}_lr{}_{}x{}_{}".format(time, MODEL, EPOCHS, LEARNING_RATE, IMG_HEIGHT, IMG_WIDTH, gan_mode)
# tf.random.set_seed(128)
# =========== Load Dataset ============ #

if dataset == "cityscapes19":
    cs_19 = True
    dataset = "cityscapes"
else:
    cs_19 = False
if not cs_19:
    cmap = generate_random_colors(bg_class=args.classes)

dataset_train = TFRecordsSeg(
    tfrecord_path=
    "{}/{}_train.tfrecords".format(args.tf_record_path, dataset)).read_tfrecords()
dataset_validation = TFRecordsSeg(
    tfrecord_path=
    "{}/{}_val.tfrecords".format(args.tf_record_path, dataset)).read_tfrecords()
augmentor = lambda image, label: augment_seg(image, label,
                                             args.flip_up_down,
                                             args.flip_left_right,
                                             (CROP_HEIGHT, CROP_WIDTH),
                                             args.random_hue,
                                             args.random_saturation,
                                             args.random_brightness,
                                             args.random_contrast,
                                             args.random_quality)
with open("/data/input/datasets/tf2_segmentation_tfrecords/data_samples.json") as f:
    data = json.load(f)
total_samples = data[dataset]
# =========== Process dataset ============ #
assert dataset_train is not None, "Training dataset can not be None"
assert dataset_validation is not None, "Either test or validation dataset should not be None"

eval_dataset = dataset_validation
get_images_processed = lambda image, label: get_images_custom(image, label, (args.height, args.width), cs_19)

processed_train = dataset_train.map(get_images_processed)
processed_train = processed_train.map(augmentor)
processed_val = dataset_validation.map(get_images_processed)
processed_train = processed_train.shuffle(args.shuffle_buffer).batch(BATCH_SIZE, drop_remainder=True).prefetch(
    tf.data.experimental.AUTOTUNE)
processed_val = processed_val.shuffle(args.shuffle_buffer).batch(BATCH_SIZE, drop_remainder=True) \
    if (dataset_validation is not None) else None
processed_train = mirrored_strategy.experimental_distribute_dataset(processed_train)
processed_val = mirrored_strategy.experimental_distribute_dataset(processed_val)


gan_loss_obj = get_loss(name="cross_entropy")
labelmix_loss_obj = get_loss(name="MSE")


# TODO: Add Regularization loss


def discriminator_loss(real_seg, fake_seg, label):
    fake_seg = tf.nn.softmax(fake_seg, axis=-1)
    real_seg = tf.nn.softmax(real_seg, axis=-1)
    real_seg += 1e-6
    fake_seg += 1e-6
    fake_label = tf.zeros_like(label) + (args.classes * [0] + [1])
    real_disc_loss = - tf.reduce_mean(label * tf.math.log(real_seg))
    # fake_disc_loss = gan_loss_obj(fake_label, fake_seg)
    fake_disc_loss = - tf.reduce_mean(fake_label * tf.math.log(fake_seg))
    return (real_disc_loss + fake_disc_loss) * 0.5


def generator_loss(fake_seg, label):
    # total_loss = gan_loss_obj(fake_seg, label)
    fake_seg = tf.nn.softmax(fake_seg, axis=-1)
    return - tf.reduce_mean(label * tf.math.log(fake_seg + 1e-5))


def generate_labelmix(label, fake_image, real_image):
    switch = tf.cast(tf.random.uniform((args.classes,), minval=0, maxval=2, dtype=tf.int32), dtype=tf.float32)    # Generate floats 0 and 1 equal to the number of classes N
    switch = tf.concat([switch, [1.]], axis=0)  # Always keep final dim N+1, 1
    switch_label = label * switch   # Multiply with the switch number to randomly turn classes off
    target_map = tf.reduce_sum(switch_label[..., :args.classes], axis=-1, keepdims=True)    # Add the classes to get target for real images
    mixed_image = target_map * real_image + (1 - target_map) * fake_image   # Create a mixed image based on target_map
    mixed_label = tf.concat([switch_label[..., :args.classes], (1-target_map)], axis=-1)
    return mixed_image, mixed_label
    # target_map = tf.argmax(label, axis=-1, output_type=tf.int32)
    # all_classes = tf.unique(tf.reshape(target_map, shape=(-1,)))
    # new_target = tf.zeros_like(target_map, dtype=tf.int32)
    # for c in all_classes:
    #     new_target += tf.cast(target_map == c, dtype=tf.int32) * tf.random.uniform((1,), 0, 2, dtype=tf.int32)
    # mixed_label = tf.one_hot(tf.where(new_target > 0, target_map, args.classes), depth=args.classes + 1)
    # new_target = tf.cast(new_target, dtype=tf.float32)
    # mixed_image = new_target * real_image + (1 - new_target) * fake_image
    # return mixed_image, mixed_label


if LEARNING_RATE_SCHEDULER == "poly":
    g_lrs = K.optimizers.schedules.PolynomialDecay(G_LEARNING_RATE,
                                                   decay_steps=EPOCHS,
                                                   end_learning_rate=1e-8, power=0.8)
    d_lrs = K.optimizers.schedules.PolynomialDecay(D_LEARNING_RATE,
                                                   decay_steps=EPOCHS,
                                                   end_learning_rate=1e-8, power=0.8)
elif LEARNING_RATE_SCHEDULER == "exp_decay":
    g_lrs = K.optimizers.schedules.ExponentialDecay(G_LEARNING_RATE,
                                                    decay_steps=1e5,
                                                    decay_rate=0.9)
    d_lrs = K.optimizers.schedules.ExponentialDecay(D_LEARNING_RATE,
                                                    decay_steps=1e5,
                                                    decay_rate=0.9)
else:
    g_lrs = G_LEARNING_RATE
    d_lrs = D_LEARNING_RATE

with mirrored_strategy.scope():
    tmp = tf.cast(tf.random.uniform((1, CROP_HEIGHT, CROP_WIDTH, 3), dtype=tf.float32, minval=0, maxval=1),
                  dtype=tf.float32)
    tmp1 = tf.cast(tf.random.uniform((1, CROP_HEIGHT, CROP_WIDTH, args.classes + 1), dtype=tf.float32, minval=0, maxval=1),
                   dtype=tf.float32)
    generator = get_model("{}_gen".format(MODEL), type="gan")
    discriminator = get_model("{}_disc".format(MODEL), type="gan", classes=args.classes)
    generator(tmp1), discriminator(tmp)
    generator_optimizer = tf.keras.optimizers.Adam(g_lrs, beta_1=0.0, beta_2=0.999)
    discriminator_optimizer = tf.keras.optimizers.Adam(d_lrs, beta_1=0.0, beta_2=0.999)


def load_models(models_parent_dir):
    assert os.path.exists(models_parent_dir), "The path {} is not valid".format(models_parent_dir)
    p_gen = K.models.load_model(os.path.join(models_parent_dir, "gen"))
    p_disc = K.models.load_model(os.path.join(models_parent_dir, "disc"))
    generator.set_weights(p_gen.get_weights())
    print("Generator loaded successfully")
    discriminator.set_weights(p_disc.get_weights())
    print("Discriminator loaded successfully")


if load_model_path is not None:
    load_models(load_model_path)
    START_EPOCH = int(load_model_path.split("/")[-1]) - 1
else:
    START_EPOCH = 0


def write_to_tensorboard(g_adv_loss, disc_loss, lm_loss, c_step, writer):
    def colorize_labels(processed_labs):
        return tf.cast(tf.squeeze(gpu_random_labels(processed_labs[..., tf.newaxis], cmap)), dtype=tf.uint8) \
            if not cs_19 else tf.cast(tf.squeeze(gpu_cs_labels(processed_labs[..., tf.newaxis])), dtype=tf.uint8)
    with writer.as_default():
        # colorize = lambda im: tf.cast(tf.squeeze(gpu_cs_labels(im)), dtype=tf.uint8) if cs_19 else \
        #     lambda im: tf.cast(tf.squeeze(gpu_random_labels(im, cmap)), dtype=tf.uint8)
        # colorize = gpu_cs_labels if cs_19 else gpu_random_labels
        tf.summary.scalar("G_Adv_Loss", g_adv_loss.numpy(), c_step)
        tf.summary.scalar("D_Loss", tf.reduce_mean(disc_loss).numpy(), c_step)
        tf.summary.scalar("Labelmix_Loss", tf.reduce_mean(lm_loss).numpy(), c_step)
        if len(physical_devices) > 1:
            o_img = tf.cast(tf.concat(mini_batch[0].values, axis=0), dtype=tf.float32) / 127.5 - 1
            o_seg = tf.cast(tf.concat(mini_batch[1].values, axis=0), dtype=tf.int32)
            img, seg = o_img, tf.one_hot(o_seg[..., 0], args.classes + 1)
            processed_labs = tf.concat(mini_batch[1].values, axis=0)
        else:
            img = mini_batch[0] / 127.5 - 1
            seg = tf.one_hot(mini_batch[1][..., 0], args.classes + 1)
            processed_labs = mini_batch[1]
        processed_labs = colorize_labels(processed_labs)
        f_image = generator(seg)
        m_im, m_lab = generate_labelmix(seg, f_image, img)
        processed_m_labs = colorize_labels(tf.argmax(m_lab, axis=-1))
        tf.summary.image("Img", (img + 1)/2, step=c_step)
        tf.summary.image("Mixed_Img", (m_im + 1)/2, step=c_step)
        tf.summary.image("translated_img", (f_image + 1)/2, step=c_step)
        tf.summary.image("Seg", processed_labs,
                         step=c_step)
        tf.summary.image("M_Seg", processed_m_labs,
                         step=c_step)


def train_step(mini_batch):
    img = mini_batch[0] / 127.5 - 1
    label = tf.one_hot(mini_batch[1][..., 0], args.classes + 1)
    with tf.GradientTape(persistent=True) as tape:
        fake_img = generator(label)
        seg_real, seg_fake = discriminator(img), discriminator(fake_img)

        # ============ Generator Cycle =============== #
        g_adv_loss = generator_loss(seg_fake, label)
        total_gen_loss = g_adv_loss

        # =========== Discriminator Cycle ============ #
        mixed_img, mixed_lbl = generate_labelmix(label, fake_img, img)
        m_pred = tf.nn.softmax(discriminator(mixed_img), axis=-1)
        lm_loss = LAMBDA_LM * tf.reduce_mean(labelmix_loss_obj(m_pred, mixed_lbl))
        # lm_loss = labelmix_loss(mixed_img, mixed_lbl, discriminator)
        # lm_loss = 0
        disc_loss = discriminator_loss(seg_real, seg_fake, label)
        total_disc_loss = disc_loss + lm_loss

    # ------------------- Disc Cycle -------------------- #
    # Calculate the gradients for generator and discriminator
    generator_gradients = tape.gradient(total_gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

    discriminator_gradients = tape.gradient(total_disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return g_adv_loss, disc_loss, lm_loss


@tf.function
def distributed_train_step(dist_mini_batch):
    pr_g_adv_losses, pr_disc_losses, pr_lm_losses = \
        mirrored_strategy.run(train_step, args=(dist_mini_batch,))
    r_g_adv_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                            pr_g_adv_losses,
                                            axis=None)
    r_disc_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                           pr_disc_losses,
                                           axis=None)
    r_lm_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                         pr_lm_losses,
                                         axis=None)
    return r_g_adv_loss, r_disc_loss, r_lm_loss


train_writer = tf.summary.create_file_writer(os.path.join(args.logdir, logdir))


def save_models():
    K.models.save_model(generator, os.path.join(save_dir, MODEL, str(epoch + 1), "gen"))
    K.models.save_model(discriminator, os.path.join(save_dir, MODEL, str(epoch + 1), "disc"))
    print("Model at Epoch {}, saved at {}".format(epoch + 1, os.path.join(save_dir, MODEL, str(epoch + 1))))


def backup_and_resume(memory_usage=90):
    """
    Added because tfrecords can cause overflow
    :param memory_usage:
    :return:
    """
    if psutil.virtual_memory().percent > memory_usage:
        save_models()
        if '-l_m' in sys.argv:
            new_arg_index = sys.argv.index('-l_m')
        elif '--load_model' in sys.argv:
            new_arg_index = sys.argv.index('--load_model')
        else:
            new_arg_index = None
        if new_arg_index is None:
            new_args = sys.argv + ['-l_m', os.path.join(save_dir, MODEL, str(epoch + 1))]
        else:
            new_args = sys.argv
            new_args[new_arg_index + 1] = os.path.join(save_dir, MODEL, str(epoch + 1))
        print(f"Restarting with {new_args} to prevent memory overflow")
        os.execv(os.path.abspath(sys.argv[0]), new_args)


c_step = 0

for epoch in range(START_EPOCH, EPOCHS):
    print("\n ----------- Epoch {} --------------\n".format(epoch + 1))
    n = 0
    backup_and_resume(args.memory_limit)  # Check and restart if memory limit is approaching
    # with train_writer.as_default():
    #     tf.summary.scalar("Learning Rate", lrs(epoch).numpy(),
    #                       epoch) if LEARNING_RATE_SCHEDULER != "constant" else tf.summary.scalar("Learning Rate", lrs,
    #                                                                                              epoch)
    with train_writer.as_default():
            tf.summary.scalar("G Learning Rate", g_lrs(epoch).numpy(),
                              c_step) if LEARNING_RATE_SCHEDULER != "constant" \
                else tf.summary.scalar("G Learning Rate",
                                       g_lrs,
                                       c_step)
            tf.summary.scalar("D Learning Rate", d_lrs(epoch).numpy(),
                              c_step) if LEARNING_RATE_SCHEDULER != "constant" \
                else tf.summary.scalar("D Learning Rate",
                                       d_lrs,
                                       c_step)
    for mini_batch in processed_train:
        c_step = (epoch * total_samples // BATCH_SIZE) + n
        g_adv_loss, disc_loss, lm_loss = distributed_train_step(mini_batch)
        print("Epoch {} \t Gen_Adv_Loss: {}, Disc_Loss: {}, LabelMix_Loss: {}, Memory_Usage:{}".format(epoch + 1,
                                                                                                       g_adv_loss,
                                                                                                       disc_loss,
                                                                                                       lm_loss,
                                                                                                       psutil.virtual_memory().percent))
        n += 1
        if n % 20 == 0:
            write_to_tensorboard(g_adv_loss, disc_loss, lm_loss, c_step, train_writer)
    if (epoch + 1) % save_interval == 0:
        save_models()
