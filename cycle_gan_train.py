import tensorflow as tf
import json
from model_provider import get_model
from utils.create_gan_tfrecords import TFRecordsGAN
from utils.augment_images import augment_autoencoder
import os
import tensorflow.keras as K
import datetime
import string
from losses import get_loss, gradient_penalty
import argparse

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
mirrored_strategy = tf.distribute.MirroredStrategy()

args = argparse.ArgumentParser(description="Train a network with specific settings")
args.add_argument("-d", "--dataset", type=str, default="zebra2horse",
                  help="Name a dataset from the tf_dataset collection",
                  choices=["zebra2horse"])
args.add_argument("-opt", "--optimizer", type=str, default="Adam", help="Select optimizer",
                  choices=["SGD", "RMSProp", "Adam"])
args.add_argument("-lrs", "--lr_scheduler", type=str, default="constant", help="Select learning rate scheduler",
                  choices=["poly", "exp_decay", "constant"])
args.add_argument("-gm", "--gan_mode", type=str, default="constant", help="Select training mode for GAN",
                  choices=["normal", "wgan_gp"])
args.add_argument("-e", "--epochs", type=int, default=1000, help="Number of epochs to train")
args.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
args.add_argument("--momentum", type=float, default=0.9, help="Momentum")
args.add_argument("-bs", "--batch_size", type=int, default=16, help="Size of mini-batch")
args.add_argument("-si", "--save_interval", type=int, default=5, help="Save interval for model")
args.add_argument("-m", "--model", type=str, default="cyclegan", help="Select model")
args.add_argument("-logs", "--logdir", type=str, default="./logs", help="Directory to save tensorboard logdir")
args.add_argument("-l_m", "--load_model", type=str,
                  default=None,
                  help="Load model from path")
args.add_argument("-s", "--save_dir", type=str, default="./cyclegan_runs",
                  help="Save directory for models and tensorboard")
args.add_argument("-tfrecs", "--tf_record_path", type=str, default="/data/input/datasets/tf2_gan_tfrecords",
                  help="Save directory that contains train and validation tfrecords")
args.add_argument("-sb", "--shuffle_buffer", type=int, default=1024, help="Size of the shuffle buffer")
args.add_argument("--width", type=int, default=286, help="Size of the shuffle buffer")
args.add_argument("--height", type=int, default=286, help="Size of the shuffle buffer")
args.add_argument("--c_width", type=int, default=256, help="Crop width")
args.add_argument("--c_height", type=int, default=256, help="Crop height")
args.add_argument("--random_seed", type=int, default=1, help="Set random seed to this if true")
args = args.parse_args()

tf_record_path = args.tf_record_path
dataset = args.dataset
BUFFER_SIZE = args.shuffle_buffer
BATCH_SIZE = args.batch_size
IMG_WIDTH = args.width
IMG_HEIGHT = args.height
CROP_HEIGHT = args.c_height if args.c_height < IMG_HEIGHT else IMG_HEIGHT
CROP_WIDTH = args.c_width if args.c_width < IMG_WIDTH else IMG_WIDTH
LAMBDA = 10
EPOCHS = args.epochs
LEARNING_RATE = args.lr
LEARNING_RATE_SCHEDULER = args.lr_scheduler
save_interval = args.save_interval
save_dir = args.save_dir
load_model_path = args.load_model
MODEL = args.model
gan_mode = args.gan_mode
time = str(datetime.datetime.now())
time = time.translate(str.maketrans('', '', string.punctuation)).replace(" ", "-")[:-8]
logdir = "{}_{}_e{}_lr{}_{}x{}_{}".format(time, MODEL, EPOCHS, LEARNING_RATE, IMG_HEIGHT, IMG_WIDTH, gan_mode)

train_A, train_B = \
    TFRecordsGAN(
        tfrecord_path=
        "{}/{}_train.tfrecords".format(tf_record_path, dataset + "_a")).read_tfrecords(), \
    TFRecordsGAN(
        tfrecord_path=
        "{}/{}_train.tfrecords".format(tf_record_path, dataset + "_b")).read_tfrecords()

with open("/data/input/datasets/tf2_gan_tfrecords/data_samples.json") as f:
    data = json.load(f)
num_samples_ab = [data[dataset + "_a"], data[dataset + "_b"]]
if num_samples_ab[0] > num_samples_ab[1]:
    total_samples = num_samples_ab[0]
    train_B = train_B.repeat()
else:
    total_samples = num_samples_ab[1]
    train_A = train_A.repeat()

augmentor = lambda batch: augment_autoencoder(batch, size=(IMG_HEIGHT, IMG_WIDTH), crop=(CROP_HEIGHT, CROP_WIDTH))
train_A = train_A.map(
    augmentor, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

train_B = train_B.map(
    augmentor, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
train_A = mirrored_strategy.experimental_distribute_dataset(train_A)
train_B = mirrored_strategy.experimental_distribute_dataset(train_B)

if gan_mode == "wgan_gp":
    gan_loss_obj = get_loss(name="Wasserstein")
else:
    gan_loss_obj = get_loss(name="binary_crossentropy")
cycle_loss_obj = get_loss(name="MAE")
id_loss_obj = get_loss(name="MAE")


def discriminator_loss(real, generated):
    if gan_mode == "wgan_gp":
        real_loss = gan_loss_obj(-tf.ones_like(real), real)
        generated_loss = gan_loss_obj(tf.ones_like(generated), generated)
    else:
        real_loss = gan_loss_obj(tf.ones_like(real), real)
        generated_loss = gan_loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = generated_loss + real_loss
    return tf.reduce_mean(total_disc_loss) * 0.5


def generator_loss(generated):
    return tf.reduce_mean(
        gan_loss_obj(-tf.ones_like(generated), generated)) if gan_mode == "wgan_gp" else tf.reduce_mean(
        gan_loss_obj(tf.ones_like(generated), generated))


def calc_cycle_loss(real_image, cycled_image):
    loss1 = cycle_loss_obj(real_image, cycled_image)
    return loss1


def identity_loss(real_image, same_image):
    loss = id_loss_obj(real_image, same_image)
    return LAMBDA * 0.5 * loss


if LEARNING_RATE_SCHEDULER == "poly":
    lrs = K.optimizers.schedules.PolynomialDecay(LEARNING_RATE,
                                                 decay_steps=EPOCHS,
                                                 end_learning_rate=1e-8, power=0.8)
elif LEARNING_RATE_SCHEDULER == "exp_decay":
    lrs = K.optimizers.schedules.ExponentialDecay(LEARNING_RATE,
                                                  decay_steps=EPOCHS,
                                                  decay_rate=0.5)
else:
    lrs = LEARNING_RATE

with mirrored_strategy.scope():
    generator_g = get_model("{}_gen".format(MODEL), type="gan")
    generator_f = get_model("{}_gen".format(MODEL), type="gan")

    discriminator_x = get_model("{}_disc".format(MODEL), type="gan")
    discriminator_y = get_model("{}_disc".format(MODEL), type="gan")
    tmp = tf.cast(tf.random.uniform((1, CROP_HEIGHT, CROP_WIDTH, 3), dtype=tf.float32, minval=0, maxval=1),
                  dtype=tf.float32)
    generator_g(tmp), generator_f(tmp), discriminator_x(tmp), discriminator_y(tmp)
    generator_g_optimizer = tf.keras.optimizers.Adam(lrs, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(lrs, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(lrs, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(lrs, beta_1=0.5)


def load_models(models_parent_dir):
    assert os.path.exists(models_parent_dir), "The path {} is not valid".format(models_parent_dir)
    p_gen_g = K.models.load_model(os.path.join(models_parent_dir, "gen_g"))
    p_gen_f = K.models.load_model(os.path.join(models_parent_dir, "gen_f"))
    p_disc_x = K.models.load_model(os.path.join(models_parent_dir, "disc_x"))
    p_disc_y = K.models.load_model(os.path.join(models_parent_dir, "disc_y"))
    generator_g.set_weights(p_gen_g.get_weights())
    print("Generator G loaded successfully")
    generator_f.set_weights(p_gen_f.get_weights())
    print("Generator F loaded successfully")
    discriminator_x.set_weights(p_disc_x.get_weights())
    print("Discriminator X loaded successfully")
    discriminator_y.set_weights(p_disc_y.get_weights())
    print("Discriminator Y loaded successfully")


if load_model_path is not None:
    load_models(load_model_path)
    START_EPOCH = int(load_model_path.split("/")[-1])
else:
    START_EPOCH = 0


def write_to_tensorboard(g_loss_g, g_loss_f, d_loss_x, d_loss_y, c_step, writer):
    with writer.as_default():
        tf.summary.scalar("G_Loss_G", g_loss_g.numpy(), c_step)
        tf.summary.scalar("G_Loss_F", g_loss_f.numpy(), c_step)
        tf.summary.scalar("D_Loss_X", tf.reduce_mean(d_loss_x).numpy(), c_step)
        tf.summary.scalar("D_Loss_Y", tf.reduce_mean(d_loss_y).numpy(), c_step)
        if len(physical_devices) > 1:
            o_img_a = tf.cast(image_x.values[0], dtype=tf.float32)
            o_img_b = tf.cast(image_y.values[0], dtype=tf.float32)
            img_a, img_b = o_img_a, o_img_b
        else:
            img_a = image_x
            img_b = image_y
        img_size_a, img_size_b = img_a.shape[1] * img_a.shape[2] * img_a.shape[3], img_b.shape[1] * img_b.shape[2] * \
                                 img_b.shape[3]
        mean_a, mean_b = tf.reduce_mean(img_a, axis=[1, 2, 3], keepdims=True), tf.reduce_mean(img_b, axis=[1, 2, 3],
                                                                                              keepdims=True)
        adjusted_std_a = tf.maximum(tf.math.reduce_std(img_a, axis=[1, 2, 3], keepdims=True),
                                    1 / tf.sqrt(img_size_a / 1.0))
        adjusted_std_b = tf.maximum(tf.math.reduce_std(img_b, axis=[1, 2, 3], keepdims=True),
                                    1 / tf.sqrt(img_size_b / 1.0))
        f_image_y = generator_g((img_a - mean_a) / adjusted_std_a, training=True)
        f_image_x = generator_f((img_b - mean_b) / adjusted_std_b, training=True)
        confidence_a = discriminator_x(f_image_x, training=True)
        confidence_b = discriminator_y(f_image_y, training=True)
        tf.summary.image("img_a", img_a, step=c_step)
        tf.summary.image("img_b", img_b, step=c_step)
        tf.summary.image("fake_img_a", (f_image_x * adjusted_std_b) + mean_b, step=c_step)
        tf.summary.image("fake_img_b", (f_image_y * adjusted_std_a) + mean_a, step=c_step)
        tf.summary.image("confidence_a", confidence_a, step=c_step)
        tf.summary.image("confidence_b", confidence_b, step=c_step)


@tf.function
def train_step(real_x, real_y, n_critic=5):
    real_x = tf.image.per_image_standardization(real_x)
    real_y = tf.image.per_image_standardization(real_y)
    with tf.GradientTape(persistent=True) as tape:
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = LAMBDA * total_cycle_loss + identity_loss(real_y, same_y) + gen_g_loss
        total_gen_f_loss = LAMBDA * total_cycle_loss + identity_loss(real_x, same_x) + gen_f_loss

        if gan_mode != "wgan_gp":
            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # ------------------- Disc Cycle -------------------- #
    if gan_mode == "wgan_gp":
        disc_x_loss, disc_y_loss = wgan_disc_apply(fake_x, fake_y, n_critic, real_x, real_y)
    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)
    if gan_mode != "wgan_gp":
        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))
    if gan_mode != "wgan_gp":
        discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))

        discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                      discriminator_y.trainable_variables))

    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss


def wgan_disc_apply(fake_x, fake_y, n_critic, real_x, real_y):
    for _ in range(n_critic):
        with tf.GradientTape(persistent=True) as disc_tape:
            disc_real_x = discriminator_x(real_x, training=True)
            disc_real_y = discriminator_y(real_y, training=True)

            disc_fake_x = discriminator_x(fake_x, training=True)
            disc_fake_y = discriminator_y(fake_y, training=True)
            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x) + 10 * gradient_penalty(real_x, fake_x,
                                                                                               discriminator_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y) + 10 * gradient_penalty(real_y, fake_y,
                                                                                               discriminator_y)

        discriminator_x_gradients = disc_tape.gradient(disc_x_loss,
                                                       discriminator_x.trainable_variables)
        discriminator_y_gradients = disc_tape.gradient(disc_y_loss,
                                                       discriminator_y.trainable_variables)
        discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                      discriminator_x.trainable_variables))

        discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                      discriminator_y.trainable_variables))
    return disc_x_loss, disc_y_loss


@tf.function
def distributed_train_step(dist_inputs_a, dist_inputs_b):
    per_replica_gen_g_losses, per_replica_gen_f_losses, per_replica_disc_x_losses, per_replica_disc_y_losses = \
        mirrored_strategy.run(train_step, args=(dist_inputs_a, dist_inputs_b))
    reduced_gen_g_loss, reduced_gen_f_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                                                      per_replica_gen_g_losses,
                                                                      axis=None), mirrored_strategy.reduce(
        tf.distribute.ReduceOp.MEAN, per_replica_gen_f_losses,
        axis=None)
    reduced_disc_x_loss, reduced_disc_y_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                                                        per_replica_disc_x_losses,
                                                                        axis=None), mirrored_strategy.reduce(
        tf.distribute.ReduceOp.MEAN, per_replica_disc_y_losses,
        axis=None)
    return reduced_gen_g_loss, reduced_gen_f_loss, reduced_disc_x_loss, reduced_disc_y_loss


train_writer = tf.summary.create_file_writer(os.path.join(args.logdir, logdir))


def save_models():
    K.models.save_model(generator_g, os.path.join(save_dir, MODEL, str(epoch + 1), "gen_g"))
    K.models.save_model(generator_f, os.path.join(save_dir, MODEL, str(epoch + 1), "gen_f"))
    K.models.save_model(discriminator_x, os.path.join(save_dir, MODEL, str(epoch + 1), "disc_x"))
    K.models.save_model(discriminator_y, os.path.join(save_dir, MODEL, str(epoch + 1), "disc_y"))
    print("Model at Epoch {}, saved at {}".format(epoch, os.path.join(save_dir, MODEL, str(epoch))))


for epoch in range(START_EPOCH, EPOCHS):
    print("\n ----------- Epoch {} --------------\n".format(epoch + 1))
    n = 0
    with train_writer.as_default():
        tf.summary.scalar("Learning Rate", lrs(epoch).numpy(),
                          epoch) if LEARNING_RATE_SCHEDULER != "constant" else tf.summary.scalar("Learning Rate", lrs,
                                                                                                 epoch)
    for image_x, image_y in zip(train_A, train_B):
        gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss = distributed_train_step(image_x, image_y)
        print(
            "Epoch {} \t Gen_G_Loss: {}, Gen_F_Loss: {}, Disc_X_Loss: {}, Disc_Y_Loss: {}".format(epoch + 1, gen_g_loss,
                                                                                                  gen_f_loss,
                                                                                                  disc_x_loss,
                                                                                                  disc_y_loss))
        n += 1
        if n % 20 == 0:
            write_to_tensorboard(gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss,
                                 (epoch * total_samples // BATCH_SIZE) + n, train_writer)
    if (epoch + 1) % save_interval == 0:
        save_models()
