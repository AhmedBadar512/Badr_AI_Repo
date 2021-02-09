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
args.add_argument("-lrs", "--lr_scheduler", type=str, default="exp_decay", help="Select learning rate scheduler",
                  choices=["poly", "exp_decay", "constant"])
args.add_argument("-gm", "--gan_mode", type=str, default="normal", help="Select training mode for GAN",
                  choices=["normal", "wgan_gp", "ls_gan"])
args.add_argument("-cm", "--cut_mode", type=str, default="cut", help="Select training mode for GAN",
                  choices=["cut", "fastcut"])
args.add_argument("-e", "--epochs", type=int, default=1000, help="Number of epochs to train")
args.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
args.add_argument("--momentum", type=float, default=0.9, help="Momentum")
args.add_argument("-bs", "--batch_size", type=int, default=8, help="Size of mini-batch")
args.add_argument("-si", "--save_interval", type=int, default=5, help="Save interval for model")
args.add_argument("-m", "--model", type=str, default="cut", help="Select model")
args.add_argument("-logs", "--logdir", type=str, default="./logs_cut", help="Directory to save tensorboard logdir")
args.add_argument("-l_m", "--load_model", type=str,
                  default=None,
                  help="Load model from path")
args.add_argument("-s", "--save_dir", type=str, default="./CUT_runs",
                  help="Save directory for models and tensorboard")
args.add_argument("-tfrecs", "--tf_record_path", type=str, default="/data/input/datasets/tf2_gan_tfrecords",
                  help="Save directory that contains train and validation tfrecords")
args.add_argument("-sb", "--shuffle_buffer", type=int, default=1024, help="Size of the shuffle buffer")
args.add_argument("--width", type=int, default=286, help="Size of the shuffle buffer")
args.add_argument("--height", type=int, default=286, help="Size of the shuffle buffer")
args.add_argument("--c_width", type=int, default=256, help="Crop width")
args.add_argument("--c_height", type=int, default=256, help="Crop height")
args = args.parse_args()

tf_record_path = args.tf_record_path
dataset = args.dataset
BUFFER_SIZE = args.shuffle_buffer
BATCH_SIZE = args.batch_size
IMG_WIDTH = args.width
IMG_HEIGHT = args.height
CROP_HEIGHT = args.c_height if args.c_height < IMG_HEIGHT else IMG_HEIGHT
CROP_WIDTH = args.c_width if args.c_width < IMG_WIDTH else IMG_WIDTH
LAMBDA = 1 if args.cut_mode == "cut" else 10
nce_identity = True if args.cut_mode == "cut" else False
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
tf.random.set_seed(128)
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
if num_samples_ab[0] < num_samples_ab[1]:
    total_samples = num_samples_ab[0]
    # train_B = train_B.repeat()
else:
    total_samples = num_samples_ab[1]
    # train_A = train_A.repeat()

augmentor = lambda batch: augment_autoencoder(batch, size=(IMG_HEIGHT, IMG_WIDTH), crop=(CROP_HEIGHT, CROP_WIDTH))
train_A = train_A.map(
    augmentor, num_parallel_calls=tf.data.AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

train_B = train_B.map(
    augmentor, num_parallel_calls=tf.data.AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
train_A = mirrored_strategy.experimental_distribute_dataset(train_A)
train_B = mirrored_strategy.experimental_distribute_dataset(train_B)

if gan_mode == "wgan_gp":
    gan_loss_obj = get_loss(name="Wasserstein")
elif gan_mode == "ls_gan":
    gan_loss_obj = get_loss(name="MSE")
else:
    gan_loss_obj = get_loss(name="binary_crossentropy")
patch_nce_loss = get_loss(name="PatchNCELoss")
id_loss_obj = get_loss(name="MSE")


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


def calc_patch_nce_loss(real_image, fake_img):
    loss1 = patch_nce_loss(real_image, fake_img, encoder, mlp)
    return loss1


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(id_loss_obj(real_image, same_image))
    return LAMBDA * loss


if LEARNING_RATE_SCHEDULER == "poly":
    lrs = K.optimizers.schedules.PolynomialDecay(LEARNING_RATE,
                                                 decay_steps=EPOCHS,
                                                 end_learning_rate=1e-8, power=0.8)
elif LEARNING_RATE_SCHEDULER == "exp_decay":
    lrs = K.optimizers.schedules.ExponentialDecay(LEARNING_RATE,
                                                  decay_steps=1e5,
                                                  decay_rate=0.9)
else:
    lrs = LEARNING_RATE

with mirrored_strategy.scope():
    generator = get_model("{}_gen".format(MODEL), type="gan")
    discriminator = get_model("{}_disc".format(MODEL), type="gan")
    encoder = get_model("{}_enc".format(MODEL), type="gan", gen=generator)
    mlp = get_model("{}_mlp".format(MODEL), type="gan", units=256, num_patches=256)
    tmp = tf.cast(tf.random.uniform((1, CROP_HEIGHT, CROP_WIDTH, 3), dtype=tf.float32, minval=0, maxval=1),
                  dtype=tf.float32)
    generator(tmp), discriminator(tmp), encoder(tmp)
    generator_optimizer = tf.keras.optimizers.Adam(lrs, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(lrs, beta_1=0.5)


def load_models(models_parent_dir):
    assert os.path.exists(models_parent_dir), "The path {} is not valid".format(models_parent_dir)
    p_gen = K.models.load_model(os.path.join(models_parent_dir, "gen"))
    p_disc = K.models.load_model(os.path.join(models_parent_dir, "disc"))
    # p_mlp = K.models.load_model(os.path.join(models_parent_dir, "mlp"))
    generator.set_weights(p_gen.get_weights())
    print("Generator loaded successfully")
    discriminator.set_weights(p_disc.get_weights())
    print("Discriminator loaded successfully")
    # mlp.set_weights(p_mlp.get_weights())
    # print("MLP loaded successfully")


if load_model_path is not None:
    load_models(load_model_path)
    START_EPOCH = int(load_model_path.split("/")[-1])
else:
    START_EPOCH = 0


def write_to_tensorboard(g_loss, d_loss, patch_nce_loss, c_step, writer):
    with writer.as_default():
        tf.summary.scalar("G_Loss", g_loss.numpy(), c_step)
        tf.summary.scalar("D_Loss", tf.reduce_mean(d_loss).numpy(), c_step)
        tf.summary.scalar("PatchNCE_Loss", tf.reduce_mean(patch_nce_loss).numpy(), c_step)
        if len(physical_devices) > 1:
            o_img_a = tf.cast(image_x.values[0], dtype=tf.float32)
            o_img_b = tf.cast(image_y.values[0], dtype=tf.float32)
            img_a, img_b = o_img_a, o_img_b
        else:
            img_a = image_x
            img_b = image_y
        # img_size_a, img_size_b = img_a.shape[1] * img_a.shape[2] * img_a.shape[3], img_b.shape[1] * img_b.shape[2] * \
        #                          img_b.shape[3]
        # mean_a = tf.reduce_mean(img_a, axis=[1, 2, 3], keepdims=True)
        # adjusted_std_a = tf.maximum(tf.math.reduce_std(img_a, axis=[1, 2, 3], keepdims=True),
        #                             1 / tf.sqrt(img_size_a / 1.0))
        # f_image = generator((img_a - mean_a) / adjusted_std_a, training=True)
        f_image = generator(img_a, training=True)
        tf.summary.image("source_img", tf.cast(127.5 * (img_a + 1), dtype=tf.uint8), step=c_step)
        tf.summary.image("target_img", tf.cast(127.5 * (img_b + 1), dtype=tf.uint8), step=c_step)
        tf.summary.image("translated_img", tf.cast((f_image + 1) * 127.5, dtype=tf.uint8), step=c_step)


@tf.function
def train_step(real_x, real_y, n_critic=5):
    # real_x = tf.image.per_image_standardization(real_x)
    # real_x = (real_x / 127.5) - 1
    # real_y = (real_y / 127.5) - 1
    # real_y = tf.image.per_image_standardization(real_y)
    real = tf.concat([real_x, real_y], axis=0) if nce_identity else real_x
    with tf.GradientTape(persistent=True) as tape:
        fake = generator(real, training=True)
        fake_y = fake[:real_x.shape[0]]
        if nce_identity:
            id_y = fake[real_x.shape[0]:]

        # Calculate discriminator losses
        disc_real_y = discriminator(real_y, training=True)
        disc_fake_y = discriminator(fake_y, training=True)
        disc_loss = discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the generator losses
        gen_loss = generator_loss(disc_fake_y)
        patch_nce_loss = calc_patch_nce_loss(real_x, fake_y)
        if nce_identity:
            NCE_B_loss = calc_patch_nce_loss(real_y, id_y)
            patch_nce_loss = (patch_nce_loss + NCE_B_loss) * 0.5

            # Total generator loss = adversarial loss + cycle loss
            total_gen_loss = LAMBDA * patch_nce_loss + identity_loss(real_y, id_y) + gen_loss
        else:
            total_gen_loss = LAMBDA * patch_nce_loss + gen_loss

        if gan_mode != "wgan_gp":
            disc_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # ------------------- Disc Cycle -------------------- #
    if gan_mode == "wgan_gp":
        disc_loss = wgan_disc_apply(fake_y, real_y, n_critic)
    # Calculate the gradients for generator and discriminator
    generator_gradients = tape.gradient(total_gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

    mlp_gradients = tape.gradient(patch_nce_loss, mlp.trainable_variables)
    generator_optimizer.apply_gradients(zip(mlp_gradients, mlp.trainable_variables))
    if gan_mode != "wgan_gp":
        discriminator_gradients = tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss, patch_nce_loss


def wgan_disc_apply(fake_y, real_y, n_critic):
    for _ in range(n_critic):
        with tf.GradientTape(persistent=True) as disc_tape:
            disc_real_y = discriminator(real_y, training=True)
            disc_fake_y = discriminator(fake_y, training=True)
            disc_loss = discriminator_loss(disc_real_y, disc_fake_y) + 10 * gradient_penalty(real_y, fake_y,
                                                                                             discriminator)

        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    return disc_loss


@tf.function
def distributed_train_step(dist_inputs_x, dist_inputs_y):
    per_replica_gen_losses, per_replica_disc_losses, per_replica_patch_nce_loss = \
        mirrored_strategy.run(train_step, args=(dist_inputs_x, dist_inputs_y))
    reduced_gen_g_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                                  per_replica_gen_losses,
                                                  axis=None)
    reduced_disc_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                                 per_replica_disc_losses,
                                                 axis=None)
    reduced_patch_nce_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                                     per_replica_patch_nce_loss,
                                                     axis=None)
    return reduced_gen_g_loss, reduced_disc_loss, reduced_patch_nce_loss


train_writer = tf.summary.create_file_writer(os.path.join(args.logdir, logdir))


def save_models():
    K.models.save_model(generator, os.path.join(save_dir, MODEL, str(epoch + 1), "gen"))
    K.models.save_model(discriminator, os.path.join(save_dir, MODEL, str(epoch + 1), "disc"))
    # K.models.save_model(mlp, os.path.join(save_dir, MODEL, str(epoch + 1), "mlp"))
    print("Model at Epoch {}, saved at {}".format(epoch + 1, os.path.join(save_dir, MODEL, str(epoch + 1))))


for epoch in range(START_EPOCH, EPOCHS):
    print("\n ----------- Epoch {} --------------\n".format(epoch + 1))
    n = 0
    # with train_writer.as_default():
    #     tf.summary.scalar("Learning Rate", lrs(epoch).numpy(),
    #                       epoch) if LEARNING_RATE_SCHEDULER != "constant" else tf.summary.scalar("Learning Rate", lrs,
    #                                                                                              epoch)
    for image_x, image_y in zip(train_A, train_B):
        c_step = (epoch * total_samples // BATCH_SIZE) + n
        gen_loss, disc_loss, patch_nce_loss = distributed_train_step(image_x, image_y)
        print("Epoch {} \t Gen_G_Loss: {}, Disc_Loss: {}, PatchNCELoss: {}".format(epoch + 1, gen_loss, disc_loss, patch_nce_loss))
        n += 1
        with train_writer.as_default():
            tf.summary.scalar("Learning Rate", lrs(c_step).numpy(),
                              c_step) if LEARNING_RATE_SCHEDULER != "constant" else tf.summary.scalar("Learning Rate", lrs,
                                                                                                     c_step)
        if n % 20 == 0:
            write_to_tensorboard(gen_loss, disc_loss, patch_nce_loss,
                                 c_step, train_writer)
    if (epoch + 1) % save_interval == 0:
        save_models()
