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

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
mirrored_strategy = tf.distribute.MirroredStrategy()

args = argparse.ArgumentParser(description="Train a network with specific settings")
args.add_argument("-d", "--dataset", type=str, default="cityscapes19",
                  help="Name a dataset from the tf_dataset collection",
                  choices=["cityscapes19"])
args.add_argument("-c", "--classes", type=int, default=19, help="Number of classes")
args.add_argument("-opt", "--optimizer", type=str, default="Adam", help="Select optimizer",
                  choices=["SGD", "RMSProp", "Adam"])
args.add_argument("-lrs", "--lr_scheduler", type=str, default="exp_decay", help="Select learning rate scheduler",
                  choices=["poly", "exp_decay", "constant"])
args.add_argument("-gm", "--gan_mode", type=str, default="hinge", help="Select training mode for GAN",
                  choices=["normal", "wgan_gp", "ls_gan", "hinge"])
args.add_argument("-cm", "--cut_mode", type=str, default="cut", help="Select training mode for GAN",
                  choices=["cut", "fastcut"])
args.add_argument("-e", "--epochs", type=int, default=1000, help="Number of epochs to train")
args.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
args.add_argument("--momentum", type=float, default=0.9, help="Momentum")
args.add_argument("-bs", "--batch_size", type=int, default=4, help="Size of mini-batch")
args.add_argument("-si", "--save_interval", type=int, default=5, help="Save interval for model")
args.add_argument("-m", "--model", type=str, default="gaugan", help="Select model")
args.add_argument("-logs", "--logdir", type=str, default="./logs_gaugan", help="Directory to save tensorboard logdir")
args.add_argument("-l_m", "--load_model", type=str,
                  default=None,
                  help="Load model from path")
args.add_argument("-s", "--save_dir", type=str, default="./gaugan_runs",
                  help="Save directory for models and tensorboard")
args.add_argument("-tfrecs", "--tf_record_path", type=str, default="/data/input/datasets/tf2_segmentation_tfrecords",
                  help="Save directory that contains train and validation tfrecords")
args.add_argument("-sb", "--shuffle_buffer", type=int, default=128, help="Size of the shuffle buffer")
args.add_argument("--width", type=int, default=584, help="Size of the shuffle buffer")
args.add_argument("--height", type=int, default=286, help="Size of the shuffle buffer")
args.add_argument("--c_width", type=int, default=512, help="Crop width")
args.add_argument("--c_height", type=int, default=256, help="Crop height")
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
LAMBDA_ADV, LAMBDA_VGG, LAMBDA_FEATURE, LAMBDA_KL = 1, 10, 10, 0.05
nce_identity = True if args.cut_mode == "cut" else False
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
tf.random.set_seed(128)
# =========== Load Dataset ============ #

if dataset == "cityscapes19":
    cs_19 = True
    dataset = "cityscapes"
else:
    cs_19 = False
if not cs_19:
    cmap = generate_random_colors(bg_class=args.bg_class)

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

if gan_mode == "hinge":
    gan_loss_obj = get_loss(name="Wasserstein")
elif gan_mode == "wgan_gp":
    gan_loss_obj = get_loss(name="Wasserstein")
elif gan_mode == "ls_gan":
    gan_loss_obj = get_loss(name="MSE")
else:
    gan_loss_obj = get_loss(name="binary_crossentropy")
kl_loss = lambda mean, logvar: 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - 1 - logvar)
feature_loss = get_loss(name="FeatureLoss")


# TODO: Add Regularization loss


def discriminator_loss(real_list, generated_list):
    total_disc_loss = 0
    for real, generated in zip(real_list, generated_list):
        real, generated = real[-1], generated[-1]
        if gan_mode == "wgan_gp":
            real_loss = gan_loss_obj(-tf.ones_like(real), real)
            generated_loss = gan_loss_obj(tf.ones_like(generated), generated)
        elif gan_mode == "hinge":
            real_loss = gan_loss_obj(-tf.ones_like(real), tf.minimum(real - 1, 0.0))
            generated_loss = gan_loss_obj(-tf.ones_like(generated), tf.minimum(-generated - 1, 0.0))
        else:
            real_loss = gan_loss_obj(tf.ones_like(real), real)
            generated_loss = gan_loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss += tf.reduce_mean(generated_loss) + tf.reduce_mean(real_loss)
    return tf.reduce_mean(total_disc_loss) * 0.5 * LAMBDA_ADV


def generator_loss(generated_list):
    total_loss = 0
    for generated in generated_list:
        generated = generated[-1]
        total_loss += (tf.reduce_mean(
            gan_loss_obj(-tf.ones_like(generated), generated)) if gan_mode == "wgan_gp" or gan_mode == "hinge"
                       else tf.reduce_mean(gan_loss_obj(-tf.ones_like(generated), generated)))
    return LAMBDA_ADV * total_loss


def calc_kl_loss(mean, logvar):
    loss = kl_loss(mean, logvar)
    return LAMBDA_KL * loss


def calc_feature_loss(real_features_list, fake_features_list):
    loss = 0
    for real_features, fake_features in zip(real_features_list, fake_features_list):
        loss = feature_loss(real_features, fake_features)
    return LAMBDA_FEATURE * loss


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
    vgg_loss = get_loss(name="VGGLoss")
    tmp = tf.cast(tf.random.uniform((1, CROP_HEIGHT, CROP_WIDTH, 3), dtype=tf.float32, minval=0, maxval=1),
                  dtype=tf.float32)
    tmp1 = tf.cast(tf.random.uniform((1, CROP_HEIGHT, CROP_WIDTH, args.classes), dtype=tf.float32, minval=0, maxval=1),
                   dtype=tf.float32)
    generator = get_model("{}_gen".format(MODEL), type="gan", segmap_shape=tmp1.shape)
    discriminator = get_model("{}_disc".format(MODEL), type="gan")
    encoder = get_model("{}_enc".format(MODEL), type="gan")
    generator.custom_build(input_shape=(1, CROP_HEIGHT, CROP_WIDTH, args.classes))
    enc_out = encoder(tmp)
    generator(enc_out, tmp1), discriminator(tmp, tmp1)
    generator_optimizer = tf.keras.optimizers.Adam(g_lrs, beta_1=0.0, beta_2=0.999)
    discriminator_optimizer = tf.keras.optimizers.Adam(d_lrs, beta_1=0.0, beta_2=0.999)


def calc_vgg_loss(real, fake):
    loss = vgg_loss(real, fake)
    return loss


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
    START_EPOCH = int(load_model_path.split("/")[-1])
else:
    START_EPOCH = 0


def write_to_tensorboard(g_adv_loss, g_kl_loss, g_vgg_loss, g_feautre_loss, disc_loss, c_step, writer):
    with writer.as_default():
        colorize = lambda img: tf.cast(tf.squeeze(gpu_cs_labels(img)), dtype=tf.uint8) if cs_19 else \
            lambda img: tf.cast(tf.squeeze(gpu_random_labels(img, cmap)), dtype=tf.uint8)
        tf.summary.scalar("G_Adv_Loss", g_adv_loss.numpy(), c_step)
        tf.summary.scalar("G_kl_Loss", g_kl_loss.numpy(), c_step)
        tf.summary.scalar("G_VGG_Loss", g_vgg_loss.numpy(), c_step)
        tf.summary.scalar("G_feat_Loss", g_feautre_loss.numpy(), c_step)
        tf.summary.scalar("D_Loss", tf.reduce_mean(disc_loss).numpy(), c_step)
        if len(physical_devices) > 1:
            o_img = tf.cast(tf.concat(mini_batch[0].values, axis=0), dtype=tf.float32) / 127.5 - 1
            o_seg = tf.cast(tf.concat(mini_batch[1].values, axis=0), dtype=tf.int32)
            img, seg = o_img, tf.one_hot(o_seg[..., 0], args.classes)
            processed_labs = tf.concat(mini_batch[1].values, axis=0)
        else:
            img = mini_batch[0] / 127.5 - 1
            seg = tf.one_hot(mini_batch[1][..., 0], args.classes)
            processed_labs = mini_batch[1]
        # img_size_a, img_size_b = img_a.shape[1] * img_a.shape[2] * img_a.shape[3], img_b.shape[1] * img_b.shape[2] * \
        #                          img_b.shape[3]
        # mean_a = tf.reduce_mean(img_a, axis=[1, 2, 3], keepdims=True)
        # adjusted_std_a = tf.maximum(tf.math.reduce_std(img_a, axis=[1, 2, 3], keepdims=True),
        #                             1 / tf.sqrt(img_size_a / 1.0))
        # f_image = generator((img_a - mean_a) / adjusted_std_a, training=True)
        enc_out = encoder(img)
        f_image = generator(enc_out, seg, training=True)
        tf.summary.image("Img", img + 1, step=c_step)
        tf.summary.image("Seg", colorize(processed_labs[..., tf.newaxis]),
                         step=c_step)  # TODO: Add color segmentation here
        tf.summary.image("translated_img", f_image + 1, step=c_step)


def train_step(mini_batch, random_style=False, n_critic=5):
    # real_x = (real_x / 127.5) - 1
    # real_y = (real_y / 127.5) - 1
    img = mini_batch[0] / 127.5 - 1
    seg = tf.one_hot(mini_batch[1][..., 0], args.classes)
    with tf.GradientTape(persistent=True) as tape:
        if random_style:
            enc_vector = tf.random.normal(shape=(args.batch_size, enc_out.shape[-1]))
        else:
            enc_vector, enc_vector_mean, enc_vector_logvar = encoder(img, training=True)
        fake_img = generator(enc_vector, seg)
        disc_real, disc_fake = discriminator(img, seg), discriminator(fake_img, seg)

        # ============ Generator Cycle =============== #
        g_adv_loss = generator_loss(disc_fake)
        g_kl_loss = calc_kl_loss(enc_vector_mean, enc_vector_logvar)
        g_vgg_loss = calc_vgg_loss(img, fake_img)
        g_feautre_loss = feature_loss(disc_real, disc_fake)

        total_gen_loss = g_adv_loss + g_kl_loss + g_vgg_loss + g_feautre_loss

        # =========== Discriminator Cycle ============ #

        if gan_mode != "wgan_gp":
            disc_loss = discriminator_loss(disc_real, disc_fake)

    # ------------------- Disc Cycle -------------------- #
    if gan_mode == "wgan_gp":
        disc_loss = wgan_disc_apply(fake_img, img, n_critic)
    # Calculate the gradients for generator and discriminator
    generator_gradients = tape.gradient(total_gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

    encoder_gradients = tape.gradient(total_gen_loss, encoder.trainable_variables)
    generator_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))

    if gan_mode != "wgan_gp":
        discriminator_gradients = tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return g_adv_loss, g_kl_loss, g_vgg_loss, g_feautre_loss, disc_loss


def wgan_disc_apply(fake, real, n_critic):
    for _ in range(n_critic):
        with tf.GradientTape(persistent=True) as disc_tape:
            disc_real = discriminator(real, training=True)
            disc_fake = discriminator(fake, training=True)
            disc_loss = discriminator_loss(disc_real, disc_fake) + 10 * gradient_penalty(real, fake,
                                                                                         discriminator)

        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    return disc_loss


@tf.function
def distributed_train_step(dist_mini_batch):
    pr_g_adv_losses, pr_g_kl_losses, pr_g_vgg_losses, pr_g_feautre_losses, pr_disc_losses = \
        mirrored_strategy.run(train_step, args=(dist_mini_batch,))
    r_g_adv_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                            pr_g_adv_losses,
                                            axis=None)
    r_g_kl_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                           pr_g_kl_losses,
                                           axis=None)
    r_vgg_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                          pr_g_vgg_losses,
                                          axis=None)
    r_feature_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                              pr_g_feautre_losses,
                                              axis=None)
    r_disc_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                           pr_disc_losses,
                                           axis=None)
    return r_g_adv_loss, r_g_kl_loss, r_vgg_loss, r_feature_loss, r_disc_loss


train_writer = tf.summary.create_file_writer(os.path.join(args.logdir, logdir))


def save_models():
    K.models.save_model(generator, os.path.join(save_dir, MODEL, str(epoch + 1), "gen"))
    K.models.save_model(discriminator, os.path.join(save_dir, MODEL, str(epoch + 1), "disc"))
    K.models.save_model(encoder, os.path.join(save_dir, MODEL, str(epoch + 1), "encoder"))
    print("Model at Epoch {}, saved at {}".format(epoch + 1, os.path.join(save_dir, MODEL, str(epoch + 1))))


for epoch in range(START_EPOCH, EPOCHS):
    print("\n ----------- Epoch {} --------------\n".format(epoch + 1))
    n = 0
    # with train_writer.as_default():
    #     tf.summary.scalar("Learning Rate", lrs(epoch).numpy(),
    #                       epoch) if LEARNING_RATE_SCHEDULER != "constant" else tf.summary.scalar("Learning Rate", lrs,
    #                                                                                              epoch)
    for mini_batch in processed_train:
        c_step = (epoch * total_samples // BATCH_SIZE) + n
        g_adv_loss, g_kl_loss, g_vgg_loss, g_feautre_loss, disc_loss = distributed_train_step(mini_batch)
        print("Epoch {} \t Gen_Adv_Loss: {}, KL_Loss: {}, Gen_VGG_Loss: {}, Gen_Feature_Loss: {}, Disc_Loss: {}".format(
            epoch + 1, g_adv_loss, g_kl_loss, g_vgg_loss, g_feautre_loss, disc_loss))
        n += 1
        with train_writer.as_default():
            tf.summary.scalar("Learning Rate", lrs(c_step).numpy(),
                              c_step) if LEARNING_RATE_SCHEDULER != "constant" else tf.summary.scalar("Learning Rate",
                                                                                                      lrs,
                                                                                                      c_step)
        if n % 20 == 0:
            write_to_tensorboard(g_adv_loss, g_kl_loss, g_vgg_loss, g_feautre_loss, disc_loss,
                                 c_step, train_writer)
    if (epoch + 1) % save_interval == 0:
        save_models()
