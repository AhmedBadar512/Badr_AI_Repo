import tensorflow as tf
import json
from model_provider import get_model
import cv2
from utils.create_gan_tfrecords import TFRecordsGAN
from utils.augment_images import augment_autoencoder
import os
import tqdm
import time
from losses import get_loss

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
mirrored_strategy = tf.distribute.MirroredStrategy()

tf_record_path = "/data/input/datasets/tf2_gan_tfrecords"
dataset = "zebra2horse"
BUFFER_SIZE = 1000
BATCH_SIZE = 16
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 10
EPOCHS = 200
LEARNING_RATE = 4e-4
checkpoint_path = "./checkpoints/train"
MODEL = "cyclegan"


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

train_A = train_A.map(
    augment_autoencoder, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

train_B = train_B.map(
    augment_autoencoder, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
train_A = mirrored_strategy.experimental_distribute_dataset(train_A)
train_B = mirrored_strategy.experimental_distribute_dataset(train_B)


gan_loss_obj = get_loss(name="binary_crossentropy")
cycle_loss_obj = get_loss(name="MAE")
id_loss_obj = get_loss(name="MAE")


def discriminator_loss(real, generated):
    real_loss = gan_loss_obj(tf.ones_like(real), real)
    generated_loss = gan_loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return tf.reduce_mean(total_disc_loss) * 0.5


def generator_loss(generated):
    return tf.reduce_mean(gan_loss_obj(tf.ones_like(generated), generated))


def calc_cycle_loss(real_image, cycled_image):
    loss1 = cycle_loss_obj(real_image, cycled_image)
    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    loss = id_loss_obj(real_image, same_image)
    return LAMBDA * 0.5 * loss


with mirrored_strategy.scope():
    generator_g = get_model("{}_gen".format(MODEL), type="gan")
    generator_f = get_model("{}_gen".format(MODEL), type="gan")

    discriminator_x = get_model("{}_disc".format(MODEL), type="gan")
    discriminator_y = get_model("{}_disc".format(MODEL), type="gan")
    generator_g_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


def generate_images(model, test_input, name=""):
    prediction = model(test_input)

    collage = tf.concat([test_input, prediction], axis=2)
    collage_list = [x.numpy() for x in collage[:3]]
    os.makedirs("./cycle_tmp", exist_ok=True)
    cv2.imwrite(os.path.join("./cycle_tmp", "epoch_{}.jpg".format(name)),
                (cv2.vconcat(collage_list)[..., ::-1] + 1) * 127.5)


@tf.function
def train_step(real_x, real_y):
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
        total_gen_g_loss = gen_g_loss + LAMBDA * total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + LAMBDA * total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))
    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss


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


# cv2.namedWindow("Test", 0)

for epoch in range(EPOCHS):
    start = time.time()
    print("\n ----------- Epoch {} --------------\n".format(epoch + 1))
    n = 0
    for image_x, image_y in zip(train_A, train_B):
        gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss = distributed_train_step(image_x, image_y)
        print("Epoch {} \t Gen_G_Loss: {}, Gen_F_Loss: {}, Disc_X_Loss: {}, Disc_Y_Loss: {}".format(epoch + 1, gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss))
        n += 1
        if n % 20 == 0:
            generate_images(generator_g, image_x.__dict__['_values'][0], name="{}_{}_ab".format(n, epoch))
            generate_images(generator_f, image_y.__dict__['_values'][0], name="{}_{}_ba".format(n, epoch))
