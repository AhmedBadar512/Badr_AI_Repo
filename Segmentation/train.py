import tensorflow.keras as K
import losses
import argparse
import os
import tensorflow as tf
import datetime
from citys_visualizer import get_images_custom
from visualization_dicts import gpu_cs_labels
from utils.create_cityscapes_tfrecords import TFRecordsSeg
import string
from model_provider import get_model

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)
print("Physical_Devices: {}".format(physical_devices))

args = argparse.ArgumentParser(description="Train a network with specific settings")
args.add_argument("--dataset", type=str, default="cityscapes19", help="Name a dataset from the tf_dataset collection",
                  choices=["cityscapes", "cityscapes19"])
args.add_argument("--classes", type=int, default=19, help="Number of classes")
args.add_argument("--optimizer", type=str, default="Adam", help="Select optimizer", choices=["SGD", "RMSProp", "Adam"])
args.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
args.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
args.add_argument("--momentum", type=float, default=0.9, help="Momentum")
args.add_argument("--logging_freq", type=int, default=50, help="Add to tfrecords after this many steps")
args.add_argument("--batch_size", type=int, default=4, help="Size of mini-batch")
args.add_argument("--save_interval", type=int, default=1, help="Save interval for model")
args.add_argument("--model", type=str, default="icnet_resnetd50b_cityscapes", help="Select model")
args.add_argument("--save_dir", type=str, default="./runs", help="Save directory for models and tensorboard")
args.add_argument("--shuffle_buffer", type=int, default=10, help="Size of the shuffle buffer")
args.add_argument("--width", type=int, default=512, help="Size of the shuffle buffer")
args.add_argument("--height", type=int, default=512, help="Size of the shuffle buffer")
parsed = args.parse_args()

dataset_name = parsed.dataset
epochs = parsed.epochs
batch_size = parsed.batch_size
classes = parsed.classes
optimizer_name = parsed.optimizer
lr = parsed.lr
momentum = parsed.momentum
model_name = parsed.model
log_freq = parsed.logging_freq
time = str(datetime.datetime.now())
time = time.translate(str.maketrans('', '', string.punctuation)).replace(" ", "-")
logdir = os.path.join(parsed.save_dir, "logs/{}_epochs-{}_bs-{}_{}_lr-{}_{}_{}".format(dataset_name, epochs, batch_size,
                                                                                       optimizer_name, lr, model_name,
                                                                                       time))
# TODO: Add save option, with a save_dir

# =========== Load Dataset ============ #

if dataset_name == "cityscapes19":
    cs_19 = True
    dataset_name = "cityscapes"
else:
    cs_19 = False

dataset_train = TFRecordsSeg(data_dir="/datasets/custom/cityscapes/", tfrecord_path="/volumes1/train.tfrecords",
                             split='train').read_tfrecords()
dataset_validation = TFRecordsSeg(data_dir="/datasets/custom/cityscapes/", tfrecord_path="/volumes1/val.tfrecords",
                                  split='val').read_tfrecords()
# dataset_test = None

# =========== Process dataset ============ #
assert dataset_train is not None, "Training dataset can not be None"
assert dataset_validation is not None, "Either test or validation dataset should not be None"

total_samples = len(list(dataset_train))

dataset_train = dataset_train.shuffle(parsed.shuffle_buffer).batch(batch_size, drop_remainder=True).prefetch(
    tf.data.experimental.AUTOTUNE)
#  TODO: Get dataset shape
dataset_validation = dataset_validation.repeat().shuffle(parsed.shuffle_buffer).batch(batch_size, drop_remainder=True) \
    if (dataset_validation is not None) else None

eval_dataset = dataset_validation

get_images_new = lambda *features: get_images_custom(features, (parsed.height, parsed.width), cs_19)

processed_train = dataset_train.map(get_images_new)
processed_val = dataset_validation.map(get_images_new)
# =========== Optimizer and Training Setup ============ #
# lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay([50, 32000, 48000, 64000],
#                                                                     [lr, lr / 10, lr / 100, lr / 1000, lr / 1e4])
lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr, decay_steps=1e6,
                                                             end_learning_rate=1e-8, power=0.9)
if optimizer_name == "Adam":
    optimizer = K.optimizers.Adam(learning_rate=lr_scheduler)
elif optimizer_name == "RMSProp":
    optimizer = K.optimizers.RMSprop(learning_rate=lr_scheduler, momentum=momentum)
else:
    optimizer = K.optimizers.SGD(learning_rate=lr_scheduler, momentum=momentum)

train_metrics = [tf.keras.metrics.Accuracy()]
val_metrics = [tf.keras.metrics.Accuracy()]
total_steps = 0


def train_step(tape, loss, model, optimizer, filter=None):
    if filter is not None:
        trainable_vars = [var for var in model.trainable_variables if filter in var.name]
    else:
        trainable_vars = model.trainable_variables
    grads = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))

# =========== Training ============ #


model = get_model(model_name, classes=classes, in_size=(parsed.height, parsed.width))
train_writer = tf.summary.create_file_writer(logdir + "/train")
val_writer = tf.summary.create_file_writer(logdir + "/validation")
test_writer = tf.summary.create_file_writer(logdir + "/test")

# TODO: Add a proper way of handling logs in absence of validation or test data
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model, iterator=dataset_train)
manager = tf.train.CheckpointManager(ckpt, logdir + "/models/", max_to_keep=10)
calc_loss = losses.get_loss(name='cross_entropy', from_logits=False)
# writer.set_as_default()
step = 0
for epoch in range(epochs):
    for step, (mini_batch, val_mini_batch) in enumerate(zip(processed_train, processed_val)):
        with tf.GradientTape() as tape:
            train_logits = model(mini_batch[0])[0]
            train_labs = tf.one_hot(mini_batch[1][..., 0], classes)
            loss = calc_loss(train_labs, train_logits)
        val_logits = model(val_mini_batch[0])[0]
        val_labs = tf.one_hot(val_mini_batch[1][..., 0], classes)
        train_step(tape, loss, model, optimizer)
        val_loss = calc_loss(val_labs, val_logits)
        print("Epoch {}: {}/{}, Loss: {} Val Loss: {}".format(epoch, step * batch_size, total_samples, loss.numpy(),
                                                              val_loss.numpy()))
        curr_step = total_steps + step
        if curr_step % log_freq == 0:
            with train_writer.as_default():
                tf.summary.scalar("loss", loss,
                                  step=curr_step)
                tf.summary.image("pred", gpu_cs_labels(tf.argmax(train_logits, axis=-1), cs_19), step=curr_step)
                tf.summary.image("gt", gpu_cs_labels(mini_batch[1][..., tf.newaxis], cs_19), step=curr_step)
            with test_writer.as_default():
                tf.summary.scalar("loss", val_loss,
                                  step=curr_step)
                tf.summary.image("val_pred", gpu_cs_labels(tf.argmax(val_logits, axis=-1), cs_19), step=curr_step)
                tf.summary.image("val_gt", gpu_cs_labels(val_mini_batch[1][..., tf.newaxis], cs_19), step=curr_step)
        with train_writer.as_default():
            tmp = lr_scheduler(step=total_steps)
            tf.summary.scalar("Learning Rate", tmp, curr_step)
    total_steps += (step + 1)
    # for t_metric, v_metric in zip(train_metrics, val_metrics):
    #     t_metric.reset_states()
    #     v_metric.reset_states()
    ckpt.step.assign_add(step + 1)
    if epoch % parsed.save_interval:
        manager.save()
