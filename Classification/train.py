from models import get_model
import tensorflow.keras as K
import tensorflow_datasets as tfds
import losses
import argparse
import os
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


args = argparse.ArgumentParser(description="Train a network with specific settings")
args.add_argument("--dataset", type=str, default="cifar10", help="Name a dataset from the tf_dataset collection")
args.add_argument("--n_classes", type=int, default=100, help="Number of classes")
args.add_argument("--optimizer", type=str, default="Adam", help="Select optimizer", choices=["SGD", "RMSProp", "Adam"])
args.add_argument("--epochs", type=int, default=40, help="Number of epochs to train")
args.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
args.add_argument("--momentum", type=float, default=0.9, help="Momentum")
args.add_argument("--logging_freq", type=int, default=5, help="Add to tfrecords after this many steps")
args.add_argument("--batch_size", type=int, default=16, help="Size of mini-batch")
args.add_argument("--save_interval", type=int, default=1, help="Save interval for model")
args.add_argument("--model", type=str, default="pr_resnet56", help="Select model")
args.add_argument("--save_dir", type=str, default="./runs", help="Save directory for models and tensorboard")
parsed = args.parse_args()

dataset_name = parsed.dataset
epochs = parsed.epochs
batch_size = parsed.batch_size
n_classes = parsed.n_classes
optimizer_name = parsed.optimizer
lr = parsed.lr
momentum = parsed.momentum
model_name = parsed.model
log_freq = parsed.logging_freq
logdir = os.path.join(parsed.save_dir, "logs/{}_epochs-{}_bs-{}_{}_lr-{}_{}".format(dataset_name, epochs, batch_size,
                                                                        optimizer_name, lr, model_name))
# TODO: Add save option, with a save_dir

# =========== Load Dataset ============ #

dataset = tfds.load(dataset_name, data_dir="/datasets/")
splits = list(dataset.keys())
dataset_train = dataset['train'] if 'train' in splits else None
dataset_test = dataset['test'] if 'test' in splits else None
dataset_validation = dataset['validation'] if 'validation' in splits else None

# =========== Process dataset ============ #
assert dataset_train is not None, "Training dataset can not be None"
assert (dataset_test or dataset_validation) is not None, "Either test or validation dataset should not be None"

total_samples = len(list(dataset_train))
dataset_train = dataset_train.shuffle(1000).batch(batch_size, drop_remainder=True).prefetch(
    tf.data.experimental.AUTOTUNE)
#  TODO: Get dataset shape
dataset_test = dataset_test.shuffle(1024, ).repeat().shuffle(1024, reshuffle_each_iteration=True).batch(batch_size,
                                                                                                        drop_remainder=True) \
    if (dataset_test is not None) else None
dataset_validation = dataset_validation.shuffle(1000).batch(batch_size, drop_remainder=True) \
    if (dataset_validation is not None) else None

eval_dataset = dataset_validation if dataset_validation else dataset_test

# =========== Optimizer and Training Setup ============ #
lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay([50, 32000, 48000, 64000],
                                                                    [lr, lr / 10, lr / 100, lr / 1000, lr / 1e4])
if optimizer_name == "Adam":
    optimizer = K.optimizers.Adam(learning_rate=lr_scheduler)
elif optimizer_name == "RMSProp":
    optimizer = K.optimizers.RMSprop(learning_rate=lr_scheduler, momentum=momentum)
else:
    optimizer = K.optimizers.SGD(learning_rate=lr_scheduler, momentum=momentum)

train_metrics = [tf.keras.metrics.Accuracy()]
val_metrics = [tf.keras.metrics.Accuracy()]
total_steps = 0


def train_step(input_imgs, labels, model, optim):
    """
    Train step for model
    :param input_imgs: Input image tensors
    :param labels: GT labels
    :param model: Keras model to be trained
    :param optim: keras/tf optimizer
    :return: loss value
    """
    with tf.GradientTape() as tape:
        logits = model(input_imgs)
        loss = tf.reduce_mean(K.losses.categorical_crossentropy(labels, logits, from_logits=True))
        # loss = tf.reduce_mean(K.losses.categorical_crossentropy(labels, probs))
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# =========== Training ============ #


model = get_model(model_name, num_classes=n_classes)
train_writer = tf.summary.create_file_writer(logdir + "/train")
val_writer = tf.summary.create_file_writer(logdir + "/validation")
test_writer = tf.summary.create_file_writer(logdir + "/test")

# TODO: Add a proper way of handling logs in absence of validation or test data
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model, iterator=dataset_train)
manager = tf.train.CheckpointManager(ckpt,  logdir + "/models/", max_to_keep=10)

# writer.set_as_default()
step = 0
for epoch in range(epochs):
    for step, (mini_batch, val_mini_batch) in enumerate(zip(dataset_train, dataset_test)):
        train_probs = tf.nn.softmax(model(mini_batch['image'] / 255))
        train_labs = tf.one_hot(mini_batch['label'], n_classes)
        val_probs = tf.nn.softmax(model(val_mini_batch['image'] / 255))
        val_labs = tf.one_hot(val_mini_batch['label'], n_classes)

        loss = train_step(mini_batch['image'] / 255, train_labs, model, optimizer)
        val_loss = losses.get_loss(val_probs,
                                   val_labs,
                                   name='cross_entropy',
                                   from_logits=False)
        print("Epoch {}: {}/{}, Loss: {} Val Loss: {}".format(epoch, step * batch_size, total_samples, loss.numpy(),
                                                              val_loss.numpy()), end='     \r', flush=True)
        curr_step = total_steps + step
        if curr_step % log_freq == 0:
            with train_writer.as_default():
                tf.summary.scalar("loss", loss,
                                  step=curr_step)
            with test_writer.as_default():
                tf.summary.scalar("loss", val_loss,
                                  step=curr_step)
            for t_metric, v_metric in zip(train_metrics, val_metrics):
                _, _ = t_metric.update_state(mini_batch['label'], tf.argmax(train_probs, axis=-1)), \
                       v_metric.update_state(val_mini_batch['label'], tf.argmax(val_probs, axis=-1))
                with train_writer.as_default():
                    tf.summary.scalar(t_metric.name, t_metric.result(), curr_step)
                with test_writer.as_default():
                    tf.summary.scalar(v_metric.name, v_metric.result(), curr_step)
        with train_writer.as_default():
            tmp = lr_scheduler(step=total_steps)
            tf.summary.scalar("Learning Rate", tmp, curr_step)
    total_steps += (step + 1)
    for t_metric, v_metric in zip(train_metrics, val_metrics):
        t_metric.reset_states()
        v_metric.reset_states()
    ckpt.step.assign_add(step + 1)
    if epoch % parsed.save_interval:
        manager.save()
