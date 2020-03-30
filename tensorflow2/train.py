import tensorflow as tf
from models import *
import tensorflow.keras as K
import tensorflow_datasets as tfds
import losses


dataset_name = "fashion_mnist"
logdir = "/home/badar/Code/Badr_AI_Repo/logs"
epochs = 5
batch_size = 24
n_classes = 10
total_steps = 0
optimizer_name = "Adam"
lr = 1e-4
momentum = 0.9
model_name = "HarDNet"


# =========== Load Dataset ============ #

dataset = tfds.load(dataset_name)
splits = list(dataset.keys())
dataset_train = dataset['train'] if 'train' in splits else None
dataset_test = dataset['test'] if 'test' in splits else None
dataset_validation = dataset['validation'] if 'validation' in splits else None

# =========== Process dataset ============ #
assert dataset_train is not None, "Training dataset can not be None"
assert (dataset_test or dataset_validation) is not None, "Either test or validation dataset should not be None"

dataset_train = dataset_train.shuffle(1000).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
dataset_test = dataset_test.shuffle(1024, ).repeat().shuffle(1024, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True) \
    if (dataset_test is not None) else None
dataset_validation = dataset_validation.shuffle(1000).batch(batch_size, drop_remainder=True) \
    if (dataset_validation is not None) else None

eval_dataset = dataset_validation if dataset_validation else dataset_test

# =========== Optimizer and Training Setup ============ #

if optimizer_name == "Adam":
    optimizer = K.optimizers.Adam(learning_rate=lr)
else:
    optimizer = K.optimizers.SGD(learning_rate=lr, momentum=momentum)


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
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# =========== Training ============ #


model = get_model(model_name, arch=39, depth_wise=True)
writer = tf.summary.create_file_writer(logdir)
writer.set_as_default()

for epoch in range(epochs):
    for step, (mini_batch, val_mini_batch) in enumerate(zip(dataset_train, dataset_test)):
        loss = train_step(tf.cast(mini_batch['image'], tf.float32), tf.one_hot(mini_batch['label'], 10), model, optimizer)
        val_loss = losses.get_loss(model(val_mini_batch['image']/1),
                                   labels=tf.one_hot(mini_batch['label'], 10),
                                   name='cross_entropy',
                                   from_logits=True)
        print("Epoch {}: {}/{}, Loss: {} Val Loss: {}".format(epoch, step*batch_size, 60000, loss.numpy(), val_loss.numpy()))
        tf.summary.scalar("loss", loss,
                          step=total_steps+step)
        tf.summary.scalar("val_loss", val_loss,
                          step=total_steps+step)
    total_steps += (step + 1)