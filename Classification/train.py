import tensorflow as tf
from models import *
import tensorflow.keras as K
import tensorflow_datasets as tfds
import losses


dataset_name = "cifar100"
logdir = "cifar100_base_test"
epochs = 200
batch_size = 128
n_classes = 100
total_steps = 0
optimizer_name = "SGD"
lr = 0.01
momentum = 0.9
model_name = "resnet20"
log_freq = 5
# TODO: Create log names based on experiment settings
# TODO: Add save option, with a save_dir
# TODO: Create args for console arguments

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
lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay([50, 32000, 48000, 64000], [lr, lr/10, lr/100, lr/1000, lr/1e4])
if optimizer_name == "Adam":
    optimizer = K.optimizers.Adam(learning_rate=lr_scheduler)
elif optimizer_name == "RMSProp":
    optimizer = K.optimizers.RMSprop(learning_rate=lr_scheduler, momentum=momentum)
else:
    optimizer = K.optimizers.SGD(learning_rate=lr_scheduler, momentum=momentum)

train_metrics = [tf.keras.metrics.Accuracy()]
val_metrics = [tf.keras.metrics.Accuracy()]


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

# writer.set_as_default()
for epoch in range(epochs):
    for step, (mini_batch, val_mini_batch) in enumerate(zip(dataset_train, dataset_test)):
        train_probs = tf.nn.softmax(model(mini_batch['image']/255))
        train_labs = tf.one_hot(mini_batch['label'], n_classes)
        val_probs = tf.nn.softmax(model(val_mini_batch['image']/255))
        val_labs = tf.one_hot(val_mini_batch['label'], n_classes)

        loss = train_step(mini_batch['image']/255, train_labs, model, optimizer)
        val_loss = losses.get_loss(val_probs,
                                   val_labs,
                                   name='cross_entropy',
                                   from_logits=False)
        print("Epoch {}: {}/{}, Loss: {} Val Loss: {}".format(epoch, step*batch_size, 60000, loss.numpy(), val_loss.numpy()), end='     \r', flush=True)
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