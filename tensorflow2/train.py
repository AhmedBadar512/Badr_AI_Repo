import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import tensorflow_datasets as tfds


dataset_name = "food101"
logdir = "/home/badar/Code/Badr_AI_Repo/logs"
epochs = 5
batch_size = 24
n_classes = 10
total_steps = 0
optimizer_name = "Adam"


# =========== Load Dataset ============ #

dataset = tfds.load(dataset_name)
splits = list(dataset.keys())
dataset_train = dataset['train'] if 'train' in splits else None
dataset_test = dataset['test'] if 'test' in splits else None
dataset_validation = dataset['validation'] if 'validation' in splits else None

# =========== Process dataset ============ #
assert dataset_train is not None, "Training dataset can not be None"
assert (dataset_test or dataset_validation) is not None, "Either test or validation dataset should not be None"

dataset_train = dataset_train.shuffle(1000).batch(batch_size, drop_remainder=True)
dataset_test = dataset_test.shuffle(1000).batch(batch_size, drop_remainder=True) \
    if (dataset_test is not None) else None
dataset_validation = dataset_validation.shuffle(1000).batch(batch_size, drop_remainder=True) \
    if (dataset_validation is not None) else None

eval_dataset = dataset_validation if dataset_validation else dataset_test

# =========== Optimizer and Training Setup ============ #