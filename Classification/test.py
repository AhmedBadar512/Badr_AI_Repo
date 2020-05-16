import numpy as np
from models import *


model = get_model("resnet56", num_classes=100)
model.build(input_shape=(None, None, None, 3))
ckpt_test = tf.train.Checkpoint(net=model)
ckpt_test.restore("logs/cifar100_epochs-1_bs-16_Adam_lr-0.0001_resnet56/models/ckpt-1").expect_partial()
print("Model restored!")