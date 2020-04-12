import tensorflow as tf
import tensorflow.keras as K
import numpy as np

batch_size = 1
max_label_seq_length = 255
num_labels = 83
label_length = 5
logit_length = 80
t_label_length = tf.constant([label_length])
t_logit_length = tf.constant([logit_length])

y_true = np.random.randint(0, num_labels, (batch_size, label_length), dtype=np.int32)
y_true = np.pad(y_true, [[0, 0], [0, max_label_seq_length-label_length]])
logits = np.random.rand(batch_size, logit_length, num_labels).astype(np.float32)
my_padding = [[0, 0], [0, logit_length-max_label_seq_length]]
y_tmp = tf.pad(y_true, my_padding)
y_tmp = tf.one_hot(y_tmp, num_labels, dtype=tf.float32)
# y_tmp = tf.nn.softmax(logits, axis=-1)
# y_true = np.tile(y_true, reps=[1, 3])

loss = tf.nn.ctc_loss(y_true, y_tmp, t_label_length, t_logit_length, logits_time_major=False)

# loss = K.backend.ctc_batch_cost(y_true, y_tmp, input_seq_length, label_seq_length)
print('Loss: {}'.format(loss))