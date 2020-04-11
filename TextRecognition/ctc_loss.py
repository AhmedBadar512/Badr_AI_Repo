import tensorflow as tf
import tensorflow.keras as K
import numpy as np

# y_true = np.random.randint(0, 83, (1, 38), dtype=np.int32)
# y_true = np.pad(y_true, [[0, 0], [0, 42]])
# # logits = np.random.rand(1, 255, 83).astype(np.float32)
# my_padding = [[0, 0], [0, 175]]
# y_tmp = tf.pad(y_true, my_padding)
# y_tmp = tf.one_hot(y_tmp, 84, dtype=tf.float32)
# # y_pred = tf.nn.softmax(logits, axis=-1)
# input_seq_length = [[255]]
# label_seq_length = [[38]]
#
# # loss = K.backend.ctc_batch_cost(y_true, y_tmp, input_seq_length, label_seq_length)
# loss = tf.nn.ctc_loss(y_true, y_tmp, input_seq_length, label_seq_length)
# print('Loss: {}'.format(loss))