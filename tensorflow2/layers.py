import tensorflow as tf
import tensorflow.keras as K


class customSeperableConv(K.layers.Layer):
    def __init__(self, filters, kernels, strides, padding="SAME", data_format="NHWC", dilations=1):
        super(customSeperableConv, self).__init__()
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilations = dilations

    def build(self, inputs_shape):
        if self.data_format == 'NHWC':
            self.strides = [1, self.strides, self.strides, 1]
            channel_index = -1
        else:
            self.strides = [1, 1, self.strides, self.strides]
            channel_index = 1
        self.w_dw = self.add_variable(name="dw_conv_w", shape=(self.kernels[0], self.kernels[1], inputs_shape[channel_index], 1))
        self.w_pw = self.add_variable(name="dw_conv_w", shape=(1, 1, inputs_shape[channel_index], self.filters))

    def call(self, inputs):
         tensor = tf.nn.depthwise_conv2d(inputs,
                                         self.w_dw,
                                         strides=self.strides,
                                         padding=self.padding,
                                         data_format=self.data_format)
         tensor = tf.nn.conv2d(tensor,
                               filters=self.w_pw,
                               strides=1,
                               padding=self.padding,
                               data_format=self.data_format)
         return tensor


myLayer = customSeperableConv(10, (3, 3), 2, data_format="NHWC")
import numpy as np

x = np.random.rand(1, 256, 256, 3)
print(myLayer(x).shape)