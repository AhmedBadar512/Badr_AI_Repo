import tensorflow.keras as K


class ConvBNRelu(K.layers.Layer):
    def __init__(self, filters, kernels=(3, 3), strides=1, padding="SAME", data_format="NHWC", dilations=1,
                 use_bias=False):
        super(ConvBNRelu, self).__init__()
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilations = dilations
        self.use_bias = use_bias

    def build(self, inputs_shape):
        if self.data_format == 'NHWC':
            channel_index = -1
            data_format_keras = 'channels_last'
        elif self.data_format == 'NCHW':
            channel_index = 1
            data_format_keras = 'channels_first'
        self.bn = K.layers.BatchNormalization(axis=channel_index)
        self.relu = K.layers.ReLU()
        self.conv = K.layers.Conv2D(filters=self.filters,
                                    kernel_size=self.kernels,
                                    strides=self.strides,
                                    padding=self.padding.lower(),
                                    data_format=data_format_keras,
                                    use_bias=False)

    def call(self, inputs):
        tensor = self.conv(inputs)
        tensor = self.bn(tensor)
        tensor = self.relu(tensor)
        return tensor


class DWConvBNRelu(K.layers.Layer):
    def __init__(self, depth_multiplier=1, kernels=(3, 3), strides=1, padding="SAME", data_format="NHWC", dilations=1,
                 use_bias=False):
        super(DWConvBNRelu, self).__init__()
        self.depth_multiplier = depth_multiplier
        self.kernels = kernels
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilations = dilations
        self.use_bias = use_bias

    def build(self, inputs_shape):
        if self.data_format == 'NHWC':
            channel_index = -1
            data_format_keras = 'channels_last'
        elif self.data_format == 'NCHW':
            channel_index = 1
            data_format_keras = 'channels_first'
        self.bn = K.layers.BatchNormalization(axis=channel_index)
        self.relu = K.layers.ReLU()
        self.conv = K.layers.DepthwiseConv2D(kernel_size=self.kernels,
                                             strides=self.strides,
                                             padding=self.padding.lower(),
                                             data_format=data_format_keras,
                                             depth_multiplier=self.depth_multiplier,
                                             use_bias=False)

    def call(self, inputs):
        tensor = self.conv(inputs)
        tensor = self.bn(tensor)
        tensor = self.relu(tensor)
        return tensor


class CombConv(K.layers.Layer):
    def __init__(self, filters, kernels=(3, 3), strides=1, padding="SAME", data_format="NHWC", dilations=1,
                 use_bias=False):
        super(CombConv, self).__init__()
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilations = dilations
        self.use_bias = use_bias

    def build(self, inputs_shape):
        self.dw = DWConvBNRelu(kernels=self.kernels, strides=self.strides, padding=self.padding,
                               data_format=self.data_format)
        self.pw = ConvBNRelu(self.filters, kernels=(1, 1), padding=self.padding, data_format=self.data_format)

    def call(self, inputs):
        tensor = self.pw(inputs)
        tensor = self.dw(tensor)
        return tensor
