import tensorflow as tf
import tensorflow.keras as K
import tensorflow_addons as tfa


class Padding2D(K.layers.Layer):
    """ 2D padding layer.
    """
    def __init__(self, padding=(1, 1), pad_type='constant', **kwargs):
        assert pad_type in ['constant', 'reflect', 'symmetric']
        super(Padding2D, self).__init__(**kwargs)
        self.padding = (padding, padding) if type(padding) is int else tuple(padding)
        self.pad_type = pad_type

    def call(self, inputs, training=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]

        return tf.pad(inputs, padding_tensor, mode=self.pad_type)


class ConvBlock(K.layers.Layer):
    """ ConBlock layer consists of Conv2D + Normalization + Activation.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 use_bias=True,
                 norm_layer=None,
                 activation='linear',
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.conv2d = K.layers.Conv2D(filters,
                                      kernel_size,
                                      strides,
                                      padding,
                                      use_bias=use_bias,
                                      kernel_initializer=initializer)
        self.activation = K.layers.Activation(activation)
        if norm_layer == 'batch':
            self.normalization = K.layers.BatchNormalization()
        elif norm_layer == 'instance':
            self.normalization = tfa.layers.InstanceNormalization()
        else:
            self.normalization = tf.identity

    def call(self, inputs, training=None):
        x = self.conv2d(inputs)
        x = self.normalization(x)
        x = self.activation(x)

        return x


class ConvTransposeBlock(K.layers.Layer):
    """ ConvTransposeBlock layer consists of Conv2DTranspose + Normalization + Activation.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 use_bias=True,
                 norm_layer=None,
                 activation='linear',
                 **kwargs):
        super(ConvTransposeBlock, self).__init__(**kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.convT2d = K.layers.Conv2DTranspose(filters,
                                                kernel_size,
                                                strides,
                                                padding,
                                                use_bias=use_bias,
                                                kernel_initializer=initializer)
        self.activation = K.layers.Activation(activation)
        if norm_layer == 'batch':
            self.normalization = K.layers.BatchNormalization()
        elif norm_layer == 'instance':
            self.normalization = tfa.layers.InstanceNormalization()
        else:
            self.normalization = tf.identity

    def call(self, inputs, training=None):
        x = self.convT2d(inputs)
        x = self.normalization(x)
        x = self.activation(x)

        return x


class ResBlock(K.layers.Layer):
    """ ResBlock is a ConvBlock with skip connections.
    Original Resnet paper (https://arxiv.org/pdf/1512.03385.pdf).
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 use_bias,
                 norm_layer,
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)

        self.reflect_pad1 = Padding2D(1, pad_type='reflect')
        self.conv_block1 = ConvBlock(filters,
                                     kernel_size,
                                     padding='valid',
                                     use_bias=use_bias,
                                     norm_layer=norm_layer,
                                     activation='relu')

        self.reflect_pad2 = Padding2D(1, pad_type='reflect')
        self.conv_block2 = ConvBlock(filters,
                                     kernel_size,
                                     padding='valid',
                                     use_bias=use_bias,
                                     norm_layer=norm_layer)

    def call(self, inputs, training=None):
        x = self.reflect_pad1(inputs)
        x = self.conv_block1(x)

        x = self.reflect_pad2(x)
        x = self.conv_block2(x)

        return inputs + x
