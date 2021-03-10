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
                 sn=False,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.conv2d = K.layers.Conv2D(filters,
                                      kernel_size,
                                      strides,
                                      padding,
                                      use_bias=use_bias,
                                      kernel_initializer=initializer)
        if sn:
            self.conv2d = tfa.layers.SpectralNormalization(self.conv2d)
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


class SPADE(K.layers.Layer):
    def __init__(self, channels, sn=False, activation=None, ks=5):
        super().__init__()
        self.conv1 = K.layers.Conv2D(128, ks, 1, padding="SAME", activation="relu")
        self.conv_gamma = K.layers.Conv2D(channels, ks, 1, padding="SAME")
        self.conv_beta = K.layers.Conv2D(channels, ks, 1, padding="SAME")
        if sn:
            self.conv1 = tfa.layers.SpectralNormalization(self.conv1)
            self.conv_gamma = tfa.layers.SpectralNormalization(self.conv_gamma)
            self.conv_beta = tfa.layers.SpectralNormalization(self.conv_beta)
        self.activation = activation
        # self.segmap_shape = tf.constant(segmap_shape[1:3], dtype=tf.int32)
        # self.avg_pool = lambda feature, strides: K.layers.AveragePooling2D(3, strides=strides, padding="SAME")(feature)

    def build(self, input_shape):
        strides = tf.constant(input_shape[1][1:3])//tf.constant(input_shape[0][1:3])
        self.avg_pool = K.layers.AveragePooling2D(3, strides=strides.numpy(), padding="SAME")

    def call(self, inputs, **kwargs):
        feature, segmap = inputs
        mean, var = tf.nn.moments(feature, axes=[1, 2], keepdims=True)
        x = (feature - mean) / tf.sqrt(var + 1e-5)
        segmap_processed = self.avg_pool(segmap)
        # new_shp = tf.shape(segmap)[1:3] // (tf.shape(segmap)[1:3] // tf.shape(x)[1:3])
        # segmap_processed = tf.image.resize(segmap, new_shp)
        segmap_processed = self.conv1(segmap_processed)
        segmap_gamma = self.conv_gamma(segmap_processed)
        segmap_beta = self.conv_beta(segmap_processed)
        x = x * (1 + segmap_gamma) + segmap_beta
        if self.activation is not None:
            x = self.activation(x)
        return x


class SPADEResBlock(K.Model):
    def __init__(self, channels, sn=False, ks=5):
        super().__init__()
        self.channels = channels
        self.sn = sn
        self.ks = ks

    def build(self, input_shape):
        channels_middle = tf.minimum(self.channels, input_shape[0][-1]).numpy()
        self.spade1 = SPADE(input_shape[0][-1], activation=tf.nn.leaky_relu, sn=False, ks=self.ks)
        self.conv1 = K.layers.Conv2D(channels_middle, 3, 1, padding="SAME")
        if self.sn:
            self.conv1 = tfa.layers.SpectralNormalization(self.conv1)
        self.spade2 = SPADE(channels_middle, activation=tf.nn.leaky_relu, sn=False, ks=self.ks)
        if self.sn:
            self.conv2 = tfa.layers.SpectralNormalization(K.layers.Conv2D(self.channels, 3, 1, padding="SAME"))
        else:
            self.conv2 = K.layers.Conv2D(self.channels, 3, 1, padding="SAME")
        if self.channels != input_shape[0][-1]:
            self.spade3 = SPADE(input_shape[0][-1], sn=False, ks=self.ks)
            if self.sn:
                self.conv3 = tfa.layers.SpectralNormalization(K.layers.Conv2D(self.channels, 3, 1, padding="SAME"))
            else:
                self.conv3 = K.layers.Conv2D(self.channels, 3, 1, padding="SAME")

    def call(self, inputs, training=None, mask=None):
        feature, segmap = inputs
        x = self.spade1((feature, segmap))
        x = self.conv1(x)
        x = self.spade2((x, segmap))
        x = self.conv2(x)
        if self.channels != inputs[0].shape[-1]:
            x_branch = self.spade3((feature, segmap))
            x_branch = self.conv3(x_branch)
            return x_branch + x
        return x + feature
