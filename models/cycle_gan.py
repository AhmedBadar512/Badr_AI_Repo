import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as K


class ConvInActivation(K.layers.Layer):
    """Convolution-Instance Normalization-Activation"""

    def __init__(self, filters=64, kernel_size=1, strides=1, activation=tf.nn.leaky_relu, norm=True):
        super().__init__()
        self.conv = K.layers.Convolution2D(filters, kernel_size, strides, padding="same", use_bias=False, kernel_initializer=tf.random_normal_initializer(0., 0.02))
        if norm:
            self.norm = tfa.layers.InstanceNormalization()
        else:
            self.norm = None
        self.activation = activation

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        if self.norm is not None:
            x = self.norm(x, **kwargs)
        return self.activation(x)


class DeConvInActivation(K.layers.Layer):
    """Convolution-Instance Normalization-Activation"""

    def __init__(self, filters=64, kernel_size=1, strides=1, activation=tf.nn.relu, norm=True):
        super().__init__()
        self.conv = K.layers.Conv2DTranspose(filters, kernel_size, strides, padding="same", use_bias=False, kernel_initializer=tf.random_normal_initializer(0., 0.02))
        if norm:
            self.norm = tfa.layers.InstanceNormalization()
        else:
            self.norm = None
        self.activation = activation

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        if self.norm is not None:
            x = self.norm(x, **kwargs)
        return self.activation(x)

class ResBlock(K.layers.Layer):
    """Convolution-Instance Normalization-Activation"""

    def __init__(self, filters=64, kernel_size=1, strides=1):
        super().__init__()
        self.conv1 = ConvInActivation(filters, kernel_size, strides)
        self.conv2 = ConvInActivation(filters, kernel_size, 1)
        self.conv_res = K.layers.Convolution2D(filters, kernel_size=1, strides=strides, padding="same")

    def call(self, inputs, **kwargs):
        identity = self.conv_res(inputs)
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + identity


class CycleGANGenerator(K.Model):
    def __init__(self,
                 base_channels=64,
                 fixed_size=False,
                 in_size=(512, 512),
                 classes=3,
                 final_activation=None,
                 kernel_size=4,
                 **kwargs):
        super().__init__(**kwargs)
        assert ((in_size[0] % 32 == 0) and (in_size[1] % 32 == 0))
        self.base_channels = base_channels
        self.in_size = in_size
        self.classes = classes
        self.fixed_size = fixed_size
        self.in_conv = ConvInActivation(base_channels, kernel_size, strides=2, norm=False)
        self.conv1 = ConvInActivation(base_channels * 2, kernel_size, strides=2)
        self.conv2 = ConvInActivation(base_channels * 4, kernel_size, strides=2)
        self.conv3 = ConvInActivation(base_channels * 8, kernel_size, strides=2)
        self.conv4 = ConvInActivation(base_channels * 8, kernel_size, strides=2)
        self.conv5 = ConvInActivation(base_channels * 8, kernel_size, strides=2)
        self.conv6 = ConvInActivation(base_channels * 8, kernel_size, strides=2)
        self.conv7 = ConvInActivation(base_channels * 8, kernel_size, strides=2)
        self.up1 = DeConvInActivation(base_channels * 8, kernel_size, 2)
        self.up2 = DeConvInActivation(base_channels * 8, kernel_size, 2)
        self.up3 = DeConvInActivation(base_channels * 8, kernel_size, 2)
        self.up4 = DeConvInActivation(base_channels * 8, kernel_size, 2)
        self.up5 = DeConvInActivation(base_channels * 4, kernel_size, 2)
        self.up6 = DeConvInActivation(base_channels * 2, kernel_size, 2)
        self.up7 = DeConvInActivation(base_channels, kernel_size, 2)
        self.final_conv = K.layers.Conv2DTranspose(classes, 4, strides=2, padding="same", activation=final_activation)

    def call(self, inputs, training=None, mask=None):
        c1 = self.in_conv(inputs, training=training)
        c2 = self.conv1(c1, training=training)
        c3 = self.conv2(c2, training=training)
        c4 = self.conv3(c3, training=training)
        c5 = self.conv4(c4, training=training)
        c6 = self.conv5(c5, training=training)
        c7 = self.conv6(c6, training=training)
        c8 = self.conv7(c7, training=training)

        u1 = self.up1(c8, training=training)
        u1 = tf.concat([u1, c7], axis=-1)
        u2 = self.up2(u1, training=training)
        u2 = tf.concat([u2, c6], axis=-1)
        u3 = self.up3(u2, training=training)
        u3 = tf.concat([u3, c5], axis=-1)
        u4 = self.up4(u3, training=training)
        u4 = tf.concat([u4, c4], axis=-1)
        u5 = self.up5(u4, training=training)
        u5 = tf.concat([u5, c3], axis=-1)
        u6 = self.up6(u5, training=training)
        u6 = tf.concat([u6, c2], axis=-1)
        u7 = self.up7(u6, training=training)
        u7 = tf.concat([u7, c1], axis=-1)
        # final = self.final_conv(tf.concat([u8, inputs], axis=-1))
        final = self.final_conv(u7)
        return final


class ResCycleGANGenerator(K.Model):
    def __init__(self,
                 base_channels=64,
                 final_activation=tf.nn.tanh,
                 classes=3,
                 **kwargs):
        super().__init__(**kwargs)
        self.main_body = K.Sequential(layers=
                                      [ConvInActivation(base_channels, 7, 1),
                                       ConvInActivation(base_channels * 2, 3, 2),
                                       ConvInActivation(base_channels * 4, 3, 2)] +
                                      [ResBlock(base_channels * 4) for _ in range(9)] +
                                      [K.layers.Conv2DTranspose(base_channels * 2, 3, 2, padding="same"),
                                       K.layers.Conv2DTranspose(base_channels, 3, 2, padding="same"),
                                       K.layers.Convolution2D(classes, 7, 1, padding="same", activation=final_activation)
                                       ])

    def call(self, inputs, training=None, mask=None):
        return self.main_body(inputs, training)


class CycleGANDiscriminator(K.Model):
    def __init__(self, base_channels=64):
        super().__init__()
        self.conv1 = ConvInActivation(base_channels, kernel_size=4, strides=2, norm=False)
        self.conv2 = ConvInActivation(base_channels * 2, kernel_size=4, strides=2)
        self.conv3 = ConvInActivation(base_channels * 4, kernel_size=4, strides=2)
        self.conv4 = ConvInActivation(base_channels * 8, kernel_size=4, strides=2)
        self.final_conv = K.layers.Conv2D(1, kernel_size=4)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.final_conv(x, training=training)
        return x


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    x = tf.random.uniform((8, 512, 512, 3))
    gen = ResCycleGANGenerator(32, classes=3)
    disc = CycleGANDiscriminator()
    gen_out = gen(x, False)
    y = disc(gen_out, True)
    print(gen.summary())
    print(gen_out.shape, y.shape)
