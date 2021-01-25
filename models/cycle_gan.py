import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as K


class ConvInActivation(K.layers.Layer):
    """Convolution-Instance Normalization-Activation"""

    def __init__(self, filters=64, kernel_size=1, strides=1, activation=tf.nn.leaky_relu, norm=True):
        super().__init__()
        self.conv = K.layers.Convolution2D(filters, kernel_size, strides, padding="same")
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
                 base_channels=32,
                 fixed_size=False,
                 in_size=(512, 512),
                 classes=3,
                 activation=tf.nn.leaky_relu,
                 norm=K.layers.BatchNormalization,
                 final_activation=None,
                 **kwargs):
        super().__init__(**kwargs)
        assert ((in_size[0] % 32 == 0) and (in_size[1] % 32 == 0))
        self.base_channels = base_channels
        self.in_size = in_size
        self.classes = classes
        self.fixed_size = fixed_size
        self.in_conv = K.layers.Conv2D(base_channels,
                                       3,
                                       activation=activation,
                                       padding='same',
                                       kernel_initializer='he_normal')
        self.conv1 = [K.layers.Conv2D(base_channels,
                                      3,
                                      activation=activation,
                                      padding='same',
                                      kernel_initializer='he_normal') for _ in range(3)]
        self.conv2 = [K.layers.Conv2D(base_channels * 2,
                                      3,
                                      activation=activation,
                                      padding='same',
                                      kernel_initializer='he_normal') for _ in range(4)]
        self.conv3 = [K.layers.Conv2D(base_channels * 4,
                                      3,
                                      activation=activation,
                                      padding='same',
                                      kernel_initializer='he_normal') for _ in range(4)]
        self.conv4 = [K.layers.Conv2D(base_channels * 8,
                                      3,
                                      activation=activation,
                                      padding='same',
                                      kernel_initializer='he_normal') for _ in range(4)]
        self.conv5 = [K.layers.Conv2D(base_channels * 16,
                                      3,
                                      activation=activation,
                                      padding='same',
                                      kernel_initializer='he_normal') for _ in range(2)]
        self.dropout1 = K.layers.Dropout(0.1)
        self.dropout2 = K.layers.Dropout(0.2)
        self.dropout3 = K.layers.Dropout(0.3)
        self.pool = K.layers.MaxPool2D(pool_size=(2, 2))
        self.bn = []
        for _ in range(18):
            self.bn.append(norm())
        self.up6 = K.layers.Conv2DTranspose(base_channels * 8, (3, 3), strides=(2, 2), padding="same")
        self.up7 = K.layers.Conv2DTranspose(base_channels * 4, (3, 3), strides=(2, 2), padding="same")
        self.up8 = K.layers.Conv2DTranspose(base_channels * 2, (3, 3), strides=(2, 2), padding="same")
        self.up9 = K.layers.Conv2DTranspose(base_channels, (3, 3), strides=(2, 2), padding="same")
        self.conv10 = K.layers.Conv2D(classes, 1, strides=1, padding="same", activation=final_activation)

    def call(self, inputs, training=None, mask=None):
        c1 = self.in_conv(inputs)
        c1 = self.bn[0](c1, training=training)
        c1 = self.dropout1(c1)
        c1 = self.conv1[0](c1)
        c1 = self.bn[1](c1, training=training)
        p1 = self.pool(c1)

        c2 = self.conv2[0](p1)
        c2 = self.bn[2](c2, training=training)
        c2 = self.dropout1(c2)
        c2 = self.conv2[1](c2)
        c2 = self.bn[3](c2, training=training)
        p2 = self.pool(c2)

        c3 = self.conv3[0](p2)
        c3 = self.bn[4](c3, training=training)
        c3 = self.dropout2(c3)
        c3 = self.conv3[1](c3)
        c3 = self.bn[5](c3, training=training)
        p3 = self.pool(c3)

        c4 = self.conv4[0](p3)
        c4 = self.bn[6](c4, training=training)
        c4 = self.dropout2(c4)
        c4 = self.conv4[1](c4)
        c4 = self.bn[7](c4, training=training)
        p4 = self.pool(c4)

        c5 = self.conv5[0](p4)
        c5 = self.bn[8](c5, training=training)
        c5 = self.dropout3(c5)
        c5 = self.conv5[1](c5)
        c5 = self.bn[9](c5, training=training)

        u6 = self.up6(c5)
        u6 = K.layers.concatenate([u6, c4])
        c6 = self.conv4[2](u6)
        c6 = self.bn[10](c6, training=training)
        c6 = self.dropout2(c6)
        c6 = self.conv4[3](c6)
        c6 = self.bn[11](c6, training=training)

        u7 = self.up7(c6)
        u7 = K.layers.concatenate([u7, c3])
        c7 = self.conv3[2](u7)
        c7 = self.bn[12](c7, training=training)
        c7 = self.dropout2(c7)
        c7 = self.conv3[3](c7)
        c7 = self.bn[13](c7, training=training)

        u8 = self.up8(c7)
        u8 = K.layers.concatenate([u8, c2])
        c8 = self.conv2[2](u8)
        c8 = self.bn[14](c8, training=training)
        c8 = self.dropout1(c8)
        c8 = self.conv2[3](c8)
        c8 = self.bn[15](c8, training=training)

        u9 = self.up9(c8)
        u9 = K.layers.concatenate([u9, c1])
        c9 = self.conv1[1](u9)
        c9 = self.bn[16](c9, training=training)
        c9 = self.dropout1(c9)
        c9 = self.conv1[2](c9)
        c9 = self.bn[17](c9, training=training)

        final = self.conv10(c9)
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
        self.final_conv = K.layers.Conv2D(1, kernel_size=1, activation="sigmoid")

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
