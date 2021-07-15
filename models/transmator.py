""" Implement the following components that are used in GauGAN model.
Generator (Resnet-based)
Discriminator (PatchGAN)
Encoder
"""

import tensorflow as tf
import tensorflow_addons as tfa

import tensorflow.keras as K
from tensorflow.keras.layers import Dense
from .layers import ConvBlock, Padding2D, SPADEResBlock
import tensorflow_probability as tfp


# class GAUEncoder(K.Model):
#     def __init__(self, channels=64, norm_layer="instance"):
#         super().__init__()
#         use_bias = norm_layer == "instance"
#         self.norm_layer = norm_layer
#         self.init_block = K.Sequential([Padding2D(3, pad_type='reflect'),
#                                         ConvBlock(channels, 7, padding="valid", use_bias=use_bias, strides=2,
#                                                   norm_layer=self.norm_layer)])
#         self.conv1 = ConvBlock(channels * 2, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
#                                activation=tf.nn.leaky_relu, sn=True)
#         self.conv2 = ConvBlock(channels * 4, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
#                                activation=tf.nn.leaky_relu, sn=True)
#         self.conv3 = ConvBlock(channels * 8, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
#                                activation=tf.nn.leaky_relu, sn=True)
#         self.conv4 = ConvBlock(channels * 8, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
#                                activation=tf.nn.leaky_relu, sn=True)
#         self.conv5 = ConvBlock(channels * 8, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
#                                activation=tf.nn.leaky_relu, sn=True)
#         self.flatten = K.layers.Flatten()
#         self.p_dense = tfa.layers.SpectralNormalization(K.layers.Dense(channels * 4))
#         self.p_dense1 = tfa.layers.SpectralNormalization(K.layers.Dense(channels * 4))
#
#     def call(self, inputs, training=None, alt=False):
#         x = tf.image.resize(inputs, (256, 256))
#         x = self.init_block(x, training=True)
#         x = self.conv1(x, training=True)
#         x = self.conv2(x, training=True)
#         x = self.conv3(x, training=True)
#         x = self.conv4(x, training=True)
#         x = self.conv5(x, training=True)
#         x = self.flatten(x)
#         loc = self.p_dense(x)
#         scale = self.p_dense1(x)
#         x = tfp.distributions.Normal(loc, tf.exp(scale * 0.5))
#         if training:
#             return x.sample(), loc, scale
#         return x.sample()


class TransmatorGenerator(K.Model):
    def __init__(self,
                 base_channels=64,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(512, 512),
                 classes=3,
                 aux=False,
                 activation=tf.nn.leaky_relu,
                 **kwargs):
        super().__init__(**kwargs)
        assert (in_channels > 0)
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
                                      kernel_initializer='he_normal')
                      if n < 1 else
                      tfa.layers.SpectralNormalization(K.layers.Conv2D(base_channels,
                                                                       3,
                                                                       activation=activation,
                                                                       padding='same',
                                                                       kernel_initializer='he_normal'))
                      for n in range(3)]
        self.conv2 = [K.layers.Conv2D(base_channels * 2,
                                      3,
                                      activation=activation,
                                      padding='same',
                                      kernel_initializer='he_normal')
                      if n < 2 else
                      tfa.layers.SpectralNormalization(K.layers.Conv2D(base_channels * 2,
                                                                       3,
                                                                       activation=activation,
                                                                       padding='same',
                                                                       kernel_initializer='he_normal'))
                      for n in range(4)]
        self.conv3 = [K.layers.Conv2D(base_channels * 4,
                                      3,
                                      activation=activation,
                                      padding='same',
                                      kernel_initializer='he_normal')
                      if n < 2 else
                      tfa.layers.SpectralNormalization(K.layers.Conv2D(base_channels * 4,
                                                                       3,
                                                                       activation=activation,
                                                                       padding='same',
                                                                       kernel_initializer='he_normal'))
                      for n in range(4)]
        self.conv4 = [K.layers.Conv2D(base_channels * 8,
                                      3,
                                      activation=activation,
                                      padding='same',
                                      kernel_initializer='he_normal')
                      if n < 2 else
                      tfa.layers.SpectralNormalization(K.layers.Conv2D(base_channels * 8,
                                                                       3,
                                                                       activation=activation,
                                                                       padding='same',
                                                                       kernel_initializer='he_normal'))
                      for n in range(4)]
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
        for _ in range(10):
            self.bn.append(K.layers.BatchNormalization())
        self.up6 = K.layers.Conv2DTranspose(base_channels * 8, (3, 3), strides=(2, 2), padding="same")
        self.up7 = K.layers.Conv2DTranspose(base_channels * 4, (3, 3), strides=(2, 2), padding="same")
        self.up8 = K.layers.Conv2DTranspose(base_channels * 2, (3, 3), strides=(2, 2), padding="same")
        self.up9 = K.layers.Conv2DTranspose(base_channels, (3, 3), strides=(2, 2), padding="same")
        self.conv10 = K.layers.Conv2D(classes, 1, strides=1, padding="same", activation=tf.nn.tanh)

    def call(self, inputs, training=None, mask=None):
        img, seg = inputs
        c1 = self.in_conv(img)
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
        # u6 = K.layers.concatenate([u6, c4])
        u6 = K.layers.concatenate([u6, tf.image.resize(seg, size=tf.shape(u6)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)])
        c6 = self.conv4[2](u6)
        # c6 = self.bn[10](c6, training=training)
        c6 = self.dropout2(c6)
        c6 = self.conv4[3](c6)
        # c6 = self.bn[11](c6, training=training)

        u7 = self.up7(c6)
        # u7 = K.layers.concatenate([u7, c3])
        u7 = K.layers.concatenate([u7, tf.image.resize(seg, size=tf.shape(u7)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)])
        c7 = self.conv3[2](u7)
        # c7 = self.bn[12](c7, training=training)
        c7 = self.dropout2(c7)
        c7 = self.conv3[3](c7)
        # c7 = self.bn[13](c7, training=training)

        u8 = self.up8(c7)
        # u8 = K.layers.concatenate([u8, c2])
        u8 = K.layers.concatenate([u8, tf.image.resize(seg, size=tf.shape(u8)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)])
        c8 = self.conv2[2](u8)
        # c8 = self.bn[14](c8, training=training)
        c8 = self.dropout1(c8)
        c8 = self.conv2[3](c8)
        # c8 = self.bn[15](c8, training=training)

        u9 = self.up9(c8)
        # u9 = K.layers.concatenate([u9, c1])
        u9 = K.layers.concatenate([u9, tf.image.resize(seg, size=tf.shape(u9)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)])
        c9 = self.conv1[1](u9)
        # c9 = self.bn[16](c9, training=training)
        c9 = self.dropout1(c9)
        c9 = self.conv1[2](c9)
        # c9 = self.bn[17](c9, training=training)

        final = self.conv10(c9)
        return final


class TransmatorDiscriminator(K.Model):
    def __init__(self, channels=64, n_scale=2, n_dis=4):
        super().__init__()
        self.conv1 = [K.layers.Conv2D(channels, 4, 2, padding="SAME", activation=tf.nn.leaky_relu)
                      for _ in range(n_scale)]
        self.conv2s = [
            [ConvBlock(tf.minimum(channels * (2 ** n), 512).numpy(), 4, strides=1 if n == (n_dis - 1) else 2,
                       padding="SAME", activation=tf.nn.relu, sn=True)
             for n in range(1, n_dis)] for _ in range(n_scale)]
        self.conv_final = [K.layers.Conv2D(1, 4, 1, padding="SAME", activation=tf.nn.leaky_relu)
                           for _ in range(n_scale)]
        self.downsample = K.layers.AveragePooling2D(3, 2, padding="same")

    def call(self, inputs, training=None, mask=None):
        # img, segmap = inputs
        outputs = []
        for conv1, conv2s, conv_final in zip(self.conv1, self.conv2s, self.conv_final):
            features = []
            # x = tf.concat([img, segmap], axis=-1)
            x = conv1(inputs, training=training)
            features.append(x)
            for conv2 in conv2s:
                x = conv2(x, training=training)
                features.append(x)
            x = conv_final(x, training=training)
            features.append(x)
            outputs.append(features)
            inputs = self.downsample(inputs)
            # segmap = self.downsample(segmap)
        return outputs


if __name__ == "__main__":
    # from losses import get_loss
    # gan_loss_obj = get_loss("Wasserstein")
    # feat_loss = get_loss("FeatureLoss")

    # def generator_loss(generated_list):
    #     total_loss = 0
    #     for generated in generated_list:
    #         generated = generated[-1]
    #         total_loss += tf.reduce_mean(gan_loss_obj(-tf.ones_like(generated), generated))
    #     return total_loss
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    # vgg_loss = get_loss("VGGLoss")
    # enc = GAUEncoder()
    gen = TransmatorGenerator(classes=3)
    disc = TransmatorDiscriminator()
    # gen.custom_build((1, 256, 256, 3))
    i = tf.random.uniform((1, 256, 512, 3))
    s = tf.random.uniform((1, 256, 512, 3))
    g_out = gen(i)
    d_out = disc(g_out)
    # K.models.save_model(gen, "here")
    # K.models.save_model(disc, "here")
    # K.models.save_model(disc, "enc")

    # a = generator_loss(d_out)
    # b = vgg_loss(i, g_out)
    # c = feat_loss(d_out, d_out)
    # j = [print(x.shape) for d in d_out for x in d]
    # print("==================\n", g_out.shape)
    # print(v_out.shape)
    # print(c)
    print(gen.summary())
    print(disc.summary())
    # print(enc.summary())
