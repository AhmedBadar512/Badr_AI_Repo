""" Implement the following components that are used in GauGAN model.
Generator (Resnet-based)
Discriminator (PatchGAN)
Encoder
"""

import tensorflow as tf
import tensorflow_addons as tfa

import tensorflow.keras as K
from tensorflow.keras.layers import Dense
from layers import ConvBlock, ConvTransposeBlock, ResBlock, Padding2D
import tensorflow_probability as tfp

tfpl = tfp.layers
tfd = tfp.distributions


class SPADE(K.layers.Layer):
    def __init__(self, channels, sn=False, activation=None):
        super().__init__()
        self.conv1 = K.layers.Conv2D(128, 5, 1, padding="SAME", activation="relu")
        self.conv_gamma = K.layers.Conv2D(channels, 5, 1, padding="SAME")
        self.conv_beta = K.layers.Conv2D(channels, 5, 1, padding="SAME")
        if sn:
            self.conv1 = tfa.layers.SpectralNormalization(self.conv1)
            self.conv_gamma = tfa.layers.SpectralNormalization(self.conv_gamma)
            self.conv_beta = tfa.layers.SpectralNormalization(self.conv_beta)
        self.activation = activation
        # self.avg_pool = lambda feature, strides: K.layers.AveragePooling2D(3, strides=strides, padding="SAME")(feature)

    def call(self, inputs, segmap, **kwargs):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        x = (inputs - mean) / tf.sqrt(var + 1e-5)
        segmap_processed = tf.nn.avg_pool(segmap, 3, tf.shape(segmap)[1:3] // tf.shape(x)[1:3], padding="SAME")
        segmap_processed = self.conv1(segmap_processed)
        segmap_gamma = self.conv_gamma(segmap_processed)
        segmap_beta = self.conv_beta(segmap_processed)
        x = x * (1 + segmap_gamma) + segmap_beta
        if self.activation is not None:
            x = self.activation(x)
        return x


class SPADEResBlock(K.Model):
    def __init__(self, channels, sn=False):
        super().__init__()
        self.channels = channels
        self.sn = sn

    def build(self, input_shape):
        channels_middle = tf.minimum(self.channels, input_shape[-1])
        self.spade1 = SPADE(input_shape[-1], activation=tf.nn.leaky_relu, sn=self.sn)
        self.conv1 = K.layers.Conv2D(channels_middle, 3, 1, padding="SAME")
        if self.sn:
            self.conv1 = tfa.layers.SpectralNormalization(self.conv1)
        self.spade2 = SPADE(channels_middle, activation=tf.nn.leaky_relu, sn=self.sn)
        if self.sn:
            self.conv2 = tfa.layers.SpectralNormalization(K.layers.Conv2D(channels_middle, 3, 1, padding="SAME"))
        else:
            self.conv2 = K.layers.Conv2D(channels_middle, 3, 1, padding="SAME")
        if self.channels != input_shape[-1]:
            self.spade3 = SPADE(input_shape[-1], sn=self.sn)
            if self.sn:
                self.conv3 = tfa.layers.SpectralNormalization(K.layers.Conv2D(self.channels, 3, 1, padding="SAME"))
            else:
                self.conv3 = K.layers.Conv2D(self.channels, 3, 1, padding="SAME")

    def call(self, inputs, segmap, training=None, mask=None):
        x = self.spade1(inputs, segmap)
        x = self.conv1(x)
        x = self.spade2(x, segmap)
        x = self.conv2(x)
        if self.channels != inputs.shape[-1]:
            x_branch = self.spade3(inputs, segmap)
            x_branch = self.conv3(x_branch)
            return x_branch + x
        return x + inputs


class Encoder(K.Model):
    def __init__(self, channels=64, norm_layer="instance"):
        super().__init__()
        use_bias = norm_layer == "instance"
        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(channels * 4), scale=1),
                                reinterpreted_batch_ndims=1)
        self.norm_layer = norm_layer
        self.init_block = K.Sequential([Padding2D(3, pad_type='reflect'),
                                        ConvBlock(channels, 7, padding="valid", use_bias=use_bias, strides=2,
                                                  norm_layer=self.norm_layer)])
        self.conv1 = ConvBlock(channels * 2, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
                               activation=tf.nn.leaky_relu)
        self.conv2 = ConvBlock(channels * 4, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
                               activation=tf.nn.leaky_relu)
        self.conv3 = ConvBlock(channels * 8, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
                               activation=tf.nn.leaky_relu)
        self.conv4 = ConvBlock(channels * 8, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
                               activation=tf.nn.leaky_relu)
        self.conv5 = ConvBlock(channels * 8, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
                               activation=tf.nn.leaky_relu)
        self.flatten = K.layers.Flatten()
        encode_size = tfpl.MultivariateNormalTriL.params_size(channels * 4)
        self.p_dense = tfa.layers.SpectralNormalization(K.layers.Dense(encode_size))
        self.prob_dense = tfpl.MultivariateNormalTriL(channels * 4,
                                                      activity_regularizer=tfpl.KLDivergenceRegularizer(prior))
        # self.fc1 = tfa.layers.SpectralNormalization(K.layers.Dense(256))
        # self.fc2 = tfa.layers.SpectralNormalization(K.layers.Dense(256))

    def call(self, inputs, training=None):
        x = tf.image.resize(inputs, (256, 256))
        x = self.init_block(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        y = self.p_dense(x)
        y = self.prob_dense(y)
        # mean, var = self.fc1(x), self.fc2(x)
        return y


class Generator(K.Model):
    def __init__(self, channels=512, num_up_layers=6, sn=True):
        super().__init__()
        self.num_up_layers = num_up_layers
        self.channels = channels
        self.sn = sn
        self.upsample = K.layers.UpSampling2D((2, 2))
        self.spade_resblock1 = SPADEResBlock(channels=channels, sn=self.sn)
        self.spade_resblock2 = SPADEResBlock(channels=channels, sn=self.sn)
        self.spade_resblock3 = SPADEResBlock(channels=channels, sn=self.sn)
        self.spade_resblocks = [SPADEResBlock(channels=channels // 2, sn=self.sn),
                                SPADEResBlock(channels=channels // 4, sn=self.sn),
                                SPADEResBlock(channels=channels // 8, sn=self.sn),
                                SPADEResBlock(channels=channels // 16, sn=self.sn)]
        if self.num_up_layers > 6:
            self.spade_resblock4 = SPADEResBlock(channels=channels // 32, sn=self.sn)
        self.final_conv = K.layers.Conv2D(3, 3, 1, padding="SAME", activation=K.activations.tanh)

    def custom_build(self, input_shape):
        self.z_height, self.z_width = input_shape[1] // (pow(2, self.num_up_layers)), input_shape[2] // (
            pow(2, self.num_up_layers))
        self.fc1 = Dense(input_shape[0] * self.z_height * self.z_width * self.channels)

    def call(self, inputs, segmap, training=None, mask=None):
        x = self.fc1(inputs)
        x = tf.reshape(x, (-1, self.z_height, self.z_width, self.channels))
        x = self.spade_resblock1(x, segmap)
        x = self.upsample(x)
        x = self.spade_resblock2(x, segmap)
        if self.num_up_layers > 5:
            x = self.upsample(x)
        x = self.spade_resblock3(x, segmap)
        for spade_blk in self.spade_resblocks:
            x = self.upsample(x)
            x = spade_blk(x, segmap)
        if self.num_up_layers > 6:
            x = self.upsample(x)
            x = self.spade_resblock4(x)
        x = tf.nn.leaky_relu(x)
        x = self.final_conv(x)
        return x


class Discriminator(K.Model):
    def __init__(self, channels=64, n_scale=6, n_dis=4):
        super().__init__()
        self.outputs = []
        self.conv1 = [tfa.layers.SpectralNormalization(
            K.layers.Conv2D(channels, 4, 2, padding="SAME", activation=tf.nn.leaky_relu))
            for _ in range(n_scale)]
        self.conv2s = [
            [ConvBlock(tf.minimum(channels * (2 ** n), 512), 4, strides=1 if n == (n_dis - 1) else 2,
                       padding="SAME", activation=tf.nn.relu, sn=True)
             for n in range(1, n_dis)] for _ in range(n_scale)]
        self.conv_final = [
            tfa.layers.SpectralNormalization(K.layers.Conv2D(1, 4, 1, padding="SAME", activation=tf.nn.leaky_relu))
            for _ in range(n_scale)]
        self.downsample = K.layers.AveragePooling2D(3, 2, padding="same")

    def call(self, inputs, segmap, training=None, mask=None):
        for conv1, conv2s, conv_final in zip(self.conv1, self.conv2s, self.conv_final):
            features = []
            x = tf.concat([inputs, segmap], axis=-1)
            x = conv1(x)
            features.append(x)
            for conv2 in conv2s:
                x = conv2(x)
                features.append(x)
            x = conv_final(x)
            features.append(x)
            self.outputs.append(features)
            inputs = self.downsample(inputs)
            segmap = self.downsample(segmap)
        return self.outputs


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    enc = Encoder()
    gen = Generator()
    disc = Discriminator()
    gen.custom_build((1, 256, 256, 5))
    i = tf.random.uniform((1, 256, 256, 3))
    s = tf.random.uniform((1, 256, 256, 5))
    v_out = enc(i)
    g_out = gen(v_out, s)
    d_out = disc(i, s)
    j = [print(x.shape) for d in d_out for x in d]
    print(len(j))
    print("==================\n", g_out.shape)
    print(v_out.shape)
