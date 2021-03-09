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

tfpl = tfp.layers
tfd = tfp.distributions


class GAUEncoder(K.Model):
    def __init__(self, channels=64, norm_layer="instance"):
        super().__init__()
        use_bias = norm_layer == "instance"
        self.norm_layer = norm_layer
        self.init_block = K.Sequential([Padding2D(3, pad_type='reflect'),
                                        ConvBlock(channels, 7, padding="valid", use_bias=use_bias, strides=2,
                                                  norm_layer=self.norm_layer)])
        self.conv1 = ConvBlock(channels * 2, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
                               activation=tf.nn.leaky_relu, sn=True)
        self.conv2 = ConvBlock(channels * 4, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
                               activation=tf.nn.leaky_relu, sn=True)
        self.conv3 = ConvBlock(channels * 8, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
                               activation=tf.nn.leaky_relu, sn=True)
        self.conv4 = ConvBlock(channels * 8, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
                               activation=tf.nn.leaky_relu, sn=True)
        self.conv5 = ConvBlock(channels * 8, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
                               activation=tf.nn.leaky_relu, sn=True)
        self.flatten = K.layers.Flatten()
        self.p_dense = tfa.layers.SpectralNormalization(K.layers.Dense(channels * 4))
        self.p_dense1 = tfa.layers.SpectralNormalization(K.layers.Dense(channels * 4))

    def call(self, inputs, training=None, alt=False):
        x = tf.image.resize(inputs, (256, 256))
        x = self.init_block(x, training=True)
        x = self.conv1(x, training=True)
        x = self.conv2(x, training=True)
        x = self.conv3(x, training=True)
        x = self.conv4(x, training=True)
        x = self.conv5(x, training=True)
        x = self.flatten(x)
        loc = self.p_dense(x)
        scale = self.p_dense1(x)
        x = tfp.distributions.Normal(loc, tf.exp(scale * 0.5))
        # y = self.prob_dense(y)
        # mean, var = self.fc1(x), self.fc2(x)
        if training:
            return x.sample(), loc, scale
        return x.sample()


class GAUGenerator(K.Model):
    def __init__(self, channels=1024, num_up_layers=6, sn=True):
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

    def build(self, input_shape):
        self.z_height, self.z_width = input_shape[1][1] // (pow(2, self.num_up_layers)), input_shape[1][2] // (
            pow(2, self.num_up_layers))
        self.fc1 = Dense(self.z_height * self.z_width * self.channels)

    def call(self, inputs, training=None, mask=None):
        features, segmap = inputs
        x = self.fc1(features)
        x = tf.reshape(x, (-1, self.z_height, self.z_width, self.channels))
        x = self.spade_resblock1([x, segmap])
        x = self.upsample(x)
        x = self.spade_resblock2([x, segmap])
        if self.num_up_layers > 5:
            x = self.upsample(x)
        x = self.spade_resblock3([x, segmap])
        for spade_blk in self.spade_resblocks:
            x = self.upsample(x)
            x = spade_blk([x, segmap])
        if self.num_up_layers > 6:
            x = self.upsample(x)
            x = self.spade_resblock4([x, segmap])
        x = tf.nn.leaky_relu(x)
        x = self.final_conv(x)
        return x


class GAUDiscriminator(K.Model):
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
        img, segmap = inputs
        outputs = []
        for conv1, conv2s, conv_final in zip(self.conv1, self.conv2s, self.conv_final):
            features = []
            x = tf.concat([img, segmap], axis=-1)
            x = conv1(x, training=training)
            features.append(x)
            for conv2 in conv2s:
                x = conv2(x, training=training)
                features.append(x)
            x = conv_final(x, training=training)
            features.append(x)
            outputs.append(features)
            img = self.downsample(img)
            segmap = self.downsample(segmap)
        return outputs

#
# from losses import get_loss
# gan_loss_obj = get_loss("Wasserstein")
# feat_loss = get_loss("FeatureLoss")


# def generator_loss(generated_list):
#     total_loss = 0
#     for generated in generated_list:
#         generated = generated[-1]
#         total_loss += tf.reduce_mean(gan_loss_obj(-tf.ones_like(generated), generated))
#     return total_loss


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    # vgg_loss = get_loss("VGGLoss")
    enc = GAUEncoder()
    gen = GAUGenerator()
    disc = GAUDiscriminator()
    # gen.custom_build((1, 256, 256, 3))
    i = tf.random.uniform((1, 256, 512, 3))
    s = tf.random.uniform((1, 256, 512, 3))
    v_out, _, _ = enc(i, training=True)
    g_out = gen([v_out, s])
    d_out = disc([i, s])
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
    print(enc.summary())
