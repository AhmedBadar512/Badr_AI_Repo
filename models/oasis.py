""" Implement the following components that are used in GauGAN model.
Generator (Resnet-based)
Discriminator (PatchGAN)
Encoder
"""

import tensorflow as tf

import tensorflow.keras as K
from .layers import SPADEResBlock, ResBlock_D


class OASISGenerator(K.Model):
    def __init__(self, channels_list=None, sn=True, init=K.initializers.glorot_uniform()):
        super().__init__()
        if channels_list is None:
            channels_list = [1024, 1024, 512, 256, 128, 64]
        self.channels_list = channels_list
        self.sn = sn
        self.upsample = K.layers.UpSampling2D((2, 2))
        self.conv = K.layers.Conv2D(channels_list[0], 3, 1, padding="SAME", kernel_initializer=init)
        self.spade_resblocks = [SPADEResBlock(channels=chn, sn=self.sn, ks=3, is_oasis=True, init=init) for chn in channels_list]
        self.final_conv = K.layers.Conv2D(3, 3, 1, padding="SAME", kernel_initializer=init)

    def build(self, input_shape):
        self.init_shp = tf.cast(input_shape[1:3], dtype=tf.int32) // 2 ** (len(self.channels_list) - 1)
        # self.latent_shp = (input_shape[0], input_shape[1], input_shape[2], 64)

    def call(self, inputs, training=None, mask=None):
        segmap = inputs
        latent_shp = [tf.shape(segmap)[0], 1, 1, 64]
        z_latent = tf.random.normal(latent_shp)
        ones_shp = tf.concat([tf.shape(segmap)[:3], [64]], axis=0)
        z_latent = tf.ones(ones_shp) * z_latent
        segmap = tf.concat([segmap, z_latent], axis=-1)
        x = tf.image.resize(segmap, self.init_shp, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = self.conv(x)
        for n, spade_blk in enumerate(self.spade_resblocks):
            x = spade_blk([x, segmap])
            if n < (len(self.spade_resblocks) - 1):
                x = self.upsample(x)
        x = tf.nn.leaky_relu(x)
        x = self.final_conv(x)
        x = K.activations.tanh(x)
        return x


class OASISDiscriminator(K.Model):
    def __init__(self, classes=19, channels_list=None, init=K.initializers.glorot_uniform()):
        super().__init__()
        if channels_list is None:
            channels_list = [128, 128, 256, 256, 512, 512]
        self.enc_blks = [ResBlock_D(channels, up_down="down", first=(n == 0), init=init) for n, channels in
                         enumerate(channels_list)]
        channels_list.reverse()
        self.dec_blks = [ResBlock_D(channels, up_down="up", init=init) for n, channels in enumerate(channels_list[1:] + [64])]
        self.conv_final = K.layers.Conv2D(classes + 1, 1, 1, padding="SAME", kernel_initializer=init)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x_encs, x_decs = [], []
        for enc_blk in self.enc_blks:
            x = enc_blk(x)
            x_encs.append(x)
        x = self.dec_blks[0](x)
        for n, dec_blk in enumerate(self.dec_blks):
            x_side = x_encs.pop()
            if n == 0:
                continue
            x = tf.concat([x, x_side], axis=-1) if n > 0 else x
            x = dec_blk(x)
            x_decs.append(x)
        x = self.conv_final(x)
        return x


if __name__ == "__main__":
    # from losses import get_loss
    # gan_loss_obj = get_loss("Wasserstein")
    # feat_loss = get_loss("FeatureLoss")

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    gen = OASISGenerator()
    disc = OASISDiscriminator(35)
    i = tf.random.uniform((1, 256, 512, 3))
    s = tf.random.uniform((1, 256, 512, 35))
    g_out = gen(s)
    d_out = disc(g_out)
    K.models.save_model(gen, "gen_here")
    # K.models.save_model(disc, "disc_here")
    print(gen.summary())
    print(disc.summary())
    # print(g_out.shape, d_out.shape)
