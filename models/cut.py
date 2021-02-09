""" Implement the following components that used in CUT/FastCUT model.
Generator (Resnet-based)
Discriminator (PatchGAN)
Encoder
PatchSampleMLP
CUT_model
"""

import tensorflow as tf

import tensorflow.keras as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda

from .layers import ConvBlock, ConvTransposeBlock, ResBlock, Padding2D


class CUTGenerator(Model):
    def __init__(self, classes=3, norm_layer="instance", use_antialias=True, resnet_blocks=9):
        super().__init__()
        use_bias = norm_layer == "instance"
        self.classes = classes
        self.norm_layer = norm_layer
        self.use_antialias = use_antialias
        self.resnet_blocks = resnet_blocks
        self.init_block = K.Sequential([Padding2D(3, pad_type='reflect'),
                                        ConvBlock(64, 7, padding="valid", use_bias=use_bias,
                                                  norm_layer=self.norm_layer)])
        if use_antialias:
            self.conv1 = ConvBlock(128, 3, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation='relu')
            self.conv2 = ConvBlock(256, 3, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation='relu')
        else:
            self.conv1 = ConvBlock(128, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
                                   activation='relu')
            self.conv2 = ConvBlock(256, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
                                   activation='relu')
        self.res_blocks = [ResBlock(256, 3, use_bias, norm_layer) for _ in range(self.resnet_blocks)]
        if use_antialias:
            self.conv3 = ConvBlock(128, 3, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation='relu')
            self.conv4 = ConvBlock(64, 3, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation='relu')
        else:
            self.conv3 = ConvTransposeBlock(128, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
                                            activation='relu')
            self.conv4 = ConvTransposeBlock(64, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer,
                                            activation='relu')
        self.padf = Padding2D(3, pad_type='reflect')
        self.outputs = ConvBlock(classes, 7, padding='valid', activation='tanh')

    def call(self, inputs, training=None, mask=None, res_feats=(0, 1), encoder_output=False):
        x = self.init_block(inputs)
        x1 = self.conv1(x)
        feats = [inputs, x1]
        if self.use_antialias:
            x1 = tf.image.resize(x1, size=tf.shape(x1)[1:3] // 2, antialias=True)
        x2 = self.conv2(x1)
        feats.append(x2)
        if self.use_antialias:
            x2 = tf.image.resize(x2, size=tf.shape(x2)[1:3] // 2, antialias=True)
        for n, res_block in enumerate(self.res_blocks):
            x2 = res_block(x2)
            if n in res_feats:
                feats.append(x2)
        x3 = self.conv3(x2)
        if self.use_antialias:
            x3 = tf.image.resize(x3, size=tf.shape(x3)[1:3] * 2, antialias=True)
        x4 = self.conv4(x3)
        if self.use_antialias:
            x4 = tf.image.resize(x4, size=tf.shape(x4)[1:3] * 2, antialias=True)
        x5 = self.padf(x4)
        output = self.outputs(x5)
        if encoder_output:
            return output, feats
        else:
            return output


class CUTDiscriminator(Model):
    def __init__(self, norm_layer="instance", use_antialias=True):
        super().__init__()
        use_bias = (norm_layer == 'instance')
        self.use_antialias = use_antialias
        if self.use_antialias:
            self.conv1 = ConvBlock(64, 4, padding='same', activation=tf.nn.leaky_relu)
            self.conv2 = ConvBlock(128, 4, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation=tf.nn.leaky_relu)
            self.conv3 = ConvBlock(256, 4, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation=tf.nn.leaky_relu)
        else:
            self.conv1 = ConvBlock(64, 4, strides=2, padding='same', activation=tf.nn.leaky_relu)
            self.conv2 = ConvBlock(128, 4, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation=tf.nn.leaky_relu)
            self.conv3 = ConvBlock(256, 4, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation=tf.nn.leaky_relu)
        self.conv4 = ConvBlock(512, 4, padding='valid', use_bias=use_bias, norm_layer=norm_layer, activation=tf.nn.leaky_relu)
        self.pad1 = Padding2D(1, pad_type='constant')
        self.pad2 = Padding2D(1, pad_type='constant')
        self.outputs = ConvBlock(1, 4, padding='valid')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        if self.use_antialias:
            x = tf.image.resize(x, size=tf.shape(x)[1:3] // 2, antialias=True)
        x = self.conv2(x)
        if self.use_antialias:
            x = tf.image.resize(x, size=tf.shape(x)[1:3] // 2, antialias=True)
        x = self.conv3(x)
        if self.use_antialias:
            x = tf.image.resize(x, size=tf.shape(x)[1:3] // 2, antialias=True)
        x = self.pad1(x)
        x = self.conv4(x)
        x = self.pad2(x)
        return self.outputs(x)


class CUTEncoder(K.Model):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def call(self, inputs, training=None, mask=None):
        return self.gen(inputs, encoder_output=True)[1]


class PatchSampleMLP(Model):
    """ Create a PatchSampleMLP.
    Adapt from official CUT implementation (https://github.com/taesungp/contrastive-unpaired-translation).
    PatchSampler samples patches from pixel/feature-space.
    Two-layer MLP projects both the input and output patches to a shared embedding space.
    """
    def __init__(self, units, num_patches, **kwargs):
        super(PatchSampleMLP, self).__init__(**kwargs)
        self.units = units
        self.num_patches = num_patches
        self.l2_norm = Lambda(lambda x: x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + 10-10))

    def build(self, input_shape):
        initializer = tf.random_normal_initializer(0., 0.02)
        feats_shape = input_shape
        for feat_id in range(len(feats_shape)):
            mlp = tf.keras.models.Sequential([
                    Dense(self.units, activation="relu", kernel_initializer=initializer),
                    Dense(self.units, kernel_initializer=initializer),
                ])
            setattr(self, f'mlp_{feat_id}', mlp)

    def call(self, inputs, patch_ids=None, training=None):
        feats = inputs
        samples = []
        ids = []
        for feat_id, feat in enumerate(feats):
            B, H, W, C = feat.shape

            feat_reshape = tf.reshape(feat, [B, -1, C])

            if patch_ids is not None:
                patch_id = patch_ids[feat_id]
            else:
                patch_id = tf.random.shuffle(tf.range(H * W))[:min(self.num_patches, H * W)]

            x_sample = tf.reshape(tf.gather(feat_reshape, patch_id, axis=1), [-1, C])
            mlp = getattr(self, f'mlp_{feat_id}')
            x_sample = mlp(x_sample)
            x_sample = self.l2_norm(x_sample)
            samples.append(x_sample)
            ids.append(patch_id)

        return samples, ids


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    a = tf.random.uniform(shape=(8, 256, 256, 3))
    c_gen = CUTGenerator(classes=5, norm_layer="instance")
    c_enc = CUTEncoder(c_gen)
    c_disc = CUTDiscriminator()
    c_mlp = PatchSampleMLP(64, 64)
    g1, g_feats = c_gen(a, encoder_output=True)
    d1 = c_disc(a)
    e_feats = c_enc(a)
    [print(e1.shape) for e1 in e_feats]
    print(g1.shape)
    print(d1.shape)
    x, x_ids = c_mlp(e_feats)
    y, y_ids = c_mlp(g_feats, x_ids)
    [print(e1.shape) for e1, e2 in zip(x, y)]
