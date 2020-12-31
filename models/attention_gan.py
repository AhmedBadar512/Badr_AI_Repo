import tensorflow as tf
import tensorflow.keras as K
from .autoencoder_layers import SelfAttentionBlock, Upscaler


class Encoder(K.Model):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.latent_dim = latent_dim
        self.flatten = K.layers.Flatten()
        self.conv1 = K.layers.Conv2D(64, 3, 1, padding="same", activation=tf.nn.relu, kernel_regularizer=K.regularizers.l2())
        self.conv2 = K.layers.Conv2D(128, 3, 2, padding="same", activation=tf.nn.relu, kernel_regularizer=K.regularizers.l2())
        self.conv3 = K.layers.Conv2D(256, 3, 2, padding="same", activation=tf.nn.relu, kernel_regularizer=K.regularizers.l2())
        self.self_attn_blk3 = SelfAttentionBlock(256)
        self.conv4 = K.layers.Conv2D(512, 3, 2, padding="same", activation=tf.nn.relu, kernel_regularizer=K.regularizers.l2())
        self.self_attn_blk4 = SelfAttentionBlock(512)
        self.conv5 = K.layers.Conv2D(latent_dim, 3, 2, padding="same", activation=tf.nn.relu, kernel_regularizer=K.regularizers.l2())
        self.dense1 = K.layers.Dense(latent_dim)
        self.dense2 = K.layers.Dense(latent_dim * 4 * 4)
        self.upscaler = Upscaler(512)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.self_attn_blk3(x)
        x = self.conv4(x)
        x = self.self_attn_blk4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.reshape(x, shape=(-1, 4, 4, self.latent_dim))
        x = self.upscaler(x)
        return x


class Decoder(K.Model):
    def __init__(self):
        super().__init__()
        self.upscaler1 = Upscaler(256)
        self.upscaler2 = Upscaler(128)
        self.self_attn_blk1 = SelfAttentionBlock(128)
        self.upscaler3 = Upscaler(64)
        self.conv1 = K.layers.Conv2D(64, 3, 1, "same", activation=tf.nn.leaky_relu, kernel_regularizer=K.regularizers.l2())
        self.conv2 = K.layers.Conv2D(64, 3, 1, "same", kernel_regularizer=K.regularizers.l2())
        self.self_attn_blk2 = SelfAttentionBlock(64)
        self.conv_f = K.layers.Conv2D(3, 5, 1, "same", activation=tf.nn.sigmoid, kernel_regularizer=K.regularizers.l2())

    def call(self, inputs, training=None, mask=None):
        x = self.upscaler1(inputs)
        x = self.upscaler2(x)
        x = self.self_attn_blk1(x)
        x = self.upscaler3(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.self_attn_blk2(x)
        x = self.conv_f(x)
        return x


class AutoEncoder(K.Model):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs, training=None, mask=None):
        x = self.encoder(inputs)
        y = self.decoder(x)
        return y
