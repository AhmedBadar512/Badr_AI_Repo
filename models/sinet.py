"""
    SINet for image segmentation, implemented in TensorFlow.
    Original paper: 'SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and
    Information Blocking Decoder,' https://arxiv.org/abs/1911.09099.
"""

# __all__ = ['SINet', 'sinet_cityscapes']

import os
import tensorflow as tf
import tensorflow.keras as K

# ============== For debugging (in case you have multiple GPUs), else comment out ================= #
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# for gpu in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu, True)
DEFAULT_ACTIVATION = K.layers.ReLU()  # Using PReLU according to paper doesn't make the parameters match

class ConvBNReLU(K.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding="same", activation=DEFAULT_ACTIVATION):
        super().__init__()
        self.conv = K.layers.Convolution2D(filters, kernel_size, strides, padding)
        self.bn = K.layers.BatchNormalization()
        self.activation = activation

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DWConvBNReLU(K.layers.Layer):
    def __init__(self, kernel_size, strides=1, padding="same", activation=DEFAULT_ACTIVATION, depth_multiplier=1):
        super().__init__()
        self.conv = K.layers.DepthwiseConv2D(kernel_size, strides, padding, depth_multiplier)
        self.bn = K.layers.BatchNormalization()
        self.activation = activation

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SEBlock(K.layers.Layer):
    def __init__(self,
                 reduction=16,
                 mid_activation=K.layers.ReLU(),
                 out_activation=K.activations.sigmoid):
        super().__init__()
        self.reduction = reduction
        self.use_conv2 = (reduction > 1)
        self.pool = K.layers.GlobalAveragePooling2D()
        if self.use_conv2:
            self.activ = mid_activation
        self.out_activation = out_activation

    def build(self, input_shape):
        mid_channels = tf.math.ceil(input_shape[-1] / self.reduction)
        self.fc1 = K.layers.Dense(mid_channels)
        self.fc2 = K.layers.Dense(units=input_shape[-1],
                                  name="fc2")

    def call(self, inputs, **kwargs):
        w = self.pool(inputs)
        w = self.fc1(w)
        if self.use_conv2:
            w = self.activ(w)
            w = self.fc2(w)
        w = self.out_activation(w)
        w = w[:, tf.newaxis, tf.newaxis, :]
        return inputs * w


class DwsConvBlock(K.layers.Layer):
    def __init__(self, filters, kernel_size=1, strides=1, dw_act=DEFAULT_ACTIVATION, pw_act=DEFAULT_ACTIVATION,
                 use_seblock=True, se_reduction=1):
        super().__init__()
        self.dwconvbnrelu = DWConvBNReLU(kernel_size, strides, activation=dw_act)
        if use_seblock:
            self.se_block = SEBlock(reduction=se_reduction)
        self.use_seblock = use_seblock
        self.pwconvbnrelu = ConvBNReLU(filters, kernel_size, 1, "same", activation=pw_act)

    def call(self, inputs, **kwargs):
        x = self.dwconvbnrelu(inputs)
        if self.use_seblock:
            x = self.se_block(x)
        x = self.pwconvbnrelu(x)
        return x


class FDWConvBlock(K.layers.Layer):
    def __init__(self, kernel_size, strides, activation=None):
        super().__init__()
        self.v_conv = DWConvBNReLU(kernel_size=(kernel_size, 1), strides=strides, padding="same", activation=None)
        self.h_conv = DWConvBNReLU(kernel_size=(1, kernel_size), strides=strides, padding="same", activation=None)
        self.activation = activation

    def call(self, inputs, **kwargs):
        x = self.v_conv(inputs) + self.h_conv(inputs)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SBBlock(K.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, scale_factor=1):
        super().__init__()
        self.use_scale = (scale_factor > 1)
        if self.use_scale:
            self.down_scale = K.layers.AveragePooling2D(
                pool_size=scale_factor,
                strides=scale_factor,
                name="down_scale")
            self.upscale = None
        use_fdw = (scale_factor > 0)
        if use_fdw:
            self.conv1 = FDWConvBlock(kernel_size, strides, activation=DEFAULT_ACTIVATION)
        else:
            self.conv1 = DWConvBNReLU(kernel_size, strides, activation=DEFAULT_ACTIVATION)
        self.conv2 = K.layers.Convolution2D(filters, 1)
        self.bn = K.layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        if self.use_scale:
            x = self.down_scale(inputs)
            x = self.conv1(x, training=None)
        else:
            x = self.conv1(inputs, training=None)

        x = self.conv2(x, training=None)

        if self.use_scale:
            x = tf.image.resize(x, inputs.shape[1:3])

        x = self.bn(x, training=None)
        return x


class ChannelShuffle(K.layers.Layer):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def build(self, input_shape):
        self.channels_per_group = tf.math.ceil(input_shape[-1] / self.groups)
        self.height = input_shape[1]
        self.width = input_shape[2]

    def call(self, inputs, **kwargs):
        x = tf.reshape(inputs, shape=(-1, self.height * self.width, self.groups, self.channels_per_group))
        x = tf.transpose(x, perm=[0, 1, 3, 2])
        x = tf.reshape(x, (-1, self.height, self.width, inputs.shape[-1]))
        return x


class ESPBlock(K.layers.Layer):
    def __init__(self, kernel_sizes, scale_factors, use_residual=True):
        super().__init__()
        assert len(kernel_sizes) == len(scale_factors) and len(kernel_sizes) > 1, "Invalid kernel or scale factor input"
        self.use_residual = use_residual
        self.kernel_sizes = kernel_sizes
        self.scale_factors = scale_factors
        self.bn = K.layers.BatchNormalization()
        self.relu = DEFAULT_ACTIVATION

    def build(self, input_shape):
        groups = len(self.kernel_sizes)
        mid_channels = int(input_shape[-1] / groups)
        res_channels = input_shape[-1] - groups * mid_channels
        # self.conv = K.layers.Convolution2D(input_shape[-1], 1, 1, padding="same", groups=groups)
        self.gconvs = [K.layers.Convolution2D(mid_channels, 1, 1, padding="same") for _ in range(groups)]
        self.channel_shuffle = ChannelShuffle(groups)
        self.sb_blocks = [SBBlock(mid_channels + res_channels, self.kernel_sizes[n], scale_factor=self.scale_factors[n])
                          if n == 0
                          else SBBlock(mid_channels, self.kernel_sizes[n], scale_factor=self.scale_factors[n])
                          for n in range(groups)]

    def call(self, inputs, **kwargs):
        # x = self.conv(inputs)
        xs = [conv(inputs) for conv in self.gconvs]
        x = tf.concat(xs, axis=-1)
        x = self.channel_shuffle(x)
        x_full = [sb_block(x) for sb_block in self.sb_blocks]
        x = tf.concat(x_full, axis=-1)
        if self.use_residual:
            x = inputs + x
        x = self.bn(x)
        x = self.relu(x)
        return x


class SBStage(K.layers.Layer):
    def __init__(self, down_channels, kernel_sizes_list, scale_factors_list, use_residual=True, se_reduction=1):
        super().__init__()
        assert len(kernel_sizes_list) == len(scale_factors_list)
        self.downconv = DwsConvBlock(down_channels, 3, 2, dw_act=None, pw_act=None, se_reduction=se_reduction)
        self.esp_blocks = [ESPBlock(kernel_sizes, scale_factors, use_residual=use_residual)
                           for kernel_sizes, scale_factors in zip(kernel_sizes_list, scale_factors_list)]
        self.bn = K.layers.BatchNormalization()
        self.relu = DEFAULT_ACTIVATION
        self.prelu = DEFAULT_ACTIVATION

    def call(self, inputs, **kwargs):
        inputs = self.downconv(inputs)
        inputs = self.prelu(
            inputs)  # Something is up with the Dws Implementation that doesn't allow it to be used in that layer
        x = inputs
        for esp_block in self.esp_blocks:
            x = esp_block(x)
        x1 = x
        x = tf.concat([inputs, x], axis=-1)
        x = self.bn(x)
        x = self.relu(x)
        return x, x1


class SBEncoderInitBlock(K.layers.Layer):
    def __init__(self, mid_channels, filters):
        super().__init__()
        self.conv = ConvBNReLU(mid_channels, 3, 2)
        self.dwsconv = DwsConvBlock(filters, 3, 2)


    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.dwsconv(x)
        return x


class SBEncoder(K.layers.Layer):
    def __init__(self, filters, init_block_channels, down_channels_list,
                 kernel_sizes_list, scale_factors_list, use_residual_list):
        super().__init__()
        self.init_block = SBEncoderInitBlock(init_block_channels[0], init_block_channels[1])
        self.stage1 = SBStage(down_channels=down_channels_list[0], kernel_sizes_list=kernel_sizes_list[0],
                              scale_factors_list=scale_factors_list[0], use_residual=use_residual_list[0])
        self.stage2 = SBStage(down_channels=down_channels_list[1], kernel_sizes_list=kernel_sizes_list[1],
                              scale_factors_list=scale_factors_list[1], use_residual=use_residual_list[1],
                              se_reduction=2)
        self.conv = K.layers.Convolution2D(filters, 1, padding="same")

    def call(self, inputs, **kwargs):
        x1 = self.init_block(inputs)
        x, x2 = self.stage1(x1)
        x, _ = self.stage2(x)
        x = self.conv(x)
        return x, x2, x1


class SBDecodeBlock(K.layers.Layer):
    def __init__(self, scale_factor=2, out_size=None):
        super().__init__()
        self.scale_factor = scale_factor
        self.bn = K.layers.BatchNormalization()
        self.out_size = out_size
        self.max_t = K.layers.Maximum()

    def call(self, inputs, y, **kwargs):
        x = tf.image.resize(inputs, (y.shape[1], y.shape[2])) if self.out_size is None \
            else tf.image.resize(inputs, (self.out_size[0], self.out_size[1]))
        x = self.bn(x)
        w_conf = tf.nn.softmax(x)
        # w_max = tf.reduce_max(w_conf, axis=-1, keepdims=True)
        w_conf = [w_conf[..., h] for h in range(w_conf.shape[-1])]
        w_max = self.max_t(w_conf)[..., tf.newaxis]
        x = y * w_max + x
        return x


class SBDecoder(K.layers.Layer):
    def __init__(self, classes, out_size):
        super().__init__()
        self.decode1 = SBDecodeBlock(out_size=((out_size[0] // 8, out_size[1] // 8) if out_size else None))
        self.decode2 = SBDecodeBlock(out_size=((out_size[0] // 4, out_size[1] // 4) if out_size else None))
        self.conv3 = K.layers.Convolution2D(classes, 1, 1, padding="same")
        self.conv3t = K.layers.Convolution2D(classes, 3, 1, padding="same")
        self.out_size = out_size

    def call(self, y3, y2, y1, **kwargs):
        y2 = self.conv3(y2)
        x = self.decode1(y3, y2, training=None)
        x = self.decode2(x, y1, training=None)
        x = tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2))
        x = self.conv3t(x)
        x = tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2)) if self.out_size is None \
            else tf.image.resize(x, (self.out_size[0], self.out_size[1]))
        return x


class SINet(K.Model):
    def __init__(self, classes, init_block_channels, down_channels_list,
                 kernel_sizes_list, scale_factors_list, use_residual_list,
                 in_size=None, aux=False):
        super().__init__()
        self.encoder = SBEncoder(filters=classes,
                                 init_block_channels=[16, classes],
                                 down_channels_list=down_channels_list,
                                 kernel_sizes_list=kernel_sizes_list,
                                 scale_factors_list=scale_factors_list,
                                 use_residual_list=use_residual_list)
        self.decoder = SBDecoder(classes, out_size=in_size)
        self.aux = aux

    def call(self, inputs, training=None, mask=None):
        y3, y2, y1 = self.encoder(inputs, training=None)
        x = self.decoder(y3, y2, y1, training=None)
        if self.aux:
            return x, y3
        else:
            return x


def get_nie_sinet(classes, in_size, aux=False):
    kernel_sizes_list = [
        [[3, 5], [3, 3], [3, 3]],
        [[3, 5], [3, 3], [5, 5], [3, 5], [3, 5], [3, 5], [3, 3], [5, 5], [3, 5], [3, 5]]]
    scale_factors_list = [
        [[1, 1], [0, 1], [0, 1]],
        [[1, 1], [0, 1], [1, 4], [2, 8], [1, 1], [1, 1], [0, 1], [1, 8], [2, 4], [0, 2]]]

    chnn = 4
    dims = [24] + [24 * (i + 2) + 4 * (chnn - 1) for i in range(3)]

    dim1 = dims[0]
    dim2 = dims[1]

    p = len(kernel_sizes_list[0])
    q = len(kernel_sizes_list[1])
    use_residual_list = [[0] + ([1] * (p - 1)), [0] + ([1] * (q // 2 - 1)) + [0] + ([1] * (q - q // 2 - 1))]

    down_channels_list = [dim1, dim2]
    sinet = SINet(classes, [16, classes], down_channels_list, kernel_sizes_list, scale_factors_list, use_residual_list, in_size=in_size, aux=aux)
    return sinet


if __name__ == "__main__":
    inp = tf.random.uniform((2, 512, 512, 3), dtype=tf.float32)
    sinet = get_nie_sinet(5, False, in_size=(512, 512))
    print(sinet(inp).shape)
    print(sinet.summary())
    # for weight in sinet.get_weights():
    #     print("---------->", weight.shape)
    # seblock = ESPBlock([3 for _ in range(10)], [2 for _ in range(10)], use_residual=True)
    # classes = 5
    # sbencoder = SBEncoder(filters=classes,
    #                       init_block_channels=[16, classes],
    #                       down_channels_list=down_channels_list,
    #                       kernel_sizes_list=kernel_sizes_list,
    #                       scale_factors_list=scale_factors_list,
    #                       use_residual_list=use_residual_list)
    # y1, y2, y3 = sbencoder(inp)
