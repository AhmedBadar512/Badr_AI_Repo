import tensorflow as tf
import tensorflow.keras as K


class Upscaler(K.layers.Layer):
    def __init__(self, f):
        super().__init__()
        self.upsample = SubpixelConv2D(2)
        self.conv = K.layers.Conv2D(f, 3, padding="same", activation=tf.nn.leaky_relu)
        self.bn = K.layers.BatchNormalization()

    def call(self, inputs, *args, **kwargs):
        x = self.upsample(inputs)
        x = self.conv(x)
        x = self.bn(x)
        return x


def hw_flatten(x):
    return tf.reshape(x, shape=tf.constant([-1, x.shape[1] * x.shape[2], x.shape[-1]], dtype=tf.int32))


class SelfAttentionBlock(K.layers.Layer):
    def __init__(self, nc, squeeze_factor=4, w_l2=1e-4):
        super().__init__()
        self.f = K.layers.Conv2D(nc // squeeze_factor, 1, kernel_regularizer=K.regularizers.l2(w_l2), padding="same")
        self.g = K.layers.Conv2D(nc // squeeze_factor, 1, kernel_regularizer=K.regularizers.l2(w_l2), padding="same")
        self.h = K.layers.Conv2D(nc, 1, kernel_regularizer=K.regularizers.l2(w_l2), padding="same")
        self.flatten = K.layers.Flatten()
        self.gamma = tf.Variable([0], dtype=tf.float32)
        self.o = K.layers.Conv2D(nc, 1, 1, padding="same")

    def call(self, inputs, **kwargs):
        f = self.f(inputs)
        g = self.g(inputs)
        h = self.h(inputs)
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)
        beta = tf.nn.softmax(s)
        o = tf.matmul(beta, hw_flatten(h))
        shp = list(inputs.shape)
        shp[0] = -1
        o = tf.reshape(o, shape=shp)
        o = self.o(o)
        return self.gamma * o + inputs


class SubpixelConv2D(K.layers.Layer):
    """ Subpixel Conv2D Layer
    upsampling a layer from (h, w, c) to (h*r, w*r, c/(r*r)),
    where r is the scaling factor, default to 4
    # Arguments
    upsampling_factor: the scaling factor
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        the second and the third dimension increased by a factor of
        `upsampling_factor`; the last layer decreased by a factor of
        `upsampling_factor^2`.
    # References
        Real-Time Single Image and Video Super-Resolution Using an Efficient
        Sub-Pixel Convolutional Neural Network Shi et Al. https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upsampling_factor=4, **kwargs):
        super(SubpixelConv2D, self).__init__(**kwargs)
        self.upsampling_factor = upsampling_factor

    def build(self, input_shape):
        last_dim = input_shape[-1]
        factor = self.upsampling_factor * self.upsampling_factor
        if last_dim % (factor) != 0:
            raise ValueError('Channel ' + str(last_dim) + ' should be of '
                                                          'integer times of upsampling_factor^2: ' +
                             str(factor) + '.')

    def call(self, inputs, **kwargs):
        return tf.nn.depth_to_space(inputs, self.upsampling_factor)

    def get_config(self):
        config = {'upsampling_factor': self.upsampling_factor, }
        base_config = super(SubpixelConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        factor = self.upsampling_factor * self.upsampling_factor
        input_shape_1 = None
        if input_shape[1] is not None:
            input_shape_1 = input_shape[1] * self.upsampling_factor
        input_shape_2 = None
        if input_shape[2] is not None:
            input_shape_2 = input_shape[2] * self.upsampling_factor
        dims = [input_shape[0],
                input_shape_1,
                input_shape_2,
                int(input_shape[3] / factor)
                ]
        return tuple(dims)
