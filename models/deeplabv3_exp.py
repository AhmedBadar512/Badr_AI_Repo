import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from .backbones import get_backbone
from .layers import ConvBlock, CompDecomp


class AtrousSpatialPyramidPooling(tf.keras.layers.Layer):
    """Atrous Spatial Pyramid Pooling layer for DeepLabV3+ architecture."""

    # !pylint:disable=too-many-instance-attributes
    def __init__(self):
        super(AtrousSpatialPyramidPooling, self).__init__()

        # layer architecture components
        self.avg_pool = None
        self.conv1, self.conv2 = None, None
        self.pool = None
        self.out1, self.out6, self.out12, self.out18 = None, None, None, None
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    @staticmethod
    def _get_conv_block(kernel_size, dilation_rate, use_bias=False):
        return ConvBlock(256,
                         kernel_size=kernel_size,
                         dilation_rate=dilation_rate,
                         padding='same',
                         use_bias=use_bias,
                         kernel_initializer=tf.keras.initializers.he_normal(),
                         activation='relu')

    def build(self, input_shape):
        dummy_tensor = tf.random.normal(input_shape)  # used for calculating
        # output shape of convolutional layers

        self.avg_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(input_shape[-3], input_shape[-2]))

        self.conv1 = AtrousSpatialPyramidPooling._get_conv_block(
            kernel_size=1, dilation_rate=1, use_bias=True)

        self.conv2 = AtrousSpatialPyramidPooling._get_conv_block(
            kernel_size=1, dilation_rate=1)

        dummy_tensor = self.conv1(self.avg_pool(dummy_tensor))

        self.pool = tf.keras.layers.UpSampling2D(
            size=(
                input_shape[-3] // dummy_tensor.shape[1],
                input_shape[-2] // dummy_tensor.shape[2]
            ),
            interpolation='bilinear'
        )

        self.out1, self.out6, self.out12, self.out18 = map(
            lambda tup: AtrousSpatialPyramidPooling._get_conv_block(
                kernel_size=tup[0], dilation_rate=tup[1]
            ),
            [(1, 1), (3, 6), (3, 12), (3, 18)]
        )

    def call(self, inputs, training=None, **kwargs):
        tensor = self.avg_pool(inputs)
        tensor = self.conv1(tensor)
        tensor = self.concat([
            self.pool(tensor),
            self.out1(inputs),
            self.out6(inputs),
            self.out12(
                inputs
            ),
            self.out18(
                inputs
            )
        ])
        tensor = self.conv2(tensor)
        return tensor


class Deeplabv3PlusExpanded(tf.keras.Model):
    def __init__(self, backbone="resnet50", classes=19, activation='relu', **kwargs):
        super().__init__()
        self.backbone = backbone
        self.aspp = AtrousSpatialPyramidPooling()
        self.convblock1 = ConvBlock(48, 1, padding="same", use_bias=False, activation=activation)
        self.convblock2 = ConvBlock(256, 3, padding="same", use_bias=False, activation=activation)
        self.convblock3 = ConvBlock(256, 3, padding="same", use_bias=False, activation=activation)
        self.conv_f = Conv2D(classes, (1, 1), name='output_layer')
        self.outputs_name = {
            "resnet50": {"aspp_layer": "conv4_block6_out", "feature_2_layer": "conv2_block3_out"},
            "resnet50v2": {"aspp_layer": "conv4_block6_out", "feature_2_layer": "conv2_block3_out"},
            "resnet101": {"aspp_layer": "conv4_block23_out", "feature_2_layer": "conv2_block3_out"},
            "resnet101v2": {"aspp_layer": "conv4_block23_out", "feature_2_layer": "conv2_block3_out"},
            "resnet152": {"aspp_layer": "conv4_block36_out", "feature_2_layer": "conv2_block3_out"},
            "xception": {"aspp_layer": "block13_sepconv2_bn", "feature_2_layer": "block3_sepconv2_bn"},
        }

    def build(self, input_shape):
        self.get_aspp_feature_backbone(input_shape)     # Spatial factor reduction of 32 e.g. 512 -> 16
        self.get_feature_2_backbone(input_shape)        # Spatial factor reduction of 8 e.g. 512 -> 64
        self.exp4 = CompDecomp(4)
        self.exp8 = CompDecomp(8)

    def get_aspp_feature_backbone(self, input_shape):
        backbone = get_backbone(self.backbone, input_shape=input_shape[1:])
        self.backbone_aspp = tf.keras.Model \
            (inputs=backbone.input,
             outputs=backbone.get_layer(self.outputs_name[self.backbone]["aspp_layer"]).output)

    def get_feature_2_backbone(self, input_shape):
        backbone = get_backbone(self.backbone, input_shape=input_shape[1:])
        self.backbone_b = tf.keras.Model(
            inputs=backbone.input,
            outputs=backbone.get_layer(self.outputs_name[self.backbone]["feature_2_layer"]).output)

    def call(self, inputs, training=None, mask=None):
        x_aspp = self.backbone_aspp(inputs)
        x_aspp = self.exp4(x_aspp)
        x_b = self.backbone_b(inputs)
        x_b = self.exp8(x_b)
        x_aspp = self.aspp(x_aspp, training)
        x_a = tf.image.resize(x_aspp, tf.shape(x_b)[1:3])
        x_b = self.convblock1(x_b, training)
        x_ab = tf.keras.layers.concatenate([x_a, x_b])
        x_ab = self.convblock2(x_ab, training)
        x_ab = self.exp8(x_ab)
        x_ab = self.convblock3(x_ab, training)
        x_ab = tf.image.resize(x_ab, tf.shape(inputs)[1:3])
        return self.conv_f(x_ab)


if __name__ == "__main__":
    deeplab_model = Deeplabv3PlusExpanded()
    a = deeplab_model(tf.random.uniform((1, 512, 512, 3)))
    print(a.shape)
