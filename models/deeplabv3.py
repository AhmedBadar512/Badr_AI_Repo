import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Conv2D
from backbones import get_backbone
import layers


class ASPP(tf.keras.layers.Layer):
    def __init__(self, activation='relu'):
        super().__init__()
        self.convblock1 = layers.ConvBlock(256, 1, padding="same", use_bias=False, activation=activation)
        self.convblock2 = layers.ConvBlock(256, 1, padding="same", use_bias=False, activation=activation)
        self.convblock3 = layers.ConvBlock(256, 3, dilation_rate=6, padding="same", use_bias=False,
                                           activation=activation)
        self.convblock4 = layers.ConvBlock(256, 3, dilation_rate=12, padding="same", use_bias=False,
                                           activation=activation)
        self.convblock5 = layers.ConvBlock(256, 3, dilation_rate=18, padding="same", use_bias=False,
                                           activation=activation)
        self.convblock6 = layers.ConvBlock(256, 1, dilation_rate=1, padding="same", use_bias=False,
                                           activation=activation)

    def build(self, input_shape):
        self.pool2d = AveragePooling2D(pool_size=(input_shape[1], input_shape[2]), name='average_pooling')

    def call(self, inputs, *args, **kwargs):
        x = self.pool2d(inputs)
        x_pool = self.convblock1(x)
        x_1 = self.convblock2(x_pool)
        x_6 = self.convblock3(x_1)
        x_12 = self.convblock4(x_6)
        x_18 = self.convblock5(x_12)
        x_concat = tf.concat([x_pool, x_1, x_6, x_12, x_18])

        return self.convblock6(x_concat)


class Deeplabv3plus(tf.keras.Model):
    def __init__(self, backbone="resnet50v2", classes=19, activation='relu', aspp_output_index=2, conv_head_index=0):
        super().__init__()
        self.backbone = backbone
        self.aspp = ASPP()
        self.convblock1 = layers.ConvBlock(48, 1, padding="same", use_bias=False, activation=activation)
        self.convblock2 = layers.ConvBlock(256, 3, padding="same", use_bias=False, activation=activation)
        self.convblock3 = layers.ConvBlock(256, 3, padding="same", use_bias=False, activation=activation)
        self.conv_f = Conv2D(classes, (1, 1), name='output_layer')
        self.aspp_output_index = aspp_output_index
        self.conv_head_index = conv_head_index

    def build(self, input_shape):
        self.backbone = get_backbone(self.backbone, input_shape=input_shape[1:])
        self.get_resnet_backbone_outputs(self.backbone)
        self.backbone_pruned = Model(inputs=self.backbone.input,
                                     outputs=
                                     [self.backbone_outputs[self.aspp_output_index].output,
                                      self.backbone_outputs[self.conv_head_index].output],
                                     name='DeepLabV3_Plus_backbone')
        del self.backbone

    def get_resnet_backbone_outputs(self, backbone, tmp_layer=None):
        self.backbone_outputs = []
        for layer in backbone.layers:
            if "out" in layer.name:
                if len(self.backbone_outputs) == 0:
                    self.backbone_outputs.append(layer)
                    tmp_layer = layer
                    continue
                self.backbone_outputs.append(layer)
                if layer.output_shape[1:3] == tmp_layer.output_shape[1:3]:
                    self.backbone_outputs.pop(-2)
                tmp_layer = layer

    def call(self, inputs, training=None, mask=None):
        x_aspp, x_b = self.backbone_pruned(inputs)
        x_a = tf.image.resize(x_aspp, tf.shape(x_b)[1:3])
        x_b = self.convblock1(x_b)
        x_ab = tf.keras.layers.concatenate([x_a, x_b])
        x_ab = self.convblock2(x_ab)
        x_ab = self.convblock3(x_ab)
        x_ab = tf.image.resize(x_ab, tf.shape(inputs)[1:3])
        return self.conv_f(x_ab)


if __name__ == "__main__":
    deeplab_model = Deeplabv3plus()
    a = deeplab_model(tf.random.uniform((1, 368, 640, 3)))
    print(a.shape)
