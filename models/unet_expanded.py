import tensorflow as tf
import tensorflow.keras as K
import tensorflow_probability as tfp
from .layers import CompDecomp


class UNet_Expanded(K.Model):
    def __init__(self,
                 base_channels=32,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(512, 512),
                 classes=21,
                 aux=False,
                 variational=False,
                 activation='relu',
                 backbone=None,
                 **kwargs):
        super().__init__(**kwargs)
        assert (in_channels > 0)
        assert ((in_size[0] % 32 == 0) and (in_size[1] % 32 == 0))
        self.base_channels = base_channels
        self.in_size = in_size
        self.classes = classes
        self.fixed_size = fixed_size
        self.comp_64 = CompDecomp(64)
        self.comp_32 = CompDecomp(32)
        self.comp_16 = CompDecomp(16)
        self.comp_4 = CompDecomp(4)
        self.in_conv = K.layers.Conv2D(base_channels,
                                       3,
                                       activation=activation,
                                       padding='same',
                                       kernel_initializer='he_normal')
        self.conv1 = [K.layers.Conv2D(base_channels,
                                      3,
                                      activation=activation,
                                      padding='same',
                                      kernel_initializer='he_normal') for _ in range(3)]
        self.conv2 = [K.layers.Conv2D(base_channels * 2,
                                      3,
                                      activation=activation,
                                      padding='same',
                                      kernel_initializer='he_normal') for _ in range(4)]
        self.conv3 = [K.layers.Conv2D(base_channels * 4,
                                      3,
                                      activation=activation,
                                      padding='same',
                                      kernel_initializer='he_normal') for _ in range(4)]
        self.conv4 = [K.layers.Conv2D(base_channels * 8,
                                      3,
                                      activation=activation,
                                      padding='same',
                                      kernel_initializer='he_normal') for _ in range(4)]
        if not variational:
            self.conv5 = [K.layers.Conv2D(base_channels * 16,
                                          3,
                                          activation=activation,
                                          padding='same',
                                          kernel_initializer='he_normal') for _ in range(2)]
        else:
            self.conv5 = [tfp.layers.Convolution2DReparameterization(base_channels * 16,
                                                                     3,
                                                                     activation=activation,
                                                                     padding='same'),
                          K.layers.Conv2D(base_channels * 16,
                                          3,
                                          activation=activation,
                                          padding='same',
                                          kernel_initializer='he_normal')
                          ]
        self.dropout1 = K.layers.Dropout(0.1)
        self.dropout2 = K.layers.Dropout(0.2)
        self.dropout3 = K.layers.Dropout(0.3)
        self.pool = K.layers.MaxPool2D(pool_size=(2, 2))
        self.bn = []
        for _ in range(18):
            self.bn.append(K.layers.BatchNormalization())
        self.up6 = K.layers.Conv2DTranspose(base_channels * 8, (3, 3), strides=(2, 2), padding="same")
        self.up7 = K.layers.Conv2DTranspose(base_channels * 4, (3, 3), strides=(2, 2), padding="same")
        self.up8 = K.layers.Conv2DTranspose(base_channels * 2, (3, 3), strides=(2, 2), padding="same")
        self.up9 = K.layers.Conv2DTranspose(base_channels, (3, 3), strides=(2, 2), padding="same")
        self.conv10 = K.layers.Conv2D(classes, 1, strides=1, padding="same")

    def call(self, inputs, training=None, mask=None):
        c1 = self.in_conv(inputs)
        c1 = self.bn[0](c1, training=training)
        c1 = self.dropout1(c1)
        c1 = self.comp_64(c1)
        c1 = self.conv1[0](c1)
        c1 = self.bn[1](c1, training=training)
        c1 = self.comp_64(c1)
        p1 = self.pool(c1)

        c2 = self.conv2[0](p1)
        c2 = self.bn[2](c2, training=training)
        c2 = self.dropout1(c2)
        c2 = self.comp_64(c2)
        c2 = self.conv2[1](c2)
        c2 = self.bn[3](c2, training=training)
        c2 = self.comp_64(c2)
        p2 = self.pool(c2)

        c3 = self.conv3[0](p2)
        c3 = self.bn[4](c3, training=training)
        c3 = self.dropout2(c3)
        c3 = self.comp_32(c3)
        c3 = self.conv3[1](c3)
        c3 = self.bn[5](c3, training=training)
        c3 = self.comp_32(c3)
        p3 = self.pool(c3)

        c4 = self.conv4[0](p3)
        c4 = self.bn[6](c4, training=training)
        c4 = self.dropout2(c4)
        c4 = self.comp_16(c4)
        c4 = self.conv4[1](c4)
        c4 = self.bn[7](c4, training=training)
        c4 = self.comp_16(c4)
        p4 = self.pool(c4)

        c5 = self.conv5[0](p4)
        c5 = self.bn[8](c5, training=training)
        c5 = self.dropout3(c5)
        c5 = self.comp_4(c5)
        c5 = self.conv5[1](c5)
        c5 = self.bn[9](c5, training=training)
        c5 = self.comp_4(c5)

        u6 = self.up6(c5)
        u6 = K.layers.concatenate([u6, c4])
        c6 = self.conv4[2](u6)
        c6 = self.bn[10](c6, training=training)
        c6 = self.dropout2(c6)
        c6 = self.conv4[3](c6)
        c6 = self.bn[11](c6, training=training)

        u7 = self.up7(c6)
        u7 = K.layers.concatenate([u7, c3])
        c7 = self.conv3[2](u7)
        c7 = self.bn[12](c7, training=training)
        c7 = self.dropout2(c7)
        c7 = self.conv3[3](c7)
        c7 = self.bn[13](c7, training=training)

        u8 = self.up8(c7)
        u8 = K.layers.concatenate([u8, c2])
        c8 = self.conv2[2](u8)
        c8 = self.bn[14](c8, training=training)
        c8 = self.dropout1(c8)
        c8 = self.conv2[3](c8)
        c8 = self.bn[15](c8, training=training)

        u9 = self.up9(c8)
        u9 = K.layers.concatenate([u9, c1])
        c9 = self.conv1[1](u9)
        c9 = self.bn[16](c9, training=training)
        c9 = self.dropout1(c9)
        c9 = self.conv1[2](c9)
        c9 = self.bn[17](c9, training=training)

        final = self.conv10(c9)
        return final


if __name__ == "__main__":
    x = tf.random.uniform((1, 512, 512, 3))
    unet = UNet_Expanded(32)
    unet.build(input_shape=(None, None, None, 3))
    print(unet.summary())
    print(unet(x).shape)
