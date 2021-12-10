import tensorflow as tf
import tensorflow.keras as K
import tensorflow_addons as tfa


class ConvBlock(K.layers.Layer):
    """ ConBlock layer consists of Conv2D + Normalization + Activation.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='same',
                 use_bias=True,
                 norm_layer=None,
                 activation='linear',
                 sn=False,
                 dilation_rate=(1, 1),
                 **kwargs):
        super(ConvBlock, self).__init__()
        self.conv2d = K.layers.Conv2D(filters,
                                      kernel_size,
                                      strides,
                                      padding,
                                      dilation_rate=dilation_rate,
                                      use_bias=use_bias,
                                      **kwargs)
        if sn:
            self.conv2d = tfa.layers.SpectralNormalization(self.conv2d)
        self.activation = K.layers.Activation(activation)
        if norm_layer:
            self.normalization = norm_layer()
        else:
            self.normalization = tf.identity

    def call(self, inputs, training=None, **kwargs):
        x = self.conv2d(inputs)
        x = self.normalization(x)
        x = self.activation(x)

        return x


def extract_outputs_at_spatial_steps(resnet_base):
    names = [l.name for l in resnet_base.layers if "_out" in l.name]
    prev_name = names[0]
    outputs = []
    for name in names:
        if prev_name.split("_")[0] == name.split("_")[0]:
            prev_name = name
            continue
        else:
            outputs.append(resnet_base.get_layer(prev_name).output)
            prev_name = name
    outputs.append(resnet_base.get_layer(names[-1]).output)
    return outputs


class P6P7_level(K.layers.Layer):
    def __init__(self, out_channels=64):
        super().__init__()
        self.conv6 = ConvBlock(out_channels, 3, 2, padding='same')
        self.conv7 = ConvBlock(out_channels, 3, 2, padding='same', activation='relu')
        self.use_p5 = False

    def build(self, input_shape):
        if input_shape[0][-1] == input_shape[1][-1]:
            self.use_p5 = True

    def call(self, inputs, *args, **kwargs):
        assert len(inputs) == 2
        c5, p5 = inputs
        x = p5 if self.use_p5 else c5
        p6 = self.conv6(x)
        p7 = self.conv7(p6)
        return [p6, p7]


class LastLevelMaxPool(K.layers.Layer):
    def __init__(self):
        super().__init__()
        self.maxpool = K.layers.MaxPool2D(1, 2)

    def call(self, inputs, *args, **kwargs):
        return self.maxpool(inputs)


class FPN(K.Model):
    def __init__(self, out_channels, top_blocks=None, norm_layer=K.layers.BatchNormalization):
        super().__init__()
        self.out_channels = out_channels
        self.top_blocks = top_blocks
        self.norm_layer = norm_layer

    def build(self, input_shape):
        self.base_convs = [ConvBlock(self.out_channels, 1, padding='same', norm_layer=self.norm_layer) for _ in range(len(input_shape))]
        self.second_convs = [ConvBlock(self.out_channels, 3, 1, padding='same', norm_layer=self.norm_layer) for _ in range(len(input_shape))]

    def call(self, inputs, training=None, mask=None):
        results = []
        last_inner = self.base_convs[-1](inputs[-1])
        results.append(self.second_convs[-1](last_inner))
        for feature, inner_block, layer_block in \
                zip(inputs[:-1][::-1], self.base_convs[:-1][::-1], self.second_convs[:-1][::-1]):
            inner_lateral = inner_block(feature)
            inner_top_down = tf.image.resize(last_inner, tf.shape(inner_lateral)[1:3], method='nearest')
            last_inner = inner_lateral + inner_top_down
            results.insert(0, layer_block(last_inner))
        if isinstance(self.top_blocks, P6P7_level):
            last_results = self.top_blocks([inputs[-1], results[-1]])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.append(last_results)
        return results


def get_backbone_outputs():
    backbone = K.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
    return extract_outputs_at_spatial_steps(backbone)[1:]


if __name__ == "__main__":

    random = tf.random.uniform((1, 512, 512, 3))
    outputs = get_backbone_outputs()
    fpn_outs = FPN(47, P6P7_level())(outputs)
    [print(fpn_out.shape) for fpn_out in outputs]
    print("------------------")
    [print(fpn_out.shape) for fpn_out in fpn_outs]
