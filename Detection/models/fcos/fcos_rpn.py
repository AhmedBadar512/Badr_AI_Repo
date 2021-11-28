from fcos_backbone import get_backbone_outputs, FPN, ConvBlock
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_addons as tfa


class Scale(K.layers.Layer):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.init_value = init_value

    def call(self, inputs, *args, **kwargs):
        return inputs * self.init_value


class FCOSHead(K.Model):
    def __init__(self, n_classes, fpn_strides=None, num_convs=4, norm_reg_targets=False, centerness_on_reg=False, use_dcn=False, norm_layer=tfa.layers.GroupNormalization):
        super(FCOSHead, self).__init__()
        if fpn_strides is None:
            fpn_strides = [8, 16, 32, 64, 128]
        self.n_classes = n_classes - 1
        self.fpn_strides = fpn_strides
        self.norm_reg_targets = norm_reg_targets
        self.centerness_on_reg = centerness_on_reg
        self.use_dcn = use_dcn  # TODO: Implement and add later
        # TODO: Add and implement FCOS prob
        self.num_convs = num_convs
        self.norm_layer = norm_layer
        self.cls_logits = ConvBlock(n_classes, 3, 1)
        self.bbox_pred = ConvBlock(4, 3, 1)
        self.centerness = ConvBlock(1, 3, 1)
        self.scales = [Scale(1.0) for _ in range(len(fpn_strides))]

    def build(self, input_shape):
        self.cls_tower = K.Sequential()
        self.bbox_tower = K.Sequential()
        [self.cls_tower.add(ConvBlock(input_shape[0][-1], 3, 1, 1, norm_layer=self.norm_layer, activation="relu")) for _ in range(self.num_convs)]
        [self.bbox_tower.add(ConvBlock(input_shape[0][-1], 3, 1, 1, norm_layer=self.norm_layer, activation="relu")) for _ in range(self.num_convs)]

    def call(self, inputs, training=None, mask=None):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(inputs):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = tf.nn.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(tf.math.exp(bbox_pred))
        return logits, bbox_reg, centerness
