from fcos_backbone import P6P7_level
import tensorflow as tf
import tensorflow.keras as K
from fcos_utils import FCOSHead, FCOSLossComputation, FCOSPostProcessor
from fcos_backbone import FPN


class FCOSModule(K.Model):
    def __init__(self, n_classes=11,
                 fpn_strides=None,
                 img_size=(512, 512),
                 pre_nms_thresh=0.05,
                 pre_nms_top_n=12000,
                 nms_thresh=0.7,
                 fpn_post_nms_top_n=2000,
                 min_size=0):
        super(FCOSModule, self).__init__()
        if fpn_strides is None:
            fpn_strides = [8, 16, 32, 64, 128]
        self.head = FCOSHead(n_classes)
        self.box_selector_test = FCOSPostProcessor(pre_nms_thresh, pre_nms_top_n, nms_thresh, fpn_post_nms_top_n, min_size, n_classes)
        self.fpn_strides = fpn_strides
        self.image_size = [img_size[0], img_size[1]]
        self.loss_evaluator = FCOSLossComputation()

    def build(self, input_shape):
        self.encoder = FPN(input_shape[0][-1], P6P7_level(input_shape[0][-1]))

    def call(self, inputs, training=None, targets=None):
        features = self.encoder(inputs)
        box_cls, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)

        if training:
            loss_dict = self.loss_evaluator(
                locations, box_cls, box_regression, centerness, targets
            )
            detections = (box_cls, box_regression, centerness)
            return detections, loss_dict

        else:
            boxes = self.box_selector_test(
                locations, box_cls, box_regression, centerness, self.image_size
            )

            return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.shape[1], feature.shape[2]
            locations_per_level = self.compute_locations_per_level(h, w, self.fpn_strides[level])
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride):
        shifts_x = tf.range(0, w * stride, delta=stride, dtype=tf.float32)
        shifts_y = tf.range(0, h * stride, delta=stride, dtype=tf.float32)
        shift_y, shift_x = tf.meshgrid(shifts_y, shifts_x)
        shift_x = tf.reshape(shift_x, (-1,))
        shift_y = tf.reshape(shift_y, (-1,))
        locations = tf.stack((shift_x, shift_y), axis=1) + stride // 2
        return locations
