from fcos_backbone import get_backbone_outputs, FPN, ConvBlock
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_addons as tfa


# ======================= FCOS Postprocessor =========================== #
class Scale(K.layers.Layer):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.init_value = init_value

    def call(self, inputs, *args, **kwargs):
        return inputs * self.init_value


class FCOSHead(K.Model):
    def __init__(self, n_classes, fpn_strides=None, num_convs=4, norm_reg_targets=False, centerness_on_reg=False,
                 use_dcn=False, norm_layer=tfa.layers.GroupNormalization):
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
        self.cls_logits = ConvBlock(self.n_classes, 3, 1)
        self.bbox_pred = ConvBlock(4, 3, 1)
        self.centerness = ConvBlock(1, 3, 1)
        self.scales = [Scale(1.0) for _ in range(len(fpn_strides))]

    def build(self, input_shape):
        self.cls_tower = K.Sequential()
        self.bbox_tower = K.Sequential()
        [self.cls_tower.add(ConvBlock(input_shape[0][-1], 3, 1, "same", norm_layer=self.norm_layer, activation="relu"))
         for _
         in range(self.num_convs)]
        [self.bbox_tower.add(ConvBlock(input_shape[0][-1], 3, 1, "same", norm_layer=self.norm_layer, activation="relu"))
         for
         _ in range(self.num_convs)]

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
                if training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(tf.math.exp(bbox_pred))
        return logits, bbox_reg, centerness


# ======================= FCOS Postprocessor =========================== #
def boxlist_ml_nms(
        boxlist, nms_thresh, max_proposals=-1, score_field="scores", label_field="labels"
):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    scores = boxlist.get_field(score_field)
    # keep = _box_ml_nms(boxes, scores, labels.float(), nms_thresh)
    keep = tf.image.non_max_suppression(boxes, scores, nms_thresh)
    if max_proposals > 0:
        keep = keep[:max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


class Container:
    """
    Help class for manage boxes, labels, etc...
    Not inherit dict due to `default_collate` will change dict's subclass to dict.
    """

    def __init__(self, *args, **kwargs):
        self._data_dict = dict(*args, **kwargs)

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

    def __getitem__(self, key):
        return self._data_dict[key]

    def __iter__(self):
        return self._data_dict.__iter__()

    def __setitem__(self, key, value):
        self._data_dict[key] = value

    def _call(self, name, *args, **kwargs):
        keys = list(self._data_dict.keys())
        for key in keys:
            value = self._data_dict[key]
            if hasattr(value, name):
                self._data_dict[key] = getattr(value, name)(*args, **kwargs)
        return self

    def to(self, *args, **kwargs):
        return self._call("to", *args, **kwargs)

    def numpy(self):
        return self._call("numpy")

    def resize(self, size):
        """resize boxes
        Args:
            size: (width, height)
        Returns:
            self
        """
        img_width = getattr(self, "img_width", -1)
        img_height = getattr(self, "img_height", -1)
        assert img_width > 0 and img_height > 0
        assert "boxes" in self._data_dict
        boxes = self._data_dict["boxes"]
        new_width, new_height = size
        boxes[:, 0::2] *= new_width / img_width
        boxes[:, 1::2] *= new_height / img_height
        return self

    def __repr__(self):
        return self._data_dict.__repr__()


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    size = bboxes[0].size
    mode = bboxes[0].mode
    fields = set(bboxes[0].fields())

    cat_boxes = BoxList(tf.concat([bbox.bbox for bbox in bboxes], axis=0), size, mode)

    for field in fields:
        data = tf.concat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        # device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = tf.convert_to_tensor(bbox, dtype=tf.float32)
        if len(bbox.shape) != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.shape(-1) != 4:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = tf.concat((xmin, ymin, xmax, ymax), axis=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = tf.concat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), axis=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = tf.split(self.bbox, 1, axis=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = tf.split(self.bbox, 1, axis=-1)
            return (
                xmin,
                ymin,
                xmin + tf.clip_by_value(w - TO_REMOVE, clip_value_min=0, clip_value_max=w),
                ymin + tf.clip_by_value(h - TO_REMOVE, clip_value_min=0, clip_value_max=w),
            )
        else:
            raise RuntimeError("Should not be here")

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox[:, 0] = tf.clip_by_value(self.bbox[:, 0], clip_value_min=0, clip_value_max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1] = tf.clip_by_value(self.bbox[:, 1], clip_value_min=0, clip_value_max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2] = tf.clip_by_value(self.bbox[:, 2], clip_value_min=0, clip_value_max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3] = tf.clip_by_value(self.bbox[:, 3], clip_value_min=0, clip_value_max=self.size[1] - TO_REMOVE)
        # self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        # self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        # self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)

        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (
                    box[:, 3] - box[:, 1] + TO_REMOVE
            )
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = tf.unstack(xywh_boxes, dim=-1)
    # keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    keep = tf.squeeze(tf.stack(tf.experimental.numpy.nonzero((ws >= min_size) & (hs >= min_size)), axis=1), axis=1)
    return boxlist[keep]


class FCOSPostProcessor(K.layers.Layer):
    def __init__(self,
                 pre_nms_thresh,
                 pre_nms_top_n,
                 nms_thresh,
                 fpn_post_nms_top_n,
                 min_size,
                 num_classes,
                 bbox_aug_enabled=False):
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled

    def call_on_single_map(self, locations, box_cls, box_regression, centerness, image_sizes):
        _, H, W, C = tf.unstack(tf.shape(box_cls))
        N = 1  # Done as for inference batch size is mostly 1, TODO: Add dynamic batch inference later
        box_cls = tf.nn.sigmoid(tf.reshape(box_cls, (N, H * W, C)))
        box_regression = tf.reshape(box_regression, (N, H * W, 4))
        centerness = tf.nn.sigmoid(tf.reshape(centerness, (N, H * W, 1)))

        # candidate_inds = tf.cond(box_cls > self.pre_nms_thresh, True,
        #                          False)  # replaced box_cls > self.pre_nms_thresh with this to hide pycharm prompt
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = tf.reshape(candidate_inds, (N, -1))
        pre_nms_top_n = tf.reduce_sum(tf.cast(pre_nms_top_n, dtype=tf.float32), axis=-1)
        pre_nms_top_n = tf.clip_by_value(pre_nms_top_n, clip_value_min=0, clip_value_max=self.pre_nms_top_n)

        box_cls = centerness * box_cls
        results = []

        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]  # This is still shape (H*W, C)
            per_box_cls = per_box_cls[
                per_candidate_inds]  # Remove anything below the nms_thresh, (H*W, C) to -> (n) where n is number of values over nms

            per_candidate_nonzeros = tf.stack(tf.experimental.numpy.nonzero(per_candidate_inds), axis=1)
            per_box_loc = tf.cast(per_candidate_nonzeros[:, 0], dtype=tf.int64)
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if tf.reduce_sum(per_candidate_inds) > per_pre_nms_top_n:
                per_box_cls, top_k_indices = tf.math.top_k(per_box_cls, per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = tf.stack([per_locations[:, 0] - per_box_regression[:, 0],
                                   per_locations[:, 1] - per_box_regression[:, 1],
                                   per_locations[:, 0] - per_box_regression[:, 2],
                                   per_locations[:, 1] - per_box_regression[:, 3]], axis=1)
            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", tf.math.sqrt(per_box_cls))
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)
        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh = tf.sort(cls_scores, axis=-1)[number_of_detections - self.fpn_post_nms_top_n + 1]
                keep = cls_scores >= image_thresh
                keep = tf.squeeze(tf.stack(tf.experimental.numpy.nonzero(keep), axis=1), axis=1)
                result = result[keep]
            results.append(result)
        return results

    def call(self, locations, box_cls, box_regression, centerness, image_sizes):
        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(
                zip(locations, box_cls, box_regression, centerness)
        ):
            sampled_boxes.append(
                self.call_on_single_map(l, o, b, c, image_sizes)
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if not self.bbox_aug_enabled:
            boxlists = self.select_over_all_levels(boxlists)

        results = []
        for box in boxlists:
            container = Container(
                boxes=box.bbox,
                labels=box.get_field("labels"),
                scores=box.get_field("scores"),
            )
            container.img_width = image_sizes[1]
            container.img_height = image_sizes[0]
            results.append(container)
        return results


# ======================= FCOS Loss Calculation =========================== #
class IOULoss(K.losses.Loss):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def call(self, y_true, y_pred, weight=None):
        pred_left = y_pred[:, 0]
        pred_top = y_pred[:, 1]
        pred_right = y_pred[:, 2]
        pred_bottom = y_pred[:, 3]

        target_left = y_true[:, 0]
        target_top = y_true[:, 1]
        target_right = y_true[:, 2]
        target_bottom = y_true[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = tf.reduce_min(pred_left, target_left) + tf.reduce_min(
            pred_right, target_right
        )
        g_w_intersect = tf.reduce_max(pred_left, target_left) + tf.reduce_max(
            pred_right, target_right
        )
        h_intersect = tf.reduce_min(pred_bottom, target_bottom) + tf.reduce_min(
            pred_top, target_top
        )
        g_h_intersect = tf.reduce_max(pred_bottom, target_bottom) + tf.reduce_max(
            pred_top, target_top
        )
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loss_type == "iou":
            losses = -tf.math.log(ious)
        elif self.loss_type == "linear_iou":
            losses = 1 - ious
        elif self.loss_type == "giou":
            losses = 1 - gious
        else:
            raise NotImplementedError
        if weight is not None and tf.reduce_sum(weight) > 0:
            return tf.reduce_sum(losses * weight)
        return tf.reduce_sum(losses)


class FCOSLossComputation(object):
    def __init__(self, fpn_strides=None, center_sampling_radius=0, iou_loss_type="iou", norm_reg_targets=False,
                 object_sizes_of_interest=None):
        super(FCOSLossComputation, self).__init__()
        if object_sizes_of_interest is None:
            object_sizes_of_interest = [[-1, 32], [32, 64], [64, 128], [128, 256], [256, 100000000], ]
        if fpn_strides is None:
            fpn_strides = [8, 16, 32, 64, 128]
        self.cls_loss_func = tfa.losses.SigmoidFocalCrossEntropy()
        self.fpn_strides = fpn_strides
        self.center_sampling_radius = center_sampling_radius
        self.iou_loss_type = iou_loss_type
        self.norm_reg_targets = norm_reg_targets
        self.object_sizes_of_interest = object_sizes_of_interest
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = K.losses.BinaryCrossentropy(from_logits=True)

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        gt = tf.tile(gt[tf.newaxis], (len(gt_xs), 1, 1))
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = tf.zeros(gt.shape)
        # no gt
        if tf.reduce_sum(center_x[..., 0]) == 0:
            # return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
            return tf.zeros(gt_xs.shape, dtype=tf.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = tf.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = tf.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = tf.where(
                xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = tf.where(
                ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, tf.newaxis] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, tf.newaxis]
        top = gt_ys[:, tf.newaxis] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, tf.newaxis]
        center_bbox = tf.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):  # TODO: Check how this function behaves for empty targets.
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = tf.convert_to_tensor(self.object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                # object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
                tf.repeat(object_sizes_of_interest_per_level[tf.newaxis], (len(points_per_level), 1))
            )

        expanded_object_sizes_of_interest = tf.concat(
            expanded_object_sizes_of_interest, axis=0
        )
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = tf.concat(points, axis=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = tf.unstack(labels[i], num_points_per_level, axis=0)

            reg_targets[i] = tf.unstack(reg_targets[i], num_points_per_level, axis=0)

        labels_level_first = []

        reg_targets_level_first = []
        for level in range(len(points)):

            labels_level_first.append(
                tf.concat([labels_per_im[level] for labels_per_im in labels], axis=0)
            )

            reg_targets_per_level = tf.concat(
                [reg_targets_per_im[level] for reg_targets_per_im in reg_targets], axis=0
            )

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(
            self, locations, targets, object_sizes_of_interest
    ):
        INF = 1e9
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]["boxes"]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")

            area = targets_per_im.area()

            l = xs[:, tf.newaxis] - bboxes[:, 0][tf.newaxis]
            t = ys[:, tf.newaxis] - bboxes[:, 1][tf.newaxis]
            r = bboxes[:, 2][tf.newaxis] - xs[:, tf.newaxis]
            b = bboxes[:, 3][tf.newaxis] - ys[:, tf.newaxis]
            reg_targets_per_im = tf.stack([l, t, r, b], axis=2)
            if reg_targets_per_im.shape[1] == 0:
                continue

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs,
                    ys,
                    radius=self.center_sampling_radius,
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                # is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
                is_in_boxes = tf.reduce_min(reg_targets_per_im, axis=2) > 0

            # max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            max_reg_targets_per_im = tf.reduce_max(reg_targets_per_im, axis=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = (
                                            max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]
                                    ) & (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            # locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area = tf.repeat(area[tf.newaxis], repeats=[len(locations), 1])
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            # locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)
            locations_to_min_area, locations_to_gt_inds = tf.reduce_min(locations_to_gt_area, axis=1), tf.argmin(
                locations_to_gt_area, axis=1)

            reg_targets_per_im = reg_targets_per_im[
                range(len(locations)), locations_to_gt_inds
            ]
            labels_per_im = labels_per_im[locations_to_gt_inds]

            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)

            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        # centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
        #     top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]
        # )
        centerness = (tf.reduce_min(left_right, axis=-1) / tf.reduce_max(left_right, axis=-1)) * (
                    tf.reduce_min(top_bottom, axis=-1) / tf.reduce_max(top_bottom, axis=-1))
        return tf.math.sqrt(centerness)

    def __call__(self, locations, box_clss, box_regression, centerness, targets=None):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        if targets is None:
            targets = []
        idxs = []
        tars = []
        num_classes = box_clss[0].shape[-1]
        for i, target in enumerate(targets):
            target = target["boxes"]
            if target.bbox.shape[0] != 0:
                idxs.append(i)
                tars.append(targets[i])

        box_cls = []
        for box_cl in box_clss:
            box_cls.append(box_cl[idxs])
        labels, reg_targets = self.prepare_targets(locations, tars)
        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):  # TODO: Fix this Badr
            box_cls_flatten.append(
                tf.reshape(box_cls[l], (-1, num_classes))
            )
            box_regression_flatten.append(
                tf.reshape(box_regression[l], (-1, 4))
            )
            # labels_flatten.append(labels[l].reshape(-1))
            labels_flatten.append(tf.reshape(labels[l], (-1,)))
            # reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            reg_targets_flatten.append(tf.reshape(reg_targets[l], (-1, 4)))
            # centerness_flatten.append(centerness[l].reshape(-1))
            centerness_flatten.append(tf.reshape(centerness[l], (-1,)))

        box_cls_flatten = tf.concat(box_cls_flatten, axis=0)
        box_regression_flatten = tf.concat(box_regression_flatten, axis=0)
        centerness_flatten = tf.concat(centerness_flatten, axis=0)
        labels_flatten = tf.concat(labels_flatten, axis=0)
        reg_targets_flatten = tf.concat(reg_targets_flatten, axis=0)

        # pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        pos_inds = tf.squeeze(tf.stack(tf.experimental.numpy.nonzero(labels_flatten > 0), axis=1), axis=1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        # num_gpus = get_num_gpus()

        # sync num_pos from all gpus
        # total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        # total_num_pos = tf.size(pos_inds)
        # num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = (
            self.cls_loss_func(box_cls_flatten, labels_flatten.int())
        )

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = tf.reduce_sum(
                centerness_targets
            )

            reg_loss = (
                    self.box_reg_loss_func(
                        box_regression_flatten, reg_targets_flatten, centerness_targets
                    )
                    / sum_centerness_targets_avg_per_gpu
            )
            centerness_loss = (
                self.centerness_loss_func(centerness_flatten, centerness_targets)
            )
        else:
            reg_loss = tf.reduce_sum(box_regression_flatten)
            # reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = tf.reduce_sum(centerness_flatten)

        losses = dict(
            reg_loss=reg_loss, cls_loss=cls_loss, centerness_loss=centerness_loss
        )
        return losses


class FCOSLoss(K.losses.Loss):
    def __init__(self, feature_sizes=None, fpn_strides=None, center_sampling_radius=0, iou_loss_type="iou",
                 norm_reg_targets=False,
                 object_sizes_of_interest=None):
        super(FCOSLoss, self).__init__()
        if feature_sizes is None:
            feature_sizes = [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)]
        if object_sizes_of_interest is None:
            object_sizes_of_interest = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 100000000], ]
        if fpn_strides is None:
            fpn_strides = [8, 16, 32, 64, 128]
        # Hyper-params defined #
        self.fpn_strides = fpn_strides
        self.center_sampling_radius = center_sampling_radius
        self.iou_loss_type = iou_loss_type
        self.norm_reg_targets = norm_reg_targets
        self.object_sizes_of_interest = tf.constant(object_sizes_of_interest)
        # =========== Losses ============== #
        self.cls_loss_func = tfa.losses.SigmoidFocalCrossEntropy()
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = K.losses.BinaryCrossentropy(from_logits=True)
        # ========== Hyper-parameters definition ======== #
        self.feature_sizes = feature_sizes
        self.locations = self.compute_locations(feature_sizes)
        self.stacked_locations = tf.concat(self.locations, axis=0)
        self.object_sizes_of_interests = tf.concat(
            [tf.tile(self.object_sizes_of_interest[n][tf.newaxis], multiples=(loc.shape[0], 1)) for n, loc in
             enumerate(self.locations)], axis=0)
        self.inf = 1e9
        self.points_per_level = [feature_size[0] * feature_size[1] for feature_size in self.feature_sizes]

    def compute_locations(self, features_sizes):
        locations = []
        for level, feature_size in enumerate(features_sizes):
            h, w = feature_size[0], feature_size[1]
            locations_per_level = self.compute_locations_per_level(h, w, self.fpn_strides[level])
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride):
        shifts_x = tf.range(0, w * stride, delta=stride, dtype=tf.float32)
        shifts_y = tf.range(0, h * stride, delta=stride, dtype=tf.float32)
        shift_x, shift_y = tf.meshgrid(shifts_x, shifts_y)
        shift_x = tf.reshape(shift_x, (-1,))
        shift_y = tf.reshape(shift_y, (-1,))
        locations = tf.stack((shift_x, shift_y), axis=1) + stride // 2
        return locations

    def compute_targets_for_location(self, bboxes, classes):
        """
        Calculate the regression targets for positive samples in batch given GT
        :param bboxes: Bounding boxes in the batch x1, y1, x2, y2 format, (B, N, 4)
        :param classes: Classes of each bounding box (B, N)
        :param object_sizes_of_interest: Max allowed regression
        :return: list of length batch size B [labels (total_features_length, 1), reg_targets (total_feature_length, 4)]
        """
        xs, ys = self.stacked_locations[:, 0], self.stacked_locations[:, 1]
        labels_batch, reg_targets_batch = [], []
        for bboxes_img, classes_img in zip(bboxes, classes):
            if bboxes_img.shape[0] < 1:
                continue
            areas = tf.convert_to_tensor([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes_img])
            l = xs[:, tf.newaxis] - bboxes_img[:, 0]
            r = -xs[:, tf.newaxis] + bboxes_img[:, 2]
            t = ys[:, tf.newaxis] - bboxes_img[:, 1]
            b = -ys[:, tf.newaxis] + bboxes_img[:, 3]
            reg_targets = tf.stack([l, t, r, b], axis=2)
            # TODO: Add center sampling later
            is_in_boxes = tf.reduce_min(reg_targets, axis=-1) > 0
            max_reg_targets_per_im = tf.cast(tf.reduce_max(reg_targets, axis=-1), dtype=tf.int32)
            is_cared_in_the_level = (max_reg_targets_per_im >= self.object_sizes_of_interests[:, 0:1]) & (
                        max_reg_targets_per_im <= self.object_sizes_of_interests[:, 1:])
            locations_to_gt_area = tf.tile(areas[tf.newaxis], (self.stacked_locations.shape[0], 1))
            # locations_to_gt_area[is_in_boxes] = self.inf
            # locations_to_gt_area[is_cared_in_the_level == 0] = self.inf
            locations_to_gt_area = tf.where(is_in_boxes, locations_to_gt_area, self.inf)
            locations_to_gt_area = tf.where(is_cared_in_the_level, locations_to_gt_area, self.inf)
            locations_to_min_area, locations_to_gt_inds = \
                tf.reduce_min(locations_to_gt_area, axis=1), tf.argmin(locations_to_gt_area, axis=1)
            # classes_img = classes_img[locations_to_gt_inds]
            inds_from_class_axis = tf.concat([tf.cast(tf.range(reg_targets.shape[0])[..., tf.newaxis], dtype=tf.int64),
                       locations_to_gt_inds[..., tf.newaxis]], axis=-1)
            reg_targets_img = tf.gather_nd(reg_targets, inds_from_class_axis)
            # reg_targets_img = tf.gather(bboxes_img, locations_to_gt_inds)
            classes_img = tf.gather(classes_img, locations_to_gt_inds)

            classes_img = tf.where(locations_to_min_area == self.inf, 0, classes_img)

            labels_batch.append(classes_img)

            reg_targets_batch.append(reg_targets_img)
        return labels_batch, reg_targets_batch

    def prepare_targets(self, gt_bboxes, gt_classes):
        labels, reg_targets = self.compute_targets_for_location(gt_bboxes, gt_classes)
        regs = tf.split(reg_targets, num_or_size_splits=self.points_per_level, axis=1)
        if self.norm_reg_targets:
            regs = [regs[n]/fpn_stride for n, fpn_stride in enumerate(self.fpn_strides)]
        labs = tf.split(labels, num_or_size_splits=self.points_per_level, axis=1)
        final_regs = [tf.reshape(reg, (-1, reg.shape[-1])) for reg in regs]
        final_labs = [tf.reshape(lab, (-1,)) for lab in labs]
        return final_labs, final_regs

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (tf.reduce_min(left_right, axis=-1) / tf.reduce_max(left_right, axis=-1)) * \
                     (tf.reduce_min(top_bottom, axis=-1) / tf.reduce_max(top_bottom, axis=-1))
        return tf.math.sqrt(centerness)

    def call(self, y_true, y_pred):
        box_cls, box_reg, box_centerness = y_pred
        bboxes, labels = y_true
        labels, reg_targets = self.prepare_targets(bboxes, labels)
        box_cls_flatten = [tf.reshape(x, shape=(-1, x.shape[-1])) for x in box_cls]
        box_reg_flatten = [tf.reshape(x, shape=(-1, x.shape[-1])) for x in box_reg]
        box_centerness_flatten = [tf.reshape(x, shape=(-1,)) for x in box_centerness]
        box_cls_flatten = tf.concat(box_cls_flatten, axis=0)
        box_reg_flatten = tf.concat(box_reg_flatten, axis=0)
        box_centerness_flatten = tf.concat(box_centerness_flatten, axis=0)
        labels_flatten = tf.concat(labels, axis=0)
        reg_targets_flatten = tf.concat(reg_targets, axis=0)
        inds = tf.experimental.numpy.nonzero(labels_flatten > 0)[0]
        box_cls_flatten = tf.gather(box_cls_flatten, inds)
        box_reg_flatten = tf.gather(box_reg_flatten, inds)
        box_centerness_flatten = tf.gather(box_centerness_flatten, inds)
        cls_loss = tf.reduce_sum(self.cls_loss_func(labels_flatten, box_cls_flatten)) / inds
        if inds > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = tf.reduce_sum(self.box_reg_loss_func(reg_targets_flatten, box_reg_flatten, centerness_targets)) /inds
            cnt_loss = tf.reduce_sum(self.centerness_loss_func(centerness_targets, box_centerness_flatten)) /inds
        else:
            reg_loss = tf.reduce_sum(box_reg_flatten)
            cnt_loss = tf.reduce_sum(0)
        return reg_loss, cls_loss, cnt_loss



if __name__ == "__main__":
    loss_calc = FCOSLoss(norm_reg_targets=True)
    batch_size = 4
    n_classes = 11
    bboxes = tf.random.uniform((batch_size, 17, 4), maxval=512, dtype=tf.float32)
    classes = tf.random.uniform((batch_size, 17,), maxval=n_classes, dtype=tf.int32)
    # loss_calc.compute_targets_for_location(bboxes, classes)
    # loss_calc.prepare_targets(bboxes, classes)
    s = [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)]
    box_cls = [tf.random.uniform((batch_size, shp[0], shp[1], n_classes)) for shp in s]
    box_reg = [tf.random.uniform((batch_size, shp[0], shp[1], 4)) for shp in s]
    box_cent = [tf.random.uniform((batch_size, shp[0], shp[1])) for shp in s]
    loss_calc((bboxes, classes), (box_cls, box_reg, box_cent))
