# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.utils import filter_scores_and_topk
from mmdet.structures.bbox import HorizontalBoxes, distance2bbox
from mmdet.structures.bbox.transforms import bbox_cxcywh_to_xyxy, scale_boxes
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig, reduce_mean)
from mmengine.config import ConfigDict
from mmengine.model import normal_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from ..utils import gt_instances_preprocess
from .rtmdet_head import RTMDetHead, RTMDetSepBNHeadModule
from mmyolo.models.utils.enn import N, ennRearrange

try:
    from mmrotate.structures.bbox import RotatedBoxes, distance2obb
    MMROTATE_AVAILABLE = True
except ImportError:
    RotatedBoxes = None
    distance2obb = None
    MMROTATE_AVAILABLE = False


@MODELS.register_module()
class MessDetRotatedSepBNHeadModule(RTMDetSepBNHeadModule):
    """Detection Head Module of RTMDet-R.

    Compared with RTMDet Detection Head Module, RTMDet-R adds
    a conv for angle prediction.
    An `angle_out_dim` arg is added, which is generated by the
    angle_coder module and controls the angle pred dim.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid.  Defaults to 1.
        feat_channels (int): Number of hidden channels. Used in child classes.
            Defaults to 256
        stacked_convs (int): Number of stacking convs of the head.
            Defaults to 2.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to (8, 16, 32).
        share_conv (bool): Whether to share conv layers between stages.
            Defaults to True.
        pred_kernel_size (int): Kernel size of ``nn.Conv2d``. Defaults to 1.
        angle_out_dim (int): Encoded length of angle, will passed by head.
            Defaults to 1.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to ``dict(type='BN')``.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Default: dict(type='SiLU', inplace=True).
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        widen_factor: float = 1.0,
        num_base_priors: int = 1,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        multibranch: bool = False,
        featmap_strides: Sequence[int] = [8, 16, 32],
        share_conv: bool = True,
        pred_kernel_size: int = 1,
        angle_out_dim: int = 1,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN'),
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        init_cfg: OptMultiConfig = None,
    ):
        self.multibranch = multibranch
        self.angle_out_dim = angle_out_dim
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            widen_factor=widen_factor,
            num_base_priors=num_base_priors,
            feat_channels=feat_channels,
            stacked_convs=stacked_convs,
            featmap_strides=featmap_strides,
            share_conv=share_conv,
            pred_kernel_size=pred_kernel_size,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def _init_layers(self):
        """Initialize layers of the head."""
        # super()._init_layers()


        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        for n in range(len(self.featmap_strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                if self.multibranch:
                    cls_convs.append(
                        ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1,
                            groups=N if i < self.stacked_convs - 1 else 1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
                    reg_convs.append(
                        ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1,
                            groups=N if i < self.stacked_convs - 1 else 1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
                else:
                    cls_convs.append(
                        ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
                    reg_convs.append(
                        ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.num_classes,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))

        if self.share_conv:
            for n in range(len(self.featmap_strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

        self.rtm_ang = nn.ModuleList()
        for _ in range(len(self.featmap_strides)):
            self.rtm_ang.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.angle_out_dim,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        # Use prior in model initialization to improve stability
        super().init_weights()
        for rtm_ang in self.rtm_ang:
            normal_init(rtm_ang, std=0.01)

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - angle_preds (list[Tensor]): Angle prediction for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * angle_out_dim.
        """

        cls_scores = []
        bbox_preds = []
        angle_preds = []
        for idx, x in enumerate(feats):
            x = ennRearrange(x)

            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            reg_dist = self.rtm_reg[idx](reg_feat)
            angle_pred = self.rtm_ang[idx](reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            angle_preds.append(angle_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds)


@MODELS.register_module()
class MessDetRotatedHead(RTMDetHead):
    """RTMDet-R head.

    Compared with RTMDetHead, RTMDetRotatedHead add some args to support
    rotated object detection.

    - `angle_version` used to limit angle_range during training.
    - `angle_coder` used to encode and decode angle, which is similar
      to bbox_coder.
    - `use_hbbox_loss` and `loss_angle` allow custom regression loss
      calculation for rotated box.

      There are three combination options for regression:

      1. `use_hbbox_loss=False` and loss_angle is None.

      .. code:: text

        bbox_pred────(tblr)───┐
                              ▼
        angle_pred          decode──►rbox_pred──(xywha)─►loss_bbox
            │                 ▲
            └────►decode──(a)─┘

      2. `use_hbbox_loss=False` and loss_angle is specified.
         A angle loss is added on angle_pred.

      .. code:: text

        bbox_pred────(tblr)───┐
                              ▼
        angle_pred          decode──►rbox_pred──(xywha)─►loss_bbox
            │                 ▲
            ├────►decode──(a)─┘
            │
            └───────────────────────────────────────────►loss_angle

      3. `use_hbbox_loss=True` and loss_angle is specified.
         In this case the loss_angle must be set.

      .. code:: text

        bbox_pred──(tblr)──►decode──►hbox_pred──(xyxy)──►loss_bbox

        angle_pred──────────────────────────────────────►loss_angle

    - There's a `decoded_with_angle` flag in test_cfg, which is similar
      to training process.

      When `decoded_with_angle=True`:

      .. code:: text

        bbox_pred────(tblr)───┐
                              ▼
        angle_pred          decode──(xywha)──►rbox_pred
            │                 ▲
            └────►decode──(a)─┘

      When `decoded_with_angle=False`:

      .. code:: text

        bbox_pred──(tblr)─►decode
                              │ (xyxy)
                              ▼
                           format───(xywh)──►concat──(xywha)──►rbox_pred
                                               ▲
        angle_pred────────►decode────(a)───────┘

    Args:
        head_module(ConfigType): Base module used for RTMDetRotatedHead.
        prior_generator: Points generator feature maps in
            2D points-based detectors.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        angle_version (str): Angle representations. Defaults to 'le90'.
        use_hbbox_loss (bool): If true, use horizontal bbox loss and
            loss_angle should not be None. Default to False.
        angle_coder (:obj:`ConfigDict` or dict): Config of angle coder.
        loss_angle (:obj:`ConfigDict` or dict, optional): Config of angle loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
            self,
            head_module: ConfigType,
            prior_generator: ConfigType = dict(
                type='mmdet.MlvlPointGenerator', strides=[8, 16, 32],
                offset=0),
            bbox_coder: ConfigType = dict(type='DistanceAnglePointCoder'),
            loss_cls: ConfigType = dict(
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0),
            loss_bbox: ConfigType = dict(
                type='mmrotate.RotatedIoULoss', mode='linear',
                loss_weight=2.0),
            angle_version: str = 'le90',
            use_hbbox_loss: bool = False,
            angle_coder: ConfigType = dict(type='mmrotate.PseudoAngleCoder'),
            loss_angle: OptConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            init_cfg: OptMultiConfig = None):
        if not MMROTATE_AVAILABLE:
            raise ImportError(
                'Please run "mim install -r requirements/mmrotate.txt" '
                'to install mmrotate first for rotated detection.')

        self.angle_version = angle_version
        self.use_hbbox_loss = use_hbbox_loss
        if self.use_hbbox_loss:
            assert loss_angle is not None, \
                ('When use hbbox loss, loss_angle needs to be specified')
        self.angle_coder = TASK_UTILS.build(angle_coder)
        self.angle_out_dim = self.angle_coder.encode_size
        if head_module.get('angle_out_dim') is not None:
            warnings.warn('angle_out_dim will be overridden by angle_coder '
                          'and does not need to be set manually')

        head_module['angle_out_dim'] = self.angle_out_dim
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

        if loss_angle is not None:
            self.loss_angle = MODELS.build(loss_angle)
        else:
            self.loss_angle = None

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into bbox
        results.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            angle_preds (list[Tensor]): Box angle for each scale level
                with shape (N, num_points * angle_dim, H, W)
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.
            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 5),
              the last dimension 4 arrange as (x, y, w, h, angle).
        """
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        # Whether to decode rbox with angle.
        # different setting lead to different final results.
        # Defaults to True.
        decode_with_angle = cfg.get('decode_with_angle', True)

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                   self.angle_out_dim)
            for angle_pred in angle_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_angle_preds = torch.cat(flatten_angle_preds, dim=1)
        flatten_angle_preds = self.angle_coder.decode(
            flatten_angle_preds, keepdim=True)

        if decode_with_angle:
            flatten_rbbox_preds = torch.cat(
                [flatten_bbox_preds, flatten_angle_preds], dim=-1)
            flatten_decoded_bboxes = self.bbox_coder.decode(
                flatten_priors[None], flatten_rbbox_preds, flatten_stride)
        else:
            flatten_decoded_hbboxes = self.bbox_coder.decode(
                flatten_priors[None], flatten_bbox_preds, flatten_stride)
            flatten_decoded_hbboxes = HorizontalBoxes.xyxy_to_cxcywh(
                flatten_decoded_hbboxes)
            flatten_decoded_bboxes = torch.cat(
                [flatten_decoded_hbboxes, flatten_angle_preds], dim=-1)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]

        results_list = []
        for (bboxes, scores, objectness,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas):
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = RotatedBoxes(bboxes)
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=RotatedBoxes(bboxes[keep_idxs]))

            if rescale:
                if pad_param is not None:
                    results.bboxes.translate_([-pad_param[2], -pad_param[0]])

                scale_factor = [1 / s for s in img_meta['scale_factor']]
                results.bboxes = scale_boxes(results.bboxes, scale_factor)

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta)

            results_list.append(results)
        return results_list

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            angle_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Decoded box for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [tl_x, tl_y, br_x, br_y] format.
            angle_preds (list[Tensor]): Angle prediction for each scale
                level with shape (N, num_anchors * angle_out_dim, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xywha
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        device = cls_scores[0].device

        # If the shape does not equal, generate new one
        if featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = featmap_sizes
            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                featmap_sizes, device=device, with_stride=True)
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)

        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1).contiguous()

        flatten_tblrs = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ], 1)
        flatten_tblrs = flatten_tblrs * self.flatten_priors_train[..., -1,
                                                                  None]
        flatten_angles = torch.cat([
            angle_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.angle_out_dim) for angle_pred in angle_preds
        ], 1)
        flatten_decoded_angle = self.angle_coder.decode(
            flatten_angles, keepdim=True)
        flatten_tblra = torch.cat([flatten_tblrs, flatten_decoded_angle],
                                  dim=-1)
        flatten_rbboxes = distance2obb(
            self.flatten_priors_train[..., :2],
            flatten_tblra,
            angle_version=self.angle_version)
        if self.use_hbbox_loss:
            flatten_hbboxes = distance2bbox(self.flatten_priors_train[..., :2],
                                            flatten_tblrs)

        assigned_result = self.assigner(flatten_rbboxes.detach(),
                                        flatten_cls_scores.detach(),
                                        self.flatten_priors_train, gt_labels,
                                        gt_bboxes, pad_bbox_flag)

        labels = assigned_result['assigned_labels'].reshape(-1)
        label_weights = assigned_result['assigned_labels_weights'].reshape(-1)
        bbox_targets = assigned_result['assigned_bboxes'].reshape(-1, 5)
        assign_metrics = assigned_result['assign_metrics'].reshape(-1)
        cls_preds = flatten_cls_scores.reshape(-1, self.num_classes)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        avg_factor = reduce_mean(assign_metrics.sum()).clamp_(min=1).item()

        loss_cls = self.loss_cls(
            cls_preds, (labels, assign_metrics),
            label_weights,
            avg_factor=avg_factor)

        pos_bbox_targets = bbox_targets[pos_inds]

        if self.use_hbbox_loss:
            bbox_preds = flatten_hbboxes.reshape(-1, 4)
            pos_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets[:, :4])
        else:
            bbox_preds = flatten_rbboxes.reshape(-1, 5)
        angle_preds = flatten_angles.reshape(-1, self.angle_out_dim)

        if len(pos_inds) > 0:
            loss_bbox = self.loss_bbox(
                bbox_preds[pos_inds],
                pos_bbox_targets,
                weight=assign_metrics[pos_inds],
                avg_factor=avg_factor)
            loss_angle = angle_preds.sum() * 0
            if self.loss_angle is not None:
                pos_angle_targets = bbox_targets[pos_inds][:, 4:5]
                pos_angle_targets = self.angle_coder.encode(pos_angle_targets)
                loss_angle = self.loss_angle(
                    angle_preds[pos_inds],
                    pos_angle_targets,
                    weight=assign_metrics[pos_inds],
                    avg_factor=avg_factor)
        else:
            loss_bbox = bbox_preds.sum() * 0
            loss_angle = angle_preds.sum() * 0

        losses = dict()
        losses['loss_cls'] = loss_cls
        losses['loss_bbox'] = loss_bbox
        if self.loss_angle is not None:
            losses['loss_angle'] = loss_angle

        return losses
