# Copyright (c) OpenMMLab. All rights reserved.
from .pose_coco import PoseCocoDataset
from .transforms import *  # noqa: F401,F403
from .utils import BatchShapePolicy, yolov5_collate
from .yolov5_coco import YOLOv5CocoDataset
from .yolov5_crowdhuman import YOLOv5CrowdHumanDataset
from .yolov5_dota import YOLOv5DOTADataset
from .yolov5_voc import YOLOv5VOCDataset
from .yolov5_dota15 import YOLOv5DOTA15Dataset
from .yolov5_hrsc import YOLOv5HRSCDataset
from .yolov5_dior import YOLOv5DIORDataset

__all__ = [
    'YOLOv5CocoDataset', 'YOLOv5VOCDataset', 'BatchShapePolicy',
    'yolov5_collate', 'YOLOv5CrowdHumanDataset', 'YOLOv5DOTADataset',
    'PoseCocoDataset',
    'YOLOv5DOTA15Dataset',
    'YOLOv5HRSCDataset',
    'YOLOv5DIORDataset'
]
