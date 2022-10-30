# coding=utf-8
# ================================================================
#
#   File name   : yolo_fastestv2_deep_sort_tracker.py
#   Author      : Faye
#   E-mail      : xiansheng14@sina.com
#   Created date: 2022/10/19 16:18 
#   Description : YoloFastestV2+deepsort
#
# ================================================================
from dcmtracking.deep_sort.tracker.base_tracker import BaseTracker
from dcmtracking.detection.yolo_fastestv2.yolo_fastestv2 import YOLO
from dcmtracking.deep_sort.deep.feature_extractor import Extractor
import torch


class YoloFastestV2DeepSortTracker(BaseTracker):

    def __init__(self, need_speed=False, need_angle=False):
        # 执行父类的init方法
        BaseTracker.__init__(self)
        # 初始化目标检测类
        self.yolo = YOLO()

    def init_extractor(self):
        """
        实现父类的init_extractor方法，初始化特征提取器
        Parameters
        ----------
        im

        Returns
        -------

        """
        model_path = "dcmtracking/deep_sort/deep/checkpoint/ckpt.t7"
        return Extractor(model_path, use_cuda=torch.cuda.is_available())

    def detect(self, im):
        """
        实现父类的detect方法
        Parameters
        ----------
        im

        Returns
        -------

        """
        pred_boxes = self.yolo.detect_image(im)
        results = []
        for pred_box in pred_boxes:
            lbl = pred_box[4]
            # print('lbl:', pred_box[5])
            if lbl == 0:
                results.append(pred_box)
        return im, results
