# coding=utf-8
# ================================================================
#
#   File name   : yolov5_deep_sort_tracker.py
#   Author      : Faye
#   E-mail      : xiansheng14@sina.com
#   Created date: 2022/10/19 16:18 
#   Description : Yolov5s+deepsort
#
# ================================================================
from dcmtracking.deep_sort.tracker.base_tracker import BaseTracker
from dcmtracking.detection.yolov5.yolo import YOLO
from PIL import Image
import cv2
import torch
from dcmtracking.deep_sort.deep.feature_extractor import Extractor


class Yolov5DeepSortTracker(BaseTracker):
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
        im_h, im_w, _ = im.shape
        im_pil = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(im_pil)
        pred_boxes = []
        top_label, top_boxes, top_conf = self.yolo.detect_image(im_pil)
        if top_label is not None:
            for (y1, x1, y2, x2), lbl, conf in zip(top_boxes, top_label, top_conf):
                if lbl != 0:
                    continue
                pred_boxes.append(
                    (int(x1), int(y1), int(x2), int(y2), lbl, conf))
        return im, pred_boxes
