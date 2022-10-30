import numpy as np
import torch

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker
import time

__all__ = ['DeepSort']


class DeepSort(object):
    """
    实现了DeepSort
    """
    def __init__(self, extractor, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, use_cuda=True):
        """

        Parameters
        ----------
        extractor:提取特征的模型实例
        max_dist:匹配的阈值。距离较大的样本被认为是无效匹配。
        min_confidence:最小置信度，小于这个值，认为是无效物体
        nms_max_overlap:执行nms时，最大重叠占比，两个bbox的iou大于这个值，将认为是同一物体
        max_iou_distance:tracker执行IOU匹配时，大于此值的关联被忽略。
        max_age:在删除track之前的最大miss数。
        n_init:在一个track被确认之前的连续探测次数。如果在第一个n_init帧内发生miss，则track状态被设置为' Deleted '。
        use_cuda:是否使用cuda
        """
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        # 生成特征向量的方法，可以根据情况替换
        self.extractor = extractor

        max_cosine_distance = max_dist
        # 每个track保留多少历史特征向量，超过这个数，旧的将被淘汰
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img):
        """
        根据目标检测的结果，执行跟踪、更新DeepSort历史状态
        Parameters
        ----------
        extractor:图像特征提取器
        bbox_xywh:目标框的中心点和宽高
        confidences：置信度
        ori_img：图片

        Returns
        -------

        """
        self.height, self.width = ori_img.shape[:2]
        # 将图片按照bbox切割 每块生成特征向量（特征向量默认长度512）
        features = self._get_features(bbox_xywh, ori_img)
        # 将左上右下的四个坐标 转换成中心点和宽高
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        # 根据features和bbox_tlwh生成detections 每个detection有features/tlwh/confidence 三个属性
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf > self.min_confidence]

        # 执行nms 去掉重复的detection 其实在目标检测阶段已经做了nms 这里不做也行
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # 执行卡尔曼滤波的predict操作 即使用上一轮次的结果 计算本轮的预测值
        self.tracker.predict()
        # 执行卡尔曼滤波的update操作
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs


    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        """
        将左上右下的四个坐标 转换成中心点和宽高
        Parameters
        ----------
        bbox_xywh

        Returns
        -------

        """
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        """
        将中心点和宽高 转换成左上右下的四个坐标
        Parameters
        ----------
        bbox_xywh

        Returns
        -------

        """
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        将左上xy和宽高 转换成左上右下的四个坐标
        Parameters
        ----------
        bbox_tlwh

        Returns
        -------

        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        """
        将左上右下的四个坐标 转换成左上xy和宽高
        Parameters
        ----------
        bbox_tlwh

        Returns
        -------

        """
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


