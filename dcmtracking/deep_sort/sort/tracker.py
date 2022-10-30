# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    DeepSORT的跟踪部分，即除了Detection以外的部分

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        用于测量到track关联的距离度量工具。
    max_iou_distance : float
        tracker执行IOU匹配时，大于此值的关联被忽略。
    max_age : int
        在删除track之前的最大miss数。
    n_init : int
        在一个track被确认之前的连续探测次数。如果在第一个n_init帧内发生miss，则track状态被设置为' Deleted '。

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        用于测量到track关联的距离度量工具。
    max_age : int
        在删除track之前的最大miss数。
    n_init : int
        在一个track被确认之前的连续探测次数。如果在第一个n_init帧内发生miss，则track状态被设置为' Deleted '。
    kf : kalman_filter.KalmanFilter
        卡尔曼滤波器实例
    tracks : List[Track]
        当前时间步的track列表。

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """
        用上一轮次的结果 计算本轮的预测值
        这个函数应该在每个时间步中，在' update '之前调用一次。
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """执行测量数据更新和跟踪管理。

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            当前时间步上的Detection List.

        """
        # 得到匹配的、未匹配的tracks、未匹配的dectections
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # 对于每个匹配成功的track，用其对应的detection进行更新
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        # 对于未匹配的成功的track，将其标记为丢失
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        # 对于未匹配成功的detection，初始化为新的track
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # 遍历所有tracks，将已经确定的track的特征向量存入metric
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            """
            基于外观信息和马氏距离，计算卡尔曼滤波预测的tracks和当前时刻检测到的detections的代价矩阵
            Parameters
            ----------
            tracks
            dets
            track_indices
            detection_indices

            Returns
            -------
            cost_matrix 代价矩阵

            """
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            # 基于外观的特征向量，计算tracks和detections的余弦距离代价矩阵
            cost_matrix = self.metric.distance(features, targets)
            # 基于马氏距离，过滤掉代价矩阵中一些不合适的项 (将其设置为一个较大的值)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # 将已经存在的tracks分成已确定和未确定，感觉这里可以优化
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # 对confirmd tracks进行级联匹配
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # 对级联匹配中未匹配的tracks和unconfirmed tracks中time_since_update为1的tracks进行IOU匹配
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        # 整合所有的匹配对和未匹配的tracks
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        """
        利用detection生成一个track，并将该track放入self.tracks
        Parameters
        ----------
        detection

        Returns
        -------

        """
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
