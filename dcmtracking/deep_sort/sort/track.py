# vim: expandtab:ts=4:sw=4


class TrackState:
    """
    单个目标跟踪状态的枚举类型。
    在收集到足够的证据之前，新创建的足迹被归类为Tentative（unconfirmed）。
    已经和历史track匹配上的，状态变为`confirmed`。
    已删除的的track被分类为`deleted`。

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    带有状态空间(x, y, a, h)和相关速度的单一目标轨迹，其中(x, y)是边界框的中心，a是纵横比，h是高度。

    Parameters
    ----------
    mean : ndarray
        初始的平均向量,即8个值的期望。
    covariance : ndarray
        初始的协方差矩阵。
    track_id : int
        每个track的唯一标识
    n_init : int
        在track被确认之前的连续探测次数。如果在第一个n_init帧内发生miss，则track状态被设置为' Deleted '。
    max_age : int
        在将track状态设置为Deleted之前，连续miss的最大次数。
    feature : Optional[ndarray]
        该track的特征向量。如果不是None，这个特性会被添加到' features '缓存中。这里特征向量默认长度512

    Attributes
    ----------
    mean : ndarray
        初始的平均向量,即8个值的期望。
    covariance : ndarray
        初始的协方差矩阵。
    track_id : int
        每个track的唯一标识
    hits : int
        测量更新的总数。
    age : int
        自第一次发生以来的总帧数。
    time_since_update : int
        自上次测量更新以来的总帧数。
    state : TrackState
        当前Track的状态
    features : List[ndarray]
        特性向量的缓存。在每次测量更新时，关联的特征向量被添加到这个列表中。

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """获取当前位置的边框，格式(左上角x，左上角y，宽度，高度)。

        Returns
        -------
        ndarray
            ret: The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """获取当前位置的边框，格式'(左上x，左上x，右下x，右下y) '

        Returns
        -------
        ndarray
            ret: The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """执行卡尔曼滤波的predict操作

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            卡尔曼滤波类的一个实例.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """执行卡尔曼滤波测量update步骤.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
        detection : Detection

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """将此track标记为miss(当前时间步骤没有关联)。
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """如果此track是tentativ(unconfirmed)则返回True。"""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """如果此track是确定的，则返回True。"""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """如果此track是删除的，则返回True。"""
        return self.state == TrackState.Deleted
