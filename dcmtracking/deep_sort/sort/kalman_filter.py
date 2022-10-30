# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
N个自由度卡方分布的0.95分位数表(包含N=1，…， 9).来自MATLAB/Octave的chi2inv函数，用作马氏距离门限值。
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    一种用于跟踪图像空间中边界框的简单卡尔曼滤波器。

    8维状态空间

        x, y, a, h, vx, vy, va, vh

    包含边界框中心位置(x, y), 高宽比 a, 高 h,
    x, vy, va, vh 为它们各自的速度.

    物体运动遵循匀速模型。将边界盒位置(x, y, a, h)作为状态空间的直接观测(线性观测模型)。

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # 构建8x8的卡尔曼滤波矩阵.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # 设置相对于当前状态估计的运动和观测不确定度，即以h为参照物的噪声。
        # 这些权重控制着模型中的不确定性。
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """创建新的track

        Parameters
        ----------
        measurement : ndarray
            观测边界框坐标(x, y, a, h)，中心位置(x, y)，纵横比a，高度h。

        Returns
        -------
        (ndarray, ndarray)
            返回新轨迹的平均向量(8维)和协方差矩阵(8x8维)。未观测到的速度初始化为平均值0。

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """执行卡尔曼滤波的predict步骤.

        Parameters
        ----------
        mean : ndarray
            前一个轮次的物体状态的8维向量的期望（均值）。
        covariance : ndarray
            前一个轮次的物体状态的8x8维协方差矩阵

        Returns
        -------
        (ndarray, ndarray)
            返回预测状态的平均向量和协方差矩阵。未观测到的速度初始化为平均值0。

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """
        将状态分布转换到测量空间，为计算卡尔曼增益和最优估计做准备。

        Parameters
        ----------
        mean : ndarray
            状态的平均向量(8维数组)。
        covariance : ndarray
            状态的协方差矩阵(8x8维)。

        Returns
        -------
        (ndarray, ndarray)
            返回给定状态估计的投影平均值和协方差矩阵。

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """卡尔曼滤波的update步骤，对观测值进行校正。

        Parameters
        ----------
        mean : ndarray
            预测状态的平均向量(8维)。
        covariance : ndarray
            状态的协方差矩阵(8x8维)。
        measurement : ndarray
            4维测量向量(x, y, a, h)，其中(x, y)是中心位置，a是纵横比，h是包围框的高度。

        Returns
        -------
        (ndarray, ndarray)
            返回经过测量校正的状态分布。

        """
        # 1.计算卡尔曼增益K
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean
        # 2.计算当前步最优估计
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # 3.更新过程噪声协方差矩阵
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """计算状态分布和测量值之间的门控距离（马氏距离）。

        可以从' chi2inv95 '中获得一个合适的距离阈值。
        如果' only_position '为False，则卡方分布有4个自由度，否则为2。

        Parameters
        ----------
        mean : ndarray
            状态分布的平均向量，即期望(8维)。
        covariance : ndarray
            状态分布的协方差(8x8维)。
        measurements : ndarray
            一个包含N个度量值的Nx4维矩阵，每个度量值的格式为(x, y, a, h)，其中(x, y)是包围框中心位置，a是纵横比，h是高度。
        only_position : Optional[bool]
            如果为True，则只对边界框中心位置进行距离计算。

        Returns
        -------
        ndarray
            返回一个长度为N的数组，其中第i个元素包含(均值，协方差)和“measurements[i]”之间的马氏距离的平方。

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
