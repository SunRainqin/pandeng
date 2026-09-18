"""卡尔曼滤波: 8 维状态 (cx, cy, a, h, vcx, vcy, va, vh) 的匀速模型。

ByteTrack 的短时轨迹预测依赖该滤波器; 这里保持与主流 ByteTrack 实现一致的
参数化方式, 便于后续替换为其他滤波后端。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["KalmanFilterXYAH"]

# 状态转移矩阵: 位置由速度积分得到
_F = np.eye(8, dtype=np.float64)
for _i in range(4):
    _F[_i, _i + 4] = 1.0

# 观测矩阵: 只观测位置分量
_H = np.eye(4, 8, dtype=np.float64)


class KalmanFilterXYAH:
    """常量速度卡尔曼滤波器, 观测为 (center_x, center_y, aspect_ratio, height)。"""

    def __init__(
        self,
        std_weight_position: float = 0.05,
        std_weight_velocity: float = 0.00625,
    ) -> None:
        self.std_weight_position = float(std_weight_position)
        self.std_weight_velocity = float(std_weight_velocity)

    # -- 初始化 -------------------------------------------------------------
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = np.asarray(measurement, dtype=np.float64).reshape(4)
        std = self._std(mean, velocity_scale=10.0)
        covariance = np.diag(np.square(std)).astype(np.float64)
        return mean, covariance

    # -- 预测 ---------------------------------------------------------------
    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        std = self._std(mean, velocity_scale=1.0)
        motion_cov = np.diag(np.square(std)).astype(np.float64)
        mean = _F @ mean
        covariance = _F @ covariance @ _F.T + motion_cov
        return mean, covariance

    # -- 观测投影 -----------------------------------------------------------
    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        std = self._std(mean, velocity_scale=0.1, position_only=True)
        innovation_cov = np.diag(np.square(std)).astype(np.float64)
        mean = _H @ mean
        covariance = _H @ covariance @ _H.T + innovation_cov
        return mean, covariance

    # -- 更新 ---------------------------------------------------------------
    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        # K = C H^T S^{-1}; 由于 C 对称, 等价于 K = (S^{-T} (H C)^T)^T
        # 其中 S = projected_cov (4x4), H C 形状 (4, 8), 解得 K 形状 (8, 4)
        kalman_gain = np.linalg.solve(projected_cov.T, _H @ covariance).T
        innovation = np.asarray(measurement, dtype=np.float64).reshape(4) - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    # -- 内部 ---------------------------------------------------------------
    def _std(
        self,
        mean: np.ndarray,
        velocity_scale: float,
        position_only: bool = False,
    ) -> np.ndarray:
        h = max(1e-3, abs(float(mean[3])))
        wp = self.std_weight_position * h
        wv = self.std_weight_velocity * h
        if position_only:
            return np.array([wp, wp, 1e-2, wp], dtype=np.float64)
        return np.array(
            [
                wp,
                wp,
                1e-2,
                wp,
                wv * velocity_scale,
                wv * velocity_scale,
                1e-5,
                wv * velocity_scale,
            ],
            dtype=np.float64,
        )
