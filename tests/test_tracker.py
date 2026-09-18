"""单元测试: Kalman 滤波器与 ByteTrack。

重点覆盖曾经出错的地方: 卡尔曼更新中的矩阵维度(增益形状必须是 (8, 4))。
"""

from __future__ import annotations

import numpy as np

from pandeng.perception.tracker import ByteTracker
from pandeng.types import Detection, Track, TrackState
from pandeng.utils.kalman import KalmanFilterXYAH


def test_track_predicted_bbox_extrapolates():
    """Track.predicted_bbox 供慢环做运动连续性打分, 必须真的外推。

    该通路曾断过: 调度器里 predicted_bbox 硬编码为 None, 使关联打分中的
    运动分量恒为常量 0.5, 配置公式上的三证据融合只剩两条。
    """
    box = np.array([100.0, 100.0, 200.0, 160.0], dtype=np.float32)
    velocity = np.array([4.0, 2.0, 4.0, 2.0], dtype=np.float32)
    track = Track(track_id=1, xyxy=box, score=0.9, velocity=velocity)

    predicted = track.predicted_bbox(1.0)
    assert np.allclose(predicted, box + velocity)
    # 无速度信息时退化为当前框, 不应抛错
    degenerate = Track(track_id=2, xyxy=box, score=0.9).predicted_bbox()
    assert np.allclose(degenerate, box)


def test_tracker_exposes_box_velocity():
    """ tracker 输出的轨迹必须带速度, 否则运动分量无从计算。"""
    from pandeng.config import load_config

    tracker = ByteTracker(load_config().tracker, (1280, 720))
    for i in range(10):
        x = 300 + i * 15
        tracks = tracker.update(
            [Detection(xyxy=np.array([x, 300, x + 120, 380], dtype=np.float32), score=0.9)],
            frame_id=i + 1,
        )
    assert tracks
    assert tracks[0].state is TrackState.TRACKED
    assert tracks[0].velocity is not None


def test_kalman_initiate_shapes():
    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 40.0]))
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)


def test_kalman_update_shapes_and_convergence():
    """更新后的均值必须仍为 (8,), 且应被拉向观测值。"""
    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 40.0]))
    mean, cov = kf.predict(mean, cov)
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)

    measurement = np.array([110.0, 205.0, 0.5, 42.0])
    new_mean, new_cov = kf.update(mean, cov, measurement)
    assert new_mean.shape == (8,)
    assert new_cov.shape == (8, 8)

    # 位置分量应向观测值移动
    before = np.abs(mean[:2] - measurement[:2]).sum()
    after = np.abs(new_mean[:2] - measurement[:2]).sum()
    assert after < before
    # 协方差仍应对称
    assert np.allclose(new_cov, new_cov.T, atol=1e-8)


def test_kalman_gain_shape_is_eight_by_four():
    """回归测试: 增益形状错误曾导致 np.linalg.solve 维度不匹配。

    正确的增益为 K = C H^T S^{-1}, 形状 (8, 4); 由于 C 对称, 可写成
    K = (S^{-T} (H C)^T)^T, 即先解 (4,4) x (4,8) 再转置。
    """
    from pandeng.utils.kalman import _H

    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(np.array([50.0, 60.0, 1.0, 30.0]))
    mean, cov = kf.predict(mean, cov)
    _, projected_cov = kf.project(mean, cov)

    assert projected_cov.shape == (4, 4)
    assert (_H @ cov).shape == (4, 8)

    gain = np.linalg.solve(projected_cov.T, _H @ cov).T
    assert gain.shape == (8, 4)

    # 与直接按定义计算的结果一致
    reference = cov @ _H.T @ np.linalg.inv(projected_cov)
    assert np.allclose(gain, reference, atol=1e-8)


def _tracker_cfg():
    from pandeng.config import load_config

    return load_config().tracker


def test_tracker_tracks_moving_box():
    cfg = _tracker_cfg()
    tracker = ByteTracker(cfg, (1280, 720))

    boxes = []
    for i in range(12):
        x = 300 + i * 12
        boxes.append(
            Detection(xyxy=np.array([x, 300, x + 120, 380], dtype=np.float32), score=0.9)
        )
    ids = set()
    for i, det in enumerate(boxes):
        tracks = tracker.update([det], frame_id=i + 1)
        assert tracks, f"第 {i} 帧应产生轨迹"
        ids.add(tracks[0].track_id)
    assert len(ids) == 1, "连续运动的同一目标不应换 ID"


def test_tracker_keeps_id_through_short_occlusion():
    """ByteTrack 的低分第二轮关联应能扛住短暂遮挡。"""
    cfg = _tracker_cfg()
    tracker = ByteTracker(cfg, (1280, 720))
    box = np.array([300, 300, 420, 380], dtype=np.float32)

    for i in range(5):
        tracker.update([Detection(xyxy=box, score=0.9)], frame_id=i + 1)

    # 遮挡: 只有低分检测
    for i in range(5, 8):
        tracker.update([Detection(xyxy=box, score=0.15)], frame_id=i + 1)

    tracks = tracker.update([Detection(xyxy=box, score=0.9)], frame_id=9)
    assert tracks, "遮挡恢复后应仍有轨迹"
    assert tracks[0].track_id == 1


def test_tracker_drops_long_lost_track():
    cfg = _tracker_cfg()
    tracker = ByteTracker(cfg, (1280, 720))
    box = np.array([300, 300, 420, 380], dtype=np.float32)
    tracker.update([Detection(xyxy=box, score=0.9)], frame_id=1)

    for i in range(1, cfg.track_buffer + 5):
        tracker.update([], frame_id=i + 1)

    assert tracker.tracks == []
    assert tracker.lost_stracks == []
