"""几何与数值工具。

只实现框架必需的、经过单测覆盖的小工具, 避免引入不必要的依赖。
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "xyxy_to_xywh",
    "xywh_to_xyxy",
    "clip_boxes",
    "box_iou",
    "iou_matrix",
    "normalize_error",
    "area_ratio",
    "centers",
    "expand_box",
    "ema",
    "slope_per_second",
    "deadzone",
    "slew_limit",
    "clamp_limits",
]


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    x, y, w, h = np.asarray(box, dtype=np.float32)
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """把框裁剪到图像范围内, 并保证 x1<x2, y1<y2。"""
    arr = np.atleast_2d(np.asarray(boxes, dtype=np.float32)).copy()
    if arr.size == 0:
        return arr.reshape(-1, 4)
    arr[:, 0::2] = np.clip(arr[:, 0::2], 0, width)
    arr[:, 1::2] = np.clip(arr[:, 1::2], 0, height)
    x1 = np.minimum(arr[:, 0], arr[:, 2])
    y1 = np.minimum(arr[:, 1], arr[:, 3])
    x2 = np.maximum(arr[:, 0], arr[:, 2])
    y2 = np.maximum(arr[:, 1], arr[:, 3])
    out = np.stack([x1, y1, x2, y2], axis=1)
    return out.astype(np.float32)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """单对框 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """向量化 IoU 矩阵, 形状 (len(a), len(b))。

    使用 [x1, y1, x2, y2] 表示, 兼容空输入。
    """
    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
    inter = inter_w * inter_h

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def normalize_error(
    center: Tuple[float, float], width: int, height: int
) -> Tuple[float, float]:
    """方案第 3 节: ex = (u - W/2)/(W/2), ey = (v - H/2)/(H/2)。"""
    u, v = center
    ex = (float(u) - width * 0.5) / max(1e-6, width * 0.5)
    ey = (float(v) - height * 0.5) / max(1e-6, height * 0.5)
    return float(np.clip(ex, -1.0, 1.0)), float(np.clip(ey, -1.0, 1.0))


def area_ratio(box: np.ndarray, width: int, height: int) -> float:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(area / max(1.0, float(width) * float(height)))


def centers(boxes: Iterable[np.ndarray]) -> np.ndarray:
    arr = np.asarray(list(boxes), dtype=np.float32).reshape(-1, 4)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([(arr[:, 0] + arr[:, 2]) * 0.5, (arr[:, 1] + arr[:, 3]) * 0.5], axis=1)


def expand_box(
    box: np.ndarray, factor: float, width: Optional[int] = None, height: Optional[int] = None
) -> np.ndarray:
    """以中心为基准外扩目标框, 用于提取鱼体区域特征。"""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half_w = max(1.0, (x2 - x1) * 0.5 * max(1.0, factor))
    half_h = max(1.0, (y2 - y1) * 0.5 * max(1.0, factor))
    out = np.array([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dtype=np.float32)
    if width is not None and height is not None:
        out = clip_boxes(out, width, height)[0]
    return out


def ema(previous: Optional[float], value: float, momentum: float) -> float:
    """指数滑动平均, momentum 为更新幅度(0~1)。"""
    if previous is None:
        return float(value)
    m = float(np.clip(momentum, 0.0, 1.0))
    return float((1.0 - m) * previous + m * value)


def slope_per_second(
    history: Sequence[Tuple[float, float]], value: float, now: float, window_s: float
) -> float:
    """基于时间窗口的线性斜率, 用于计算框面积增长率(/s)。

    `history` 为 [(t, value), ...] 的升序序列, 函数会剔除窗口外的旧样本。
    """
    trimmed = [(t, v) for t, v in history if now - t <= window_s]
    trimmed.append((now, float(value)))
    if len(trimmed) < 2:
        return 0.0
    ts = np.array([t for t, _ in trimmed], dtype=np.float64)
    vs = np.array([v for _, v in trimmed], dtype=np.float64)
    ts = ts - ts[0]
    if ts[-1] <= 1e-6:
        return 0.0
    # 最小二乘斜率
    denom = float(np.sum((ts - ts.mean()) ** 2))
    if denom <= 1e-9:
        return 0.0
    slope = float(np.sum((ts - ts.mean()) * (vs - vs.mean())) / denom)
    return slope


def deadzone(value: float, threshold: float) -> float:
    """死区内输出 0, 死区外去掉死区偏置, 保证控制量连续。"""
    v = float(value)
    t = abs(float(threshold))
    if t <= 0.0 or abs(v) <= t:
        return 0.0
    return float(np.sign(v) * (abs(v) - t) / max(1e-6, 1.0 - t))


def slew_limit(target: float, previous: float, max_delta: float) -> float:
    """变化率限制。"""
    delta = float(target) - float(previous)
    if abs(delta) <= max_delta:
        return float(target)
    return float(previous) + float(np.sign(delta) * max_delta)


def clamp_limits(value: float, limits: Sequence[float]) -> float:
    lo, hi = float(limits[0]), float(limits[1])
    if lo > hi:
        lo, hi = hi, lo
    return float(np.clip(value, lo, hi))
