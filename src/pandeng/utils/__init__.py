"""通用工具: 几何、数值、Kalman 与日志。"""

from __future__ import annotations

from .geometry import (
    area_ratio,
    box_iou,
    clamp_limits,
    clip_boxes,
    deadzone,
    expand_box,
    iou_matrix,
    normalize_error,
    slew_limit,
)
from .logging_utils import setup_logging
from .kalman import KalmanFilterXYAH

__all__ = [
    "KalmanFilterXYAH",
    "area_ratio",
    "box_iou",
    "clamp_limits",
    "clip_boxes",
    "deadzone",
    "expand_box",
    "iou_matrix",
    "normalize_error",
    "setup_logging",
    "slew_limit",
]
