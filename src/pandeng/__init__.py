"""pandeng: 基于视觉基座与时序预测的快慢双环跟踪框架。

第一版交付范围(方案第 6 节):
    检测(YOLO11n) + 跟踪(ByteTrack) + DINOv3 低频校正 + 传统图像伺服控制。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
