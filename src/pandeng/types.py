"""跨模块共享的数据契约。

对应方案第 3 节(快速跟踪)、第 4 节(低频校正)与第 4.2 节(异步调度)中
需要在快慢双环之间传递的字段, 尤其是校正结果必须携带的
"原图时间、帧号、目标代次和置信度"。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "BBox",
    "TrackState",
    "TargetMode",
    "ControlMode",
    "Detection",
    "Track",
    "TargetState",
    "CorrectionRequest",
    "CorrectionResult",
    "CorrectionStatus",
    "ControlCommand",
    "DeviceHealth",
]


BBox = np.ndarray  # float32 (4,) 格式为 xyxy, 像素坐标


class TrackState(str, Enum):
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


class TargetMode(str, Enum):
    """目标模式(方案第 3 节): 单鱼使用连续可见性与切换滞回; 鱼群跟随群体中心。"""

    SINGLE = "single"
    SCHOOL = "school"


class CorrectionStatus(str, Enum):
    """慢速环结果状态, 用于门控决策。"""

    OK = "ok"                     # 关联确认, 可用于校正
    DRIFT = "drift"               # 判定跟踪漂移
    REDETECTED = "redetected"     # 目标重现并重新确认
    TARGET_SWITCH = "target_switch"  # 关联到其他个体, 需要修正目标
    NO_MATCH = "no_match"         # 未找到一致候选
    STALE = "stale"               # 结果过期, 已丢弃
    TIMEOUT = "timeout"           # 推理超时
    OVERLOADED = "overloaded"     # 资源超限, 本次跳过
    DISABLED = "disabled"


class ControlMode(str, Enum):
    TRACK = "track"        # 正常跟随
    APPROACH = "approach"  # 允许低输出前进
    HOLD = "hold"          # 保持位置, 不追近
    SEARCH = "search"      # 限时转向搜索
    EXIT = "exit"          # 退出跟随, 执行底层退出策略


@dataclass
class Detection:
    """单帧检测框。"""

    xyxy: BBox
    score: float
    cls: int = 0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (float((x1 + x2) * 0.5), float((y1 + y2) * 0.5))


@dataclass
class Track:
    """ByteTrack 输出的短时轨迹。"""

    track_id: int
    xyxy: BBox
    score: float
    state: TrackState = TrackState.TRACKED
    age: int = 0                     # 轨迹存活帧数
    hits: int = 0                    # 命中的检测次数
    time_since_update: int = 0       # 距上次命中的帧数
    velocity: Optional[np.ndarray] = None  # (4,) 框速度, 像素/帧

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (float((x1 + x2) * 0.5), float((y1 + y2) * 0.5))

    def predicted_bbox(self, dt_frames: float = 1.0) -> BBox:
        """按框速度外推目标框, 供慢环做运动连续性打分与跨帧位置对齐。

        无速度信息时退化为返回当前框。
        """
        box = np.asarray(self.xyxy, dtype=np.float32)
        if self.velocity is None:
            return box.copy()
        return (box + np.asarray(self.velocity, dtype=np.float32) * float(dt_frames)).astype(
            np.float32
        )


@dataclass
class TargetState:
    """快环输出的目标状态, 是控制器的唯一输入。

    慢环不得绕过快速状态估计直接控制推进器, 只能通过
    `association_confidence` / `drift_flag` / `recovered` 等字段影响该状态。
    """

    frame_id: int
    timestamp: float
    bbox: BBox
    track_id: int
    visible: bool
    generation: int = 0            # 目标代次: 每次确认切换/重捕后 +1
    ex: float = 0.0                # 归一化水平偏差 [-1, 1]
    ey: float = 0.0                # 归一化垂直偏差 [-1, 1]
    area_ratio: float = 0.0        # 框面积 / 图像面积
    area_growth: float = 0.0       # 框面积增长率 (/s)
    score: float = 0.0             # 检测置信度
    association_confidence: float = 1.0   # 慢速环关联置信度
    drift_flag: bool = False       # 慢速环判定跟踪漂移
    recovered: bool = False        # 本帧由慢速环重捕恢复
    group_size: int = 1            # 鱼群模式下参与中心计算的框数量
    age_frames: int = 0            # 目标连续可见帧数

    def copy(self, **changes) -> "TargetState":
        return replace(self, **changes)

    @property
    def is_trustworthy(self) -> bool:
        """快速链路健康且目标可信。"""
        return self.visible and not self.drift_flag and self.association_confidence >= 0.5


@dataclass
class CorrectionRequest:
    """提交给慢速环的校正请求。"""

    frame_id: int
    timestamp: float
    image: np.ndarray                       # 原图引用(只读, 快速环不修改已提交帧)
    candidates: Sequence[Detection] = ()    # 有限数量候选区域
    target_bbox: Optional[BBox] = None      # 快速环当前目标框
    predicted_bbox: Optional[BBox] = None   # 快速环对"当前帧"的运动预测框
    target_track_id: int = -1
    generation: int = 0                     # 请求发起时的目标代次
    epoch: int = 0                          # 序列纪元: 换序列后旧结果作废
    trigger: str = "interval"               # interval | conf_drop | edge | cross | reid
    priority: int = 0

    def age_s(self) -> float:
        return max(0.0, time.monotonic() - self.timestamp)


@dataclass
class CorrectionResult:
    """慢速环校正结果。

    携带原图时间、帧号、目标代次和置信度, 由快速环做门控后再应用。
    """

    frame_id: int
    timestamp: float                   # 原图采集时间(monotonic)
    wall_time: float                   # 推理完成时间(monotonic)
    status: CorrectionStatus
    target_track_id: int = -1
    generation: int = 0
    confidence: float = 0.0            # 关联置信度
    matched_bbox: Optional[BBox] = None
    matched_detection_index: int = -1
    drift_score: float = 1.0           # 1 = 完全一致, 0 = 完全不一致
    latency_s: float = 0.0
    message: str = ""
    scores: dict[str, float] = field(default_factory=dict)

    def age_s(self, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else now
        return max(0.0, now - self.timestamp)

    def is_stale(self, max_age_s: float, now: Optional[float] = None) -> bool:
        return self.age_s(now) > max_age_s


@dataclass
class ControlCommand:
    """受限控制建议, 由快速环发布, 经仲裁后交由飞控执行。"""

    timestamp: float
    frame_id: int
    mode: ControlMode
    surge: float = 0.0   # 前进
    yaw: float = 0.0     # 转向
    heave: float = 0.0   # 深度
    reason: str = ""
    valid: bool = True

    def clamped(self) -> "ControlCommand":
        return replace(
            self,
            surge=float(np.clip(self.surge, -1.0, 1.0)),
            yaw=float(np.clip(self.yaw, -1.0, 1.0)),
            heave=float(np.clip(self.heave, -1.0, 1.0)),
        )


@dataclass
class DeviceHealth:
    """感知与设备健康状态, 用于"感知过期或设备异常"的退出判据。"""

    perception_ok: bool = True
    camera_ok: bool = True
    last_perception_time: float = 0.0
    slow_link_ok: bool = True
    note: str = ""
