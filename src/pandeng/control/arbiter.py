"""控制仲裁与安全退出。

对应方案第 3 节末行"感知过期或设备异常 -> 撤销跟随, 执行底层退出策略";
对应方案第 4.2 节末段"慢链路超时或资源超限时暂停校正; 快速链路健康且目标可信时
继续基线跟踪, 否则退出追近。慢链路不能绕过快速状态估计直接控制推进器。"

本模块是 ROS 2 仲裁节点的占位实现: 结构上它只接受 `ControlCommand`
(快速环产出)与健康标志, 不接受任何来自慢速环的推力请求, 从而在代码层面
保证"慢链路不能绕过快速状态估计直接控制推进器"。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..types import ControlCommand, ControlMode, DeviceHealth, TargetState
from ..utils.geometry import clamp_limits

__all__ = ["LinkHealth", "ArbitrationResult", "ControlArbiter"]

LOGGER = logging.getLogger(__name__)


class LinkHealth(str, Enum):
    """链路健康状态。"""

    OK = "ok"
    DEGRADED = "degraded"   # 慢链路降级, 快速环仍可信
    FAULT = "fault"        # 必须退出


@dataclass
class ArbitrationResult:
    command: ControlCommand
    health: LinkHealth
    reason: str
    allowed_to_approach: bool

    @property
    def is_exit(self) -> bool:
        return self.command.mode is ControlMode.EXIT


class ControlArbiter:
    """最终控制仲裁。"""

    def __init__(self, cfg) -> None:
        safety = cfg.get_path("safety", {}) if hasattr(cfg, "get_path") else {}
        self.perception_timeout_s = float(
            safety.get("perception_timeout_s", 0.5) if hasattr(safety, "get") else 0.5
        )
        self.on_perception_stale = str(
            safety.get("on_perception_stale", "exit") if hasattr(safety, "get") else "exit"
        )
        self.on_device_fault = str(
            safety.get("on_device_fault", "exit") if hasattr(safety, "get") else "exit"
        )
        self.limits = {
            "surge": self._limits(cfg.get_path("limits.surge", [-0.2, 0.5])),
            "yaw": self._limits(cfg.get_path("limits.yaw", [-0.6, 0.6])),
            "heave": self._limits(cfg.get_path("limits.heave", [-0.3, 0.3])),
        }
        self.health = DeviceHealth()
        self.last_result: Optional[ArbitrationResult] = None

    def reset(self) -> None:
        """切换到新序列: 清空健康状态与上一帧结论。

        保留 `last_perception_time` 会让新序列开头误判为感知陈旧。
        """
        self.health = DeviceHealth()
        self.last_result = None

    @staticmethod
    def _limits(value) -> tuple[float, float]:
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            seq = list(value)
            return (float(seq[0]), float(seq[1]))
        return (-1.0, 1.0)

    # -- 健康上报 -----------------------------------------------------------
    def report_perception(self, now: Optional[float] = None) -> None:
        self.health.last_perception_time = time.monotonic() if now is None else float(now)
        self.health.perception_ok = True

    def report_device(self, *, camera_ok: bool = True, note: str = "") -> None:
        self.health.camera_ok = bool(camera_ok)
        self.health.note = note

    def report_slow_link(self, *, ok: bool) -> None:
        self.health.slow_link_ok = bool(ok)

    @property
    def perception_age(self) -> float:
        if self.health.last_perception_time <= 0.0:
            return float("inf")
        return max(0.0, time.monotonic() - self.health.last_perception_time)

    # -- 仲裁 ---------------------------------------------------------------
    def evaluate_health(self, state: Optional[TargetState]) -> tuple[LinkHealth, str]:
        if not self.health.camera_ok:
            return LinkHealth.FAULT, f"camera fault ({self.health.note})"

        age = self.perception_age
        if age > self.perception_timeout_s:
            return LinkHealth.FAULT, f"perception stale ({age:.2f}s)"

        if not self.health.slow_link_ok:
            # 慢链路异常时: 快速链路健康且目标可信 -> 继续基线跟踪; 否则退出追近
            if state is not None and state.is_trustworthy:
                return LinkHealth.DEGRADED, "slow link degraded, baseline tracking"
            return LinkHealth.FAULT, "slow link degraded and target untrustworthy"

        if state is not None and state.drift_flag and state.association_confidence < 0.3:
            return LinkHealth.DEGRADED, "drift with low association confidence"

        return LinkHealth.OK, "nominal"

    def arbitrate(
        self,
        command: ControlCommand,
        state: Optional[TargetState],
    ) -> ArbitrationResult:
        """对快速环命令做最终仲裁, 输出给飞控。"""
        health, reason = self.evaluate_health(state)
        previous_health = self.last_result.health if self.last_result is not None else None

        if health is LinkHealth.FAULT:
            mode = ControlMode.EXIT if (
                self.on_perception_stale == "exit" or "camera" in reason
            ) else ControlMode.HOLD
            final = ControlCommand(
                timestamp=command.timestamp,
                frame_id=command.frame_id,
                mode=mode,
                surge=0.0,
                yaw=0.0,
                heave=0.0,
                reason=f"arbiter_override: {reason}",
                valid=False,
            )
            result = ArbitrationResult(final, health, reason, allowed_to_approach=False)
            self.last_result = result
            if previous_health is not LinkHealth.FAULT:
                LOGGER.warning("仲裁触发退出策略: %s", reason)
            return result

        # 漂移 / 低关联置信度: 允许转向维持视角, 但禁止追近。
        # 慢链路降级(DEGRADED)本身**不**禁止追近 —— 方案第 4.2 节要求
        # "快速链路健康且目标可信时继续基线跟踪", 这正是实验组 A 的基线能力。
        target_untrustworthy = state is not None and (
            state.drift_flag or state.association_confidence < 0.4
        )
        allowed_approach = health is not LinkHealth.FAULT and not target_untrustworthy

        surge = command.surge
        if not allowed_approach:
            surge = min(0.0, surge)

        final = ControlCommand(
            timestamp=command.timestamp,
            frame_id=command.frame_id,
            mode=command.mode,
            surge=clamp_limits(surge, self.limits["surge"]),
            yaw=clamp_limits(command.yaw, self.limits["yaw"]),
            heave=clamp_limits(command.heave, self.limits["heave"]),
            reason=command.reason if allowed_approach else f"{command.reason} | no-approach",
            valid=command.valid,
        )
        result = ArbitrationResult(final, health, reason, allowed_approach)
        self.last_result = result
        return result
