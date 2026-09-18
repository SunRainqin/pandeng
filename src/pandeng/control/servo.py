"""图像伺服 PD/PID 控制器。

对应方案第 3 节。控制律与策略表逐条对应:

============ ==========================================================
条件          控制动作
============ ==========================================================
目标偏离中心  减小前进, 优先转向
目标稳定居中  允许低输出前进
鱼框快速增大  减速或停止接近
目标丢失      取消追近, 限时转向搜索, 超时退出
感知过期/异常 撤销跟随, 执行底层退出策略
============ ==========================================================

约定:
- `surge` 正值为前进, `yaw` 正值为右转, `heave` 正值为下潜;
- 所有输出经过死区、限幅与变化率限制;
- 框面积只用于判断"是否在靠近", **不换算米制鱼距**(方案并要求朝下声纳
  只提供离底约束, 不在本模块内换算距离);
- 控制器按 `rate_hz` 发布受限控制建议, 未到发布时刻返回上一次的命令。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..types import ControlCommand, ControlMode, TargetState
from ..utils.geometry import clamp_limits, deadzone, slew_limit

__all__ = ["ImageServoController", "PIDState"]

LOGGER = logging.getLogger(__name__)


@dataclass
class PIDState:
    """单轴 PID 内部状态。"""

    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    integral: float = 0.0
    previous_error: Optional[float] = None
    previous_time: Optional[float] = None
    integral_limit: float = 1.0
    output: float = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None
        self.previous_time = None
        self.output = 0.0

    def step(self, error: float, now: float) -> float:
        dt = 0.0 if self.previous_time is None else max(1e-4, now - self.previous_time)
        derivative = 0.0
        if self.previous_error is not None and dt > 0.0:
            derivative = (error - self.previous_error) / dt
        if self.ki != 0.0 and dt > 0.0:
            self.integral = float(
                np.clip(self.integral + error * dt, -self.integral_limit, self.integral_limit)
            )
        self.previous_error = float(error)
        self.previous_time = float(now)
        value = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.output = float(value)
        return self.output


class ImageServoController:
    """图像伺服控制器。"""

    def __init__(self, cfg, image_size: Tuple[int, int]) -> None:
        self.cfg = cfg
        self.width, self.height = int(image_size[0]), int(image_size[1])
        self.rate_hz = float(cfg.get_path("rate_hz", 10.0))
        self.mode = str(cfg.get_path("mode", "pd")).lower()

        gains = cfg.get_path("gains", {})
        self.yaw_pid = self._make_pid(gains, "yaw")
        self.heave_pid = self._make_pid(gains, "heave")
        self.surge_pid = self._make_pid(gains, "surge")

        dz = cfg.get_path("deadzone", {})
        self.deadzone_ex = float(dz.get("ex", 0.05) if hasattr(dz, "get") else 0.05)
        self.deadzone_ey = float(dz.get("ey", 0.05) if hasattr(dz, "get") else 0.05)

        lim = cfg.get_path("limits", {})
        self.limits = {
            "surge": self._limits(lim, "surge", (-0.2, 0.5)),
            "yaw": self._limits(lim, "yaw", (-0.6, 0.6)),
            "heave": self._limits(lim, "heave", (-0.3, 0.3)),
        }
        rl = cfg.get_path("rate_limit", {})
        self.rate_limit = {
            "surge": self._scalar(rl, "surge", 0.4),
            "yaw": self._scalar(rl, "yaw", 0.8),
            "heave": self._scalar(rl, "heave", 0.3),
        }

        sp = cfg.get_path("surge_policy", {})
        self.surge_base = self._scalar(sp, "base", 0.15)
        self.surge_center_gate = self._scalar(sp, "center_gate", 0.25)
        self.area_stop = self._scalar(sp, "area_stop", 0.30)
        self.area_growth_stop = self._scalar(sp, "area_growth_stop", 0.08)
        self.area_growth_decay = self._scalar(sp, "area_growth_decay", 0.5)

        sr = cfg.get_path("search", {})
        self.search_yaw_rate = self._scalar(sr, "yaw_rate", 0.35)
        self.search_timeout_s = self._scalar(sr, "timeout_s", 6.0)

        dp = cfg.get_path("depth", {})
        self.fixed_depth_enabled = bool(dp.get("fixed_enabled", True) if hasattr(dp, "get") else True)
        self.fixed_target_m = self._scalar(dp, "fixed_target_m", 1.5)
        self.slow_depth_adjust = bool(dp.get("slow_adjust", False) if hasattr(dp, "get") else False)
        self.depth_adjust_gain = self._scalar(dp, "adjust_gain", 0.15)

        sf = cfg.get_path("safety", {})
        self.perception_timeout_s = self._scalar(sf, "perception_timeout_s", 0.5)
        self.on_perception_stale = str(
            sf.get("on_perception_stale", "exit") if hasattr(sf, "get") else "exit"
        )

        # 运行状态
        self._last_publish_t = -1e9
        self._last_command = ControlCommand(
            timestamp=time.monotonic(), frame_id=0, mode=ControlMode.HOLD, reason="init"
        )
        self._previous_outputs = {"surge": 0.0, "yaw": 0.0, "heave": 0.0}
        self._lost_since: Optional[float] = None
        self._last_perception_t: Optional[float] = None

    # -- 配置读取工具 -------------------------------------------------------
    @staticmethod
    def _scalar(node, key: str, default: float) -> float:
        if node is None or not hasattr(node, "get"):
            return float(default)
        return float(node.get(key, default))

    @staticmethod
    def _limits(node, key: str, default: Tuple[float, float]) -> Tuple[float, float]:
        value = node.get(key, default) if hasattr(node, "get") else default
        return (float(value[0]), float(value[1]))

    def _make_pid(self, gains, axis: str) -> PIDState:
        node = gains.get(axis, {}) if hasattr(gains, "get") else {}
        kp = float(node.get("kp", 0.0) if hasattr(node, "get") else 0.0)
        ki = float(node.get("ki", 0.0) if hasattr(node, "get") else 0.0)
        kd = float(node.get("kd", 0.0) if hasattr(node, "get") else 0.0)
        if self.mode == "pd":
            ki = 0.0
        return PIDState(kp=kp, ki=ki, kd=kd)

    # -- 生命周期 -----------------------------------------------------------
    def reset(self) -> None:
        self.yaw_pid.reset()
        self.heave_pid.reset()
        self.surge_pid.reset()
        self._previous_outputs = {"surge": 0.0, "yaw": 0.0, "heave": 0.0}
        self._lost_since = None

    def notify_perception(
        self,
        now: Optional[float] = None,
        *,
        device_ok: bool = True,
    ) -> None:
        """由感知环每帧调用, 用于判断"感知过期"。"""
        if device_ok:
            self._last_perception_t = time.monotonic() if now is None else float(now)

    @property
    def perception_age(self) -> float:
        if self._last_perception_t is None:
            return float("inf")
        return max(0.0, time.monotonic() - self._last_perception_t)

    # -- 主接口 -------------------------------------------------------------
    def update(
        self,
        state: Optional[TargetState],
        *,
        now: Optional[float] = None,
        device_ok: bool = True,
        exploration_yaw: float = 0.0,
        prediction=None,
    ) -> ControlCommand:
        """计算控制建议。

        Args:
            state: 目标状态; None 表示本帧无目标。
            device_ok: 设备健康标志。
            exploration_yaw: 上层仲裁给出的搜索转向(如视觉丢失时的扫描),
                优先级高于图像伺服输出。
            prediction: `PredictionResult` 或 None。预测只提供**受限**修正:
                目标丢失时不使用预测(禁止盲目前进), 且修正量远小于伺服主输出。

        Returns:
            经完整限幅后的控制命令。未到发布时刻时返回上一次命令。
        """
        now = time.monotonic() if now is None else float(now)

        # 注意: 这里**不能**更新时间戳。感知时间戳只应由 `notify_perception`
        # 在真正拿到新一帧时写入, 否则控制器每次自调用都会把"感知过期"刷新掉,
        # 过期检测将永远不触发。
        # 感知过期或设备异常: 撤销跟随, 执行底层退出策略
        age = self.perception_age
        if not device_ok or age > self.perception_timeout_s:
            reason = "device_fault" if not device_ok else f"perception_stale({age:.2f}s)"
            mode = ControlMode.EXIT if self.on_perception_stale == "exit" else ControlMode.HOLD
            return self._emit(
                now, state, mode=mode, surge=0.0, yaw=0.0, heave=0.0, reason=reason
            )

        due = (now - self._last_publish_t) >= 1.0 / max(1e-6, self.rate_hz)
        if not due:
            return self._last_command
        self._last_publish_t = now

        if state is None or not state.visible:
            # 目标丢失: 预测不参与(禁止盲目前进), 走搜索/退出分支
            return self._handle_lost(now, state, exploration_yaw)

        return self._handle_visible(now, state, exploration_yaw, prediction)

    # -- 分支 ---------------------------------------------------------------
    def _handle_lost(
        self,
        now: float,
        state: Optional[TargetState],
        exploration_yaw: float,
    ) -> ControlCommand:
        """目标丢失: 取消追近, 限时转向搜索, 超时退出。"""
        if self._lost_since is None:
            self._lost_since = now
        lost_duration = now - self._lost_since

        if lost_duration > self.search_timeout_s:
            return self._emit(
                now, state, mode=ControlMode.EXIT, surge=0.0, yaw=0.0, heave=0.0,
                reason=f"search_timeout({lost_duration:.1f}s)",
            )

        yaw = exploration_yaw if exploration_yaw != 0.0 else self.search_yaw_rate
        return self._emit(
            now, state, mode=ControlMode.SEARCH, surge=0.0, yaw=yaw, heave=0.0,
            reason=f"lost({lost_duration:.1f}s)",
        )

    def _handle_visible(
        self,
        now: float,
        state: TargetState,
        exploration_yaw: float,
        prediction=None,
    ) -> ControlCommand:
        self._lost_since = None

        # 慢速环判定漂移: 停止错误追近, 但仍允许转向以维持视角
        drifting = state.drift_flag

        ex = deadzone(state.ex, self.deadzone_ex)
        ey = deadzone(state.ey, self.deadzone_ey)

        yaw = self.yaw_pid.step(ex, now)
        heave_raw = 0.0 if self.fixed_depth_enabled and not self.slow_depth_adjust \
            else self.heave_pid.step(ey, now)

        # --- 前进策略 -----------------------------------------------------
        surge, surge_reason = self._surge_policy(state, drifting)

        # 目标偏离中心 -> 减小前进, 优先转向
        centered = abs(state.ex) <= self.surge_center_gate
        if not centered:
            surge *= self._center_scale(state.ex)

        # --- 预测的受限修正 -----------------------------------------------
        # 只做小幅偏置, 绝不改变"丢失即禁止前进"的安全约束
        prediction_note = ""
        if prediction is not None and getattr(prediction, "valid", False) and not drifting:
            yaw += float(prediction.yaw_hint)
            scale = float(np.clip(prediction.surge_scale, 0.0, 1.0))
            surge *= scale
            if getattr(prediction, "will_leave_view", False):
                surge = min(surge, 0.0)
                prediction_note = " | pred:leave_view"
            else:
                prediction_note = f" | pred(conf={prediction.confidence:.2f})"

        if exploration_yaw != 0.0:
            yaw = exploration_yaw
            surge = min(surge, 0.0)

        if drifting:
            surge = min(surge, 0.0)
            surge_reason = "drift: 停止追近"

        if self.fixed_depth_enabled and not self.slow_depth_adjust:
            mode = ControlMode.APPROACH if surge > 1e-3 else ControlMode.HOLD
            if not centered:
                mode = ControlMode.TRACK
            reason = "fixed_depth | " + surge_reason + prediction_note
        else:
            mode = ControlMode.APPROACH if surge > 1e-3 else ControlMode.HOLD
            reason = surge_reason + prediction_note

        return self._emit(
            now, state, mode=mode, surge=surge, yaw=yaw, heave=heave_raw, reason=reason
        )

    def _surge_policy(self, state: TargetState, drifting: bool) -> Tuple[float, str]:
        """按"鱼框快速增大 -> 减速或停止接近"决定前进量。"""
        if drifting:
            return 0.0, "drift"

        if state.area_ratio >= self.area_stop:
            return 0.0, f"area {state.area_ratio:.3f} >= {self.area_stop:.2f}: 停止接近"

        if state.area_growth > self.area_growth_stop:
            # 面积增长越快, 前进量压得越低, 允许短暂后退
            excess = state.area_growth - self.area_growth_stop
            scale = float(np.clip(1.0 - self.area_growth_decay * (1.0 + excess), -0.5, 1.0))
            return self.surge_base * scale, f"area_growth {state.area_growth:.3f}: 减速"

        # 目标稳定居中 -> 允许低输出前进
        return self.surge_base, "low_output_approach"

    def _center_scale(self, ex: float) -> float:
        """偏离中心时按程度压低前进量。"""
        return float(np.clip(1.0 - abs(ex), 0.0, 1.0))

    # -- 输出处理 -----------------------------------------------------------
    def _emit(
        self,
        now: float,
        state: Optional[TargetState],
        *,
        mode: ControlMode,
        surge: float,
        yaw: float,
        heave: float,
        reason: str,
    ) -> ControlCommand:
        """统一施加限幅与变化率限制后发布。"""
        # 控制命令按 rate_hz 更新, 因此变化率上限以发布周期折算
        dt = 1.0 / max(1e-6, self.rate_hz)

        values = {}
        for axis, raw in (("surge", surge), ("yaw", yaw), ("heave", heave)):
            limited = slew_limit(
                clamp_limits(raw, self.limits[axis]),
                self._previous_outputs[axis],
                self.rate_limit[axis] * dt,
            )
            values[axis] = clamp_limits(limited, self.limits[axis])
        self._previous_outputs = dict(values)

        command = ControlCommand(
            timestamp=now,
            frame_id=state.frame_id if state is not None else -1,
            mode=mode,
            surge=values["surge"],
            yaw=values["yaw"],
            heave=values["heave"],
            reason=reason,
            valid=mode is not ControlMode.EXIT,
        ).clamped()

        self._last_command = command
        return command

    @property
    def last_command(self) -> ControlCommand:
        return self._last_command

    def depth_setpoint(self) -> Optional[float]:
        """固定深度阶段的目标深度(米)。"""
        return self.fixed_target_m if self.fixed_depth_enabled else None
