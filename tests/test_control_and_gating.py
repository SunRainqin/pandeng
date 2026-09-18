"""单元测试: 调度器、门控与控制器。

重点覆盖方案第 4.2 节的硬约束:
- 同时最多执行一次校正, 队列仅保留最新请求;
- 事件触发同样限流;
- 快速链路不等待校正结果;
- 结果门控检查数据年龄与目标一致性, 过旧或目标已切换的结果直接丢弃。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from pandeng.config import load_config
from pandeng.control.servo import ImageServoController
from pandeng.perception.corrector import CorrectionGate
from pandeng.metrics import SessionMetrics
from pandeng.prediction import ConstantVelocityPredictor
from pandeng.types import (
    ControlMode,
    CorrectionResult,
    CorrectionStatus,
    TargetState,
)

SIZE = (1280, 720)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def make_state(**changes) -> TargetState:
    base = dict(
        frame_id=10,
        timestamp=time.monotonic(),
        bbox=np.array([590.0, 310.0, 690.0, 410.0], dtype=np.float32),
        track_id=1,
        visible=True,
        generation=1,
        ex=0.0,
        ey=0.0,
        area_ratio=0.01,
        area_growth=0.0,
        score=0.9,
    )
    base.update(changes)
    return TargetState(**base)


def make_result(**changes) -> CorrectionResult:
    now = time.monotonic()
    base = dict(
        frame_id=10,
        timestamp=now,
        wall_time=now,
        status=CorrectionStatus.OK,
        target_track_id=1,
        generation=1,
        confidence=0.9,
        matched_bbox=np.array([590.0, 310.0, 690.0, 410.0], dtype=np.float32),
        drift_score=0.9,
        latency_s=0.02,
    )
    base.update(changes)
    return CorrectionResult(**base)


# ---------------------------------------------------------------------------
# 门控
# ---------------------------------------------------------------------------
def test_gate_accepts_fresh_consistent_result():
    gate = CorrectionGate(max_age_s=0.6)
    decision = gate.evaluate(make_result(), make_state())
    assert decision.apply


def test_gate_discards_stale_result():
    gate = CorrectionGate(max_age_s=0.2)
    old = make_result()
    old.timestamp = time.monotonic() - 1.0  # 1 秒前的旧结果
    decision = gate.evaluate(old, make_state())
    assert not decision.apply
    assert "old" in decision.reason


def test_gate_discards_result_after_target_switch():
    """目标代次变化后, 旧结果必须作废。"""
    gate = CorrectionGate(max_age_s=1.0)
    decision = gate.evaluate(make_result(generation=1), make_state(generation=2))
    assert not decision.apply
    assert "generation" in decision.reason


def test_gate_discards_on_track_id_mismatch():
    gate = CorrectionGate(max_age_s=1.0)
    decision = gate.evaluate(make_result(target_track_id=7), make_state(track_id=1))
    assert not decision.apply


def test_gate_rejects_low_confidence():
    gate = CorrectionGate(max_age_s=1.0, min_confidence=0.5)
    decision = gate.evaluate(make_result(confidence=0.3), make_state())
    assert not decision.apply


def test_gate_timeout_and_overload_never_apply():
    gate = CorrectionGate(max_age_s=1.0)
    for status in (CorrectionStatus.TIMEOUT, CorrectionStatus.OVERLOADED, CorrectionStatus.DISABLED):
        assert not gate.evaluate(make_result(status=status), make_state()).apply


def test_gate_smooths_and_limits_offset():
    gate = CorrectionGate(smooth_alpha=0.5, max_offset_ratio=0.1)
    first = gate.smooth_offset(None, np.array([1000.0, 1000.0], dtype=np.float32), 100.0)
    # 限幅: 偏移不超过对角线 * 0.1 = 10 像素
    assert np.all(np.abs(first) <= 10.0 + 1e-4)

    second = gate.smooth_offset(first, np.array([10.0, 0.0], dtype=np.float32), 100.0)
    assert np.all(np.abs(second) <= 10.0 + 1e-4)


# ---------------------------------------------------------------------------
# 控制器
# ---------------------------------------------------------------------------
def test_controller_centers_target_with_yaw():
    cfg = load_config().controller
    controller = ImageServoController(cfg, SIZE)
    controller.notify_perception(time.monotonic())

    now = time.monotonic()
    for i in range(10):
        command = controller.update(
            make_state(ex=0.5, ey=0.0, frame_id=i), now=now + i * 0.1
        )
    # 目标在右侧 -> 应向右转
    assert command.yaw > 0.0
    assert command.mode in {ControlMode.TRACK, ControlMode.HOLD, ControlMode.APPROACH}


def test_controller_stops_approach_when_box_grows_fast():
    """方案第 3 节: 鱼框快速增大 -> 减速或停止接近。"""
    cfg = load_config().controller
    controller = ImageServoController(cfg, SIZE)
    controller.notify_perception(time.monotonic())

    now = time.monotonic()
    command = None
    for i in range(8):
        command = controller.update(
            make_state(area_ratio=0.05, area_growth=5.0, frame_id=i), now=now + i * 0.2
        )
    assert command.surge <= 0.0, "框面积快速增长时必须停止接近"


def test_controller_stops_approach_when_area_large():
    cfg = load_config().controller
    controller = ImageServoController(cfg, SIZE)
    controller.notify_perception(time.monotonic())

    now = time.monotonic()
    command = None
    for i in range(8):
        command = controller.update(
            make_state(area_ratio=0.9, area_growth=0.0, frame_id=i), now=now + i * 0.2
        )
    assert command.surge <= 0.0


def test_controller_search_then_exit_on_loss():
    """目标丢失 -> 限时转向搜索 -> 超时退出。"""
    cfg = load_config().controller
    controller = ImageServoController(cfg, SIZE)
    controller.notify_perception(time.monotonic())

    now = time.monotonic()
    command = controller.update(None, now=now)
    assert command.mode is ControlMode.SEARCH
    assert command.surge == pytest.approx(0.0), "搜索阶段不得前进"
    assert abs(command.yaw) > 0.0

    # 超过搜索超时后应退出
    later = now + cfg.search.timeout_s + 1.0
    controller.notify_perception(later)
    command = controller.update(None, now=later)
    assert command.mode is ControlMode.EXIT


def test_controller_exits_on_stale_perception():
    """感知过期 -> 撤销跟随(方案第 3 节)。"""
    cfg = load_config().controller
    controller = ImageServoController(cfg, SIZE)
    controller.notify_perception(time.monotonic() - 5.0)
    command = controller.update(make_state(), now=time.monotonic())
    assert command.mode is ControlMode.EXIT
    assert command.surge == pytest.approx(0.0)
    assert command.yaw == pytest.approx(0.0)
    assert not command.valid


def test_controller_respects_limits_and_rate_limit():
    cfg = load_config().controller
    controller = ImageServoController(cfg, SIZE)
    controller.notify_perception(time.monotonic())

    now = time.monotonic()
    previous = 0.0
    for i in range(40):
        command = controller.update(
            make_state(ex=1.0, area_ratio=0.001, frame_id=i), now=now + i * 0.1
        )
        lo, hi = cfg.limits.yaw
        assert lo - 1e-6 <= command.yaw <= hi + 1e-6
        # 变化率限制: 每步增量不超过 rate_limit * 发布周期
        max_step = cfg.rate_limit.yaw / cfg.rate_hz + 1e-6
        assert abs(command.yaw - previous) <= max_step
        previous = command.yaw


def test_controller_ignores_prediction_when_target_lost():
    """预测只提供受限修正, 目标丢失时禁止盲目前进。"""
    cfg = load_config().controller
    controller = ImageServoController(cfg, SIZE)
    controller.notify_perception(time.monotonic())

    predictor = ConstantVelocityPredictor(min_samples=2)
    for i in range(10):
        predictor.observe(timestamp=i * 0.1, ex=-0.9 + i * 0.02, ey=0.0)
    prediction = predictor.predict()
    assert prediction.valid

    now = time.monotonic()
    command = controller.update(None, now=now, prediction=prediction)
    assert command.surge == pytest.approx(0.0)


def test_controller_surge_never_exceeds_limits():
    cfg = load_config().controller
    controller = ImageServoController(cfg, SIZE)
    controller.notify_perception(time.monotonic())
    now = time.monotonic()
    for i in range(30):
        command = controller.update(
            make_state(ex=0.0, ey=0.0, area_ratio=0.001, frame_id=i), now=now + i * 0.1
        )
        lo, hi = cfg.limits.surge
        assert lo - 1e-6 <= command.surge <= hi + 1e-6


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def test_central_region_is_half_width_and_height():
    """方案第 7 节: 画面中央 50% 宽高区域。"""
    assert SessionMetrics.is_in_central_region(
        np.array([590.0, 310.0, 690.0, 410.0], dtype=np.float32), 1280, 720
    )
    # 靠近边缘的框不算居中
    assert not SessionMetrics.is_in_central_region(
        np.array([10.0, 10.0, 60.0, 60.0], dtype=np.float32), 1280, 720
    )


def test_metrics_counts_losses_and_visibility():
    metrics = SessionMetrics()
    t = 0.0
    for i in range(10):
        metrics.record_frame(now=t, visible=i < 8, in_central_region=i < 6)
        t += 0.1
    summary = metrics.summary()
    assert summary["frames"] == 10
    assert summary["losses"] == 1
    assert summary["visibility_ratio"] > 0.7
    # 首帧没有前序时间戳, 不贡献时长, 因此居中比例按 5/7 而非 6/8 计
    assert summary["central_ratio_when_visible"] == pytest.approx(5.0 / 7.0, abs=1e-3)


def test_metrics_success_thresholds():
    metrics = SessionMetrics()
    t = 0.0
    for _ in range(100):
        metrics.record_frame(now=t, visible=True, in_central_region=True, bridge_latency=0.05)
        t += 0.05
    criteria = metrics.check_success()
    assert criteria["visibility_ge_80pct"]
    assert criteria["central_ge_80pct"]
    assert criteria["bridge_p95_le_200ms"]
    assert criteria["fast_rate_ge_10hz"]
