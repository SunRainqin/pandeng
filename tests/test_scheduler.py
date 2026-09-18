"""单元测试: 异步调度器。

覆盖方案第 4.2 节的可验证条款:
- 默认每 N 个处理帧触发一次;
- 同时最多执行一次校正(不排队);
- 等待队列仅保留最新请求;
- 事件触发同样限流;
- 快速链路不等待校正结果;
- 慢链路超时被标记, 连续超限后暂停校正。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from pandeng.config import load_config
from pandeng.memory.template_bank import TemplateBank
from pandeng.perception.corrector import SlowCorrector
from pandeng.perception.embedder import MockEmbedder
from pandeng.scheduling.corrector_scheduler import SlowLoopScheduler
from pandeng.types import CorrectionStatus, Detection, TargetState

SIZE = (1280, 720)


class DummyCorrector:
    """可控制耗时的假校正器。"""

    def __init__(self, latency_s: float = 0.0) -> None:
        self.latency_s = latency_s
        self.calls = 0

    def correct(self, request, *, predicted_bbox=None):
        from pandeng.types import CorrectionResult

        self.calls += 1
        if self.latency_s:
            time.sleep(self.latency_s)
        now = time.monotonic()
        return CorrectionResult(
            frame_id=request.frame_id,
            timestamp=request.timestamp,
            wall_time=now,
            status=CorrectionStatus.OK,
            target_track_id=request.target_track_id,
            generation=request.generation,
            confidence=0.9,
            latency_s=self.latency_s,
        )


def make_state(frame_id: int, **changes) -> TargetState:
    base = dict(
        frame_id=frame_id,
        timestamp=time.monotonic(),
        bbox=np.array([590.0, 310.0, 690.0, 410.0], dtype=np.float32),
        track_id=1,
        visible=True,
        generation=1,
        score=0.9,
    )
    base.update(changes)
    return TargetState(**base)


def make_detections():
    return [Detection(xyxy=np.array([590.0, 310.0, 690.0, 410.0], dtype=np.float32), score=0.9)]


def make_scheduler(corrector=None, **overrides) -> SlowLoopScheduler:
    cfg = load_config()
    data = cfg.scheduler.to_dict()
    for key_path, value in overrides.items():
        node = data
        parts = key_path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    from pandeng.config import Config

    scheduler_cfg = Config(data)
    corrector = corrector or DummyCorrector()
    return SlowLoopScheduler(scheduler_cfg, corrector, SIZE)


def test_periodic_trigger_every_n_frames():
    """默认每 10 个处理帧触发一次。

    本用例只验证周期触发逻辑, 因此关闭限流(限流由
    `test_event_trigger_is_rate_limited` 单独覆盖), 避免纯计算循环中
    "两次提交间隔 < 1/max_request_hz" 造成的偶发跳帧。
    """
    scheduler = make_scheduler(interval_frames=10, warmup_skip_frames=0)
    scheduler._respect_rate_limit = lambda trigger, now: True
    image = np.zeros((720, 1280, 3), dtype=np.uint8)

    triggered = []
    for frame_id in range(1, 61):
        reason = scheduler.maybe_submit(
            frame_id=frame_id,
            timestamp=time.monotonic(),
            state=make_state(frame_id),
            detections=make_detections(),
            image=image,
        )
        if reason == "interval":
            triggered.append(frame_id)
    assert triggered == [10, 20, 30, 40, 50, 60], f"实际触发帧: {triggered}"


def test_warmup_frames_are_skipped():
    scheduler = make_scheduler(warmup_skip_frames=5, interval_frames=1, max_request_hz=1000.0)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    for frame_id in range(1, 6):
        assert (
            scheduler.maybe_submit(
                frame_id=frame_id,
                timestamp=time.monotonic(),
                state=make_state(frame_id),
                detections=make_detections(),
                image=image,
            )
            is None
        )
    assert scheduler.stats.skipped_warmup == 5


def test_fast_loop_does_not_wait_for_correction():
    """快速链路不等待校正结果: 慢校正进行中时, maybe_submit 必须立即返回。"""
    corrector = DummyCorrector(latency_s=0.3)
    scheduler = make_scheduler(corrector, interval_frames=1, warmup_skip_frames=0, max_request_hz=1000.0)
    scheduler.start()
    try:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        scheduler.maybe_submit(
            frame_id=1,
            timestamp=time.monotonic(),
            state=make_state(1),
            detections=make_detections(),
            image=image,
        )
        time.sleep(0.05)  # 让工作线程进入校正
        assert scheduler.inflight

        started = time.monotonic()
        for frame_id in range(2, 30):
            scheduler.maybe_submit(
                frame_id=frame_id,
                timestamp=time.monotonic(),
                state=make_state(frame_id),
                detections=make_detections(),
                image=image,
            )
        elapsed = time.monotonic() - started
        assert elapsed < 0.1, f"快速链路被阻塞了 {elapsed:.3f}s"
        # 校正进行中时不得再提交
        assert scheduler.stats.submitted == 1
    finally:
        scheduler.stop()


def test_at_most_one_correction_in_flight():
    corrector = DummyCorrector(latency_s=0.2)
    scheduler = make_scheduler(corrector, interval_frames=1, warmup_skip_frames=0, max_request_hz=1000.0)
    scheduler.start()
    try:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        for frame_id in range(1, 200):
            scheduler.maybe_submit(
                frame_id=frame_id,
                timestamp=time.monotonic(),
                state=make_state(frame_id),
                detections=make_detections(),
                image=image,
            )
        time.sleep(0.5)
        # 队列容量为 1, 因此在 200 次请求中真正执行的次数远小于 200
        assert corrector.calls < 20
    finally:
        scheduler.stop()


def test_queue_keeps_only_latest_request():
    # 通过构造参数设置, 而不是构造后再赋属性 —— 后者的 _periodic_anchor 仍是
    # 按旧 interval_frames 算出的值, 会让节拍永远不触发。
    scheduler = make_scheduler(queue_size=1, interval_frames=1, warmup_skip_frames=0)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    # 不启动工作线程, 手动提交两次; 需要越过 1/max_request_hz 的提交间隔
    scheduler.maybe_submit(
        frame_id=1, timestamp=time.monotonic(), state=make_state(1),
        detections=make_detections(), image=image,
    )
    scheduler._last_submit_t -= 1.0  # 模拟时间已推进
    scheduler.maybe_submit(
        frame_id=2, timestamp=time.monotonic(), state=make_state(2),
        detections=make_detections(), image=image,
    )
    assert scheduler._queue.qsize() == 1
    assert scheduler._queue.get_nowait().frame_id == 2, "队列应只保留最新请求"
    assert scheduler.stats.dropped_queue_full == 1, "被覆盖的旧请求应计入丢弃统计"


def test_deferred_periodic_tick_fires_when_window_opens():
    """被限流推迟的周期节拍, 必须在限流窗口打开后立刻补发, 不能干等到下一个节拍边界。

    这是真实踩到的缺陷: 慢链路抢占 CPU/GIL 后主循环追赶, 帧突发把多个节拍压缩
    到同一个限流窗口内, 墙钟限流器把它们逐个吃掉, 实测 30 次周期触发只提交了
    11 次(0.74Hz, 而设计值是 2Hz)。周期节拍在方案第 4.2 节里是按帧数定义的,
    所以它只能"晚到", 不能"丢失"。
    """
    interval = 50
    scheduler = make_scheduler(
        interval_frames=interval,
        warmup_skip_frames=0,
        max_request_hz=2.0,          # 0.5s 限流窗口, 远大于 50 帧内喂帧的耗时
        event_min_interval_s=10.0,
    )
    image = np.zeros((720, 1280, 3), dtype=np.uint8)

    def feed(fid: int) -> None:
        scheduler.maybe_submit(
            frame_id=fid,
            timestamp=time.monotonic(),
            state=make_state(fid),
            detections=make_detections(),
            image=image,
        )

    # 1) 正常节奏走到第一个节拍(frame 50), 正常提交
    for fid in range(1, interval + 1):
        feed(fid)
        time.sleep(0.01)
    assert scheduler.stats.periodic == 1
    assert scheduler._periodic_anchor == interval

    # 2) 帧突发: 第二个节拍(frame 100)在限流窗口内到期 -> 被推迟
    for fid in range(interval + 1, 2 * interval + 1):
        feed(fid)
    assert scheduler.stats.periodic == 1, "限流窗口内不应提交"
    assert scheduler.stats.deferred_periodic == 1

    # 3) 继续喂普通帧; 窗口一打开就应补发, 而不是等到下一个边界(frame 150)
    fid = 2 * interval
    deadline = time.monotonic() + 3.0
    while scheduler.stats.periodic == 1 and time.monotonic() < deadline:
        fid += 1
        feed(fid)
        time.sleep(0.02)

    assert scheduler.stats.periodic == 2, "被推迟的节拍没有补发"
    assert fid < 3 * interval - 10, (
        f"补发发生在 frame {fid}, 说明它一直等到了下一个节拍边界 "
        f"(应远早于 frame {3 * interval})"
    )


def test_periodic_cadence_is_not_accelerated_by_deferral():
    """补发后必须重新锚定: 实际周期不得快于 1/interval_frames(方案第 4.2 节)。"""
    interval = 10
    scheduler = make_scheduler(
        interval_frames=interval,
        warmup_skip_frames=0,
        max_request_hz=5.0,       # 上限 5Hz, 高于 20fps/10帧=2Hz 的设计值
    )
    image = np.zeros((720, 1280, 3), dtype=np.uint8)

    frames = 400
    for fid in range(1, frames + 1):
        scheduler.maybe_submit(
            frame_id=fid,
            timestamp=time.monotonic(),
            state=make_state(fid),
            detections=make_detections(),
            image=image,
        )
        time.sleep(0.001)         # 1000fps: 每 10 帧 0.01s, 远快于 0.2s 限流窗口

    # 400 帧最多 40 个节拍; 无论怎么补发都不能超过它
    assert scheduler.stats.periodic <= frames // interval, (
        f"提交了 {scheduler.stats.periodic} 次, 超过节拍上限 {frames // interval}"
    )


def test_event_trigger_is_rate_limited():
    scheduler = make_scheduler(
        interval_frames=1_000_000,
        warmup_skip_frames=0,
        event_min_interval_s=10.0,
        max_request_hz=1000.0,
    )
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    now = time.monotonic()

    # 第一帧建立基线(不可见 -> 可见 会触发 reid)
    scheduler.maybe_submit(
        frame_id=1, timestamp=now, state=make_state(1, visible=False),
        detections=make_detections(), image=image,
    )
    first = scheduler.maybe_submit(
        frame_id=2, timestamp=now, state=make_state(2, visible=True),
        detections=make_detections(), image=image,
    )
    assert first == "reid"

    # 立刻再次满足事件条件, 但应被限流
    scheduler._prev_visible = False
    second = scheduler.maybe_submit(
        frame_id=3, timestamp=now, state=make_state(3, visible=True),
        detections=make_detections(), image=image,
    )
    assert second is None
    assert scheduler.stats.rate_limited >= 1


def test_conf_drop_triggers_event():
    scheduler = make_scheduler(
        interval_frames=1_000_000, warmup_skip_frames=0, max_request_hz=1000.0
    )
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    scheduler.maybe_submit(
        frame_id=1, timestamp=time.monotonic(), state=make_state(1, score=0.9),
        detections=make_detections(), image=image,
    )
    reason = scheduler.maybe_submit(
        frame_id=2, timestamp=time.monotonic(), state=make_state(2, score=0.2),
        detections=make_detections(), image=image,
    )
    assert reason == "conf_drop"


def test_edge_trigger():
    scheduler = make_scheduler(
        interval_frames=1_000_000, warmup_skip_frames=0, max_request_hz=1000.0
    )
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    edge_bbox = np.array([0.0, 300.0, 80.0, 400.0], dtype=np.float32)
    reason = scheduler.maybe_submit(
        frame_id=1, timestamp=time.monotonic(), state=make_state(1, bbox=edge_bbox),
        detections=make_detections(), image=image,
    )
    assert reason == "edge"


def test_timeout_marks_result_and_pauses_after_repeat_overruns():
    """单次超时被标记为 TIMEOUT; 连续超限后暂停校正, 快速环继续基线跟踪。"""
    corrector = DummyCorrector(latency_s=0.2)
    scheduler = make_scheduler(
        corrector,
        interval_frames=1,
        warmup_skip_frames=0,
        timeout_s=0.05,
        max_consecutive_overruns=2,
        cooldown_s=30.0,
    )
    scheduler._respect_rate_limit = lambda trigger, now: True

    scheduler.start()
    try:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        deadline = time.monotonic() + 5.0
        frame_id = 0
        while time.monotonic() < deadline and not scheduler.is_paused:
            frame_id += 1
            scheduler.maybe_submit(
                frame_id=frame_id,
                timestamp=time.monotonic(),
                state=make_state(frame_id),
                detections=make_detections(),
                image=image,
            )
            time.sleep(0.01)

        assert scheduler.stats.timeouts >= 2, (
            f"应至少记录 2 次超时, 实际 {scheduler.stats.timeouts}"
        )
        assert scheduler.is_paused, "连续超限后应暂停校正"

        # 暂停期间不再接受新请求
        before = scheduler.stats.submitted
        assert (
            scheduler.maybe_submit(
                frame_id=frame_id + 1,
                timestamp=time.monotonic(),
                state=make_state(frame_id + 1),
                detections=make_detections(),
                image=image,
            )
            is None
        )
        assert scheduler.stats.submitted == before
        assert scheduler.stats.paused >= 1
    finally:
        scheduler.stop()


def test_disabled_slow_link_still_yields_baseline_tracking():
    """慢链路不可用时, 快速链路健康且目标可信 -> 继续基线跟踪。"""
    from pandeng.control.arbiter import ControlArbiter, LinkHealth
    from pandeng.types import ControlCommand, ControlMode

    cfg = load_config().controller
    arbiter = ControlArbiter(cfg)
    arbiter.report_perception(time.monotonic())
    arbiter.report_slow_link(ok=False)

    state = make_state(1)
    health, reason = arbiter.evaluate_health(state)
    assert health is LinkHealth.DEGRADED

    command = ControlCommand(
        timestamp=time.monotonic(), frame_id=1, mode=ControlMode.APPROACH, surge=0.2
    )
    result = arbiter.arbitrate(command, state)
    assert result.health is LinkHealth.DEGRADED
    assert result.allowed_to_approach, "慢链路降级不应影响可信目标的基线跟踪"


def test_slow_link_unavailable_and_untrustworthy_target_exits():
    from pandeng.control.arbiter import ControlArbiter, LinkHealth

    cfg = load_config().controller
    arbiter = ControlArbiter(cfg)
    arbiter.report_perception(time.monotonic())
    arbiter.report_slow_link(ok=False)

    health, _ = arbiter.evaluate_health(make_state(1, visible=False))
    assert health is LinkHealth.FAULT


def test_stale_perception_forces_exit():
    from pandeng.control.arbiter import ControlArbiter, LinkHealth
    from pandeng.types import ControlCommand, ControlMode

    cfg = load_config().controller
    arbiter = ControlArbiter(cfg)
    arbiter.health.last_perception_time = time.monotonic() - 10.0

    health, _ = arbiter.evaluate_health(make_state(1))
    assert health is LinkHealth.FAULT

    result = arbiter.arbitrate(
        ControlCommand(timestamp=time.monotonic(), frame_id=1, mode=ControlMode.APPROACH, surge=0.3),
        make_state(1),
    )
    assert result.command.mode is ControlMode.EXIT
    assert result.command.surge == pytest.approx(0.0)


def test_slow_corrector_never_returns_control_command():
    """结构性约束: 慢速环的返回类型里没有控制量。"""
    embedder = MockEmbedder()
    bank = TemplateBank()
    corrector = SlowCorrector(load_config().corrector, embedder, bank, SIZE)

    from pandeng.types import CorrectionRequest

    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    request = CorrectionRequest(
        frame_id=1,
        timestamp=time.monotonic(),
        image=image,
        candidates=tuple(make_detections()),
        target_bbox=np.array([590.0, 310.0, 690.0, 410.0], dtype=np.float32),
        target_track_id=1,
        generation=1,
    )
    result = corrector.correct(request)
    assert not hasattr(result, "surge")
    assert not hasattr(result, "yaw")
    assert not hasattr(result, "heave")
