"""慢速环异步调度。

严格对应方案第 4.2 节:

- 默认每十个已处理图像帧触发一次; 20Hz 快速链路对应约 2Hz 请求;
- 置信度下降、鱼体交叉、目标接近视场边缘时提前触发;
- 同时最多执行一次校正, 等待队列仅保留最新请求, 事件触发同样限流;
- 快速链路不等待校正结果;
- 慢链路超时或资源超限时暂停校正;
- 慢链路不能绕过快速状态估计直接控制推进器 —— 本模块只产出
  `CorrectionResult`, 绝不产生 `ControlCommand`。

实现要点:
* 单工作线程 + 容量为 1 的队列(旧请求被最新请求覆盖);
* 使用 `threading.Event` 唤醒工作线程, 空闲时不占 CPU;
* 所有时间基于 `time.monotonic()`, 与图像帧时间戳一致。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import List, Optional, Sequence

import numpy as np

from ..perception.association import edge_margin
from ..perception.corrector import SlowCorrector
from ..types import (
    CorrectionRequest,
    CorrectionResult,
    CorrectionStatus,
    Detection,
    TargetState,
)

__all__ = ["SchedulerStats", "SlowLoopScheduler"]

LOGGER = logging.getLogger(__name__)


@dataclass
class SchedulerStats:
    """慢链路运行统计(方案第 7 节要求单独上报这些指标)。"""

    submitted: int = 0
    periodic: int = 0
    event_triggered: int = 0
    dropped_queue_full: int = 0
    rate_limited: int = 0
    completed: int = 0
    timeouts: int = 0
    errors: int = 0
    skipped_warmup: int = 0
    paused: int = 0
    deferred_periodic: int = 0
    latencies: List[float] = field(default_factory=list)
    ages: List[float] = field(default_factory=list)

    def record_latency(self, value: float, window: int = 500) -> None:
        self.latencies.append(float(value))
        if len(self.latencies) > window:
            del self.latencies[: len(self.latencies) - window]

    def record_age(self, value: float, window: int = 500) -> None:
        self.ages.append(float(value))
        if len(self.ages) > window:
            del self.ages[: len(self.ages) - window]

    @staticmethod
    def _p95(values: Sequence[float]) -> Optional[float]:
        if not values:
            return None
        return float(np.percentile(np.asarray(values, dtype=np.float64), 95))

    def summary(self, elapsed_s: float = 0.0) -> dict:
        elapsed = max(1e-6, elapsed_s)
        return {
            "submitted": self.submitted,
            "periodic": self.periodic,
            "event_triggered": self.event_triggered,
            "completed": self.completed,
            "timeouts": self.timeouts,
            "errors": self.errors,
            "dropped_queue_full": self.dropped_queue_full,
            "rate_limited": self.rate_limited,
            "deferred_periodic": self.deferred_periodic,
            "skipped_warmup": self.skipped_warmup,
            "paused": self.paused,
            # 请求率与完成率分开报告: 请求频率不等于实际完成频率
            "request_hz": round(self.submitted / elapsed, 3),
            "completion_hz": round(self.completed / elapsed, 3),
            "completion_ratio": round(self.completed / max(1, self.submitted), 3),
            "latency_p95_s": self._p95(self.latencies),
            "result_age_p95_s": self._p95(self.ages),
        }


class SlowLoopScheduler:
    """DINOv3 低频校正的异步调度器。"""

    def __init__(
        self,
        cfg,
        corrector: SlowCorrector,
        image_size: tuple[int, int],
    ) -> None:
        self.cfg = cfg
        self.corrector = corrector
        self.width, self.height = int(image_size[0]), int(image_size[1])

        self.interval_frames = max(1, int(cfg.get_path("interval_frames", 10)))
        self.max_request_hz = float(cfg.get_path("max_request_hz", 5.0))
        self.event_min_interval_s = float(cfg.get_path("event_min_interval_s", 0.4))
        self.timeout_s = float(cfg.get_path("timeout_s", 1.0))
        self.queue_size = max(1, int(cfg.get_path("queue_size", 1)))
        self.warmup_skip_frames = max(0, int(cfg.get_path("warmup_skip_frames", 5)))

        ev = cfg.get_path("event_triggers", {})
        self.events_enabled = bool(ev.get("enabled", True) if hasattr(ev, "get") else True)
        self.conf_drop = float(ev.get("conf_drop", 0.35) if hasattr(ev, "get") else 0.35)
        self.edge_margin_thresh = float(ev.get("edge_margin", 0.12) if hasattr(ev, "get") else 0.12)
        self.cross_check = bool(ev.get("cross_check", True) if hasattr(ev, "get") else True)
        self.reid_check = bool(ev.get("reid_check", True) if hasattr(ev, "get") else True)

        # 资源保护: 连续超限则暂停校正一段时间
        self.max_consecutive_overruns = int(cfg.get_path("max_consecutive_overruns", 3))
        self.cooldown_s = float(cfg.get_path("cooldown_s", 5.0))
        # 序列纪元: 换序列后递增, 用于丢弃属于上一序列的在飞结果。
        # 仅靠目标代次无法区分 —— 新序列的代次同样从 1 开始。
        self._epoch = 0

        self.stats = SchedulerStats()
        self._queue: Queue[CorrectionRequest] = Queue(maxsize=self.queue_size)
        self._results: Queue[CorrectionResult] = Queue(maxsize=16)
        self._stop = threading.Event()
        self._wakeup = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._inflight = False
        self._lock = threading.Lock()

        self._last_submit_t = -1e9
        self._last_event_t = -1e9
        self._prev_score: Optional[float] = None
        self._prev_visible = False
        self._consecutive_overruns = 0
        self._paused_until = 0.0
        # 周期节拍的锚点帧。节拍按**帧数**定义, 用锚点计数而不是取模:
        # 取模会在补发之后仍按固定边界触发, 使实际节拍快于 1/interval_frames;
        # 锚点法则保证"提交后至少再过 interval_frames 帧才到下一个节拍"。
        self._periodic_anchor = self.warmup_skip_frames

    # -- 生命周期 -----------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="slow-loop", daemon=True)
        self._thread.start()
        LOGGER.info(
            "慢速环已启动: 每 %d 帧触发一次, 请求上限 %.1fHz, 队列容量 %d, 超时 %.2fs",
            self.interval_frames,
            self.max_request_hz,
            self.queue_size,
            self.timeout_s,
        )

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wakeup.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def __enter__(self) -> "SlowLoopScheduler":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    @property
    def inflight(self) -> bool:
        with self._lock:
            return self._inflight

    @property
    def is_paused(self) -> bool:
        return time.monotonic() < self._paused_until

    # -- 提交 ---------------------------------------------------------------
    def begin_sequence(self, epoch: int) -> None:
        """切换序列: 丢弃待处理请求与已产出结果, 并重置节拍与熔断状态。

        工作线程保持运行, 不做 join —— 快速环不应因换序列停顿。正在执行的
        那一次校正会自然结束, 但其结果会因纪元不匹配而被丢弃。
        """
        self._epoch = int(epoch)
        self._drain_queue(self._queue)
        self._drain_queue(self._results)
        self._periodic_anchor = self.warmup_skip_frames
        self._prev_score = None
        self._prev_visible = False
        self._last_submit_t = -1e9
        self._last_event_t = -1e9
        self._consecutive_overruns = 0
        self._paused_until = 0.0

    @staticmethod
    def _drain_queue(queue: "Queue") -> None:
        while True:
            try:
                queue.get_nowait()
            except Empty:
                return

    @property
    def epoch(self) -> int:
        return self._epoch

    def maybe_submit(
        self,
        *,
        frame_id: int,
        timestamp: float,
        state: TargetState,
        detections: Sequence[Detection],
        image: np.ndarray,
        crossing: bool = False,
        predicted_bbox: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """按调度策略决定是否提交校正请求。

        Returns:
            触发原因字符串; 未触发时返回 None。
        """
        now = time.monotonic()

        if self.is_paused:
            self.stats.paused += 1
            return None
        if frame_id <= self.warmup_skip_frames:
            self.stats.skipped_warmup += 1
            return None

        trigger = self._select_trigger(frame_id, state, crossing, now)
        if trigger is None:
            return None

        # 同时最多执行一次校正; 快速链路继续运行, 不等待
        if self.inflight:
            self._note_deferral(trigger, frame_id)
            return None

        if not self._respect_rate_limit(trigger, now):
            self.stats.rate_limited += 1
            self._note_deferral(trigger, frame_id)
            return None

        if trigger == "interval":
            # 重新锚定: 下一次周期节拍要再等 interval_frames 帧
            self._periodic_anchor = int(frame_id)

        request = CorrectionRequest(
            frame_id=int(frame_id),
            timestamp=float(timestamp),
            image=image,
            candidates=tuple(detections),
            target_bbox=None if state.bbox is None else np.asarray(state.bbox, dtype=np.float32).copy(),
            predicted_bbox=None if predicted_bbox is None else np.asarray(predicted_bbox, dtype=np.float32),
            target_track_id=int(state.track_id),
            generation=int(state.generation),
            epoch=int(self._epoch),
            trigger=trigger,
        )

        # 队列仅保留最新请求
        try:
            self._queue.put_nowait(request)
        except Full:
            try:
                self._queue.get_nowait()   # 丢弃旧请求
                self.stats.dropped_queue_full += 1
                self._queue.put_nowait(request)
            except (Empty, Full):  # pragma: no cover - 竞态保护
                return None

        self.stats.submitted += 1
        if trigger == "interval":
            self.stats.periodic += 1
        else:
            self.stats.event_triggered += 1
        self._last_submit_t = now
        if trigger != "interval":
            self._last_event_t = now
        self._wakeup.set()
        return trigger

    def poll(self) -> Optional[CorrectionResult]:
        """非阻塞取回最新校正结果。"""
        latest: Optional[CorrectionResult] = None
        while True:
            try:
                latest = self._results.get_nowait()
            except Empty:
                break
        if latest is not None:
            self.stats.record_latency(latest.latency_s)
            self.stats.record_age(latest.age_s())
        return latest

    # -- 触发策略 -----------------------------------------------------------
    def _note_deferral(self, trigger: str, frame_id: int) -> None:
        """记录一次"周期节拍被推迟"。

        只在节拍的**名义帧**上计一次, 否则节拍在等待期间每帧都会重复计数,
        统计值会失去意义。
        """
        if trigger != "interval":
            return
        if frame_id == self._periodic_anchor + self.interval_frames:
            self.stats.deferred_periodic += 1

    def _select_trigger(
        self,
        frame_id: int,
        state: TargetState,
        crossing: bool,
        now: float,
    ) -> Optional[str]:
        """返回触发原因; None 表示本帧不触发。"""
        prev_score = self._prev_score
        prev_visible = self._prev_visible
        self._prev_score = state.score if state.visible else None
        self._prev_visible = state.visible

        if self.events_enabled:
            # 1) 目标重现: 由不可见转为可见
            if self.reid_check and state.visible and not prev_visible and frame_id > self.warmup_skip_frames + 1:
                return "reid"
            # 2) 置信度下降
            if (
                prev_score is not None
                and state.visible
                and state.score < self.conf_drop
                and prev_score >= self.conf_drop
            ):
                return "conf_drop"
            # 3) 鱼体交叉
            if self.cross_check and crossing:
                return "cross"
            # 4) 接近视场边缘
            if state.visible and state.bbox is not None:
                margin = edge_margin(np.asarray(state.bbox, dtype=np.float32), self.width, self.height)
                if margin < self.edge_margin_thresh:
                    return "edge"

        # 5) 周期性触发: 距上次周期提交已过 interval_frames 帧。
        #    被限流或在飞时该条件保持为真, 因此节拍会晚到但不会丢失。
        if frame_id - self._periodic_anchor >= self.interval_frames:
            return "interval"
        return None

    def _respect_rate_limit(self, trigger: str, now: float) -> bool:
        if now - self._last_submit_t < 1.0 / max(1e-6, self.max_request_hz):
            return False
        if trigger != "interval" and now - self._last_event_t < self.event_min_interval_s:
            return False
        return True

    # -- 工作线程 -----------------------------------------------------------
    def _worker(self) -> None:
        while not self._stop.is_set():
            self._wakeup.wait(timeout=0.2)
            self._wakeup.clear()
            if self._stop.is_set():
                break
            try:
                request = self._queue.get_nowait()
            except Empty:
                continue

            with self._lock:
                self._inflight = True
            try:
                result = self.corrector.correct(request, predicted_bbox=request.predicted_bbox)
                if request.epoch != self._epoch:
                    # 换序列了: 旧结果不能进入下一序列的关联/记忆
                    LOGGER.debug("丢弃跨序列校正结果: epoch %d != %d", request.epoch, self._epoch)
                    continue
                if result.latency_s > self.timeout_s:
                    # 方案 4.2 节: 慢链路超时则暂停校正
                    result.status = CorrectionStatus.TIMEOUT
                    result.message = (
                        f"latency {result.latency_s:.3f}s > timeout {self.timeout_s:.3f}s"
                    )
                    self.stats.timeouts += 1
                    self._register_overrun()
                else:
                    self.stats.completed += 1
                    self._consecutive_overruns = 0
                self._publish(result)
            except Exception as exc:  # noqa: BLE001 - 慢链路异常不得影响快速环
                self.stats.errors += 1
                LOGGER.exception("慢速环校正异常: %s", exc)
                self._register_overrun()
            finally:
                with self._lock:
                    self._inflight = False

    def _publish(self, result: CorrectionResult) -> None:
        try:
            self._results.put_nowait(result)
        except Full:  # pragma: no cover - 消费端落后时丢弃最旧结果
            try:
                self._results.get_nowait()
                self._results.put_nowait(result)
            except (Empty, Full):
                pass

    def _register_overrun(self) -> None:
        self._consecutive_overruns += 1
        if self._consecutive_overruns >= self.max_consecutive_overruns:
            self._paused_until = time.monotonic() + self.cooldown_s
            self._consecutive_overruns = 0
            LOGGER.warning(
                "慢链路连续超时/异常, 暂停校正 %.1fs。快速链路继续基线跟踪。",
                self.cooldown_s,
            )
