"""快慢双环主闭环。

数据流(方案第 1 节):

```text
图像 -> YOLO11n -> ByteTrack -> 目标状态 -> 图像伺服 -> 仲裁 -> 飞控
  |                                          ^
  +-- 每 10 个处理帧 / 事件触发 -> DINOv3 + 目标记忆
      -> 关联、漂移检查与重捕校正(异步, 快速环不等待)
```

关键约束:
1. 快速环每帧永不阻塞等待慢速环; 慢速环在独立线程中运行;
2. 慢速环的结果必须先过门控(年龄 + 目标一致性)才允许影响目标状态;
3. 慢速环只能影响目标状态(置信度/漂移/平滑位置修正), 不能直接产生控制量;
4. 慢链路异常时, 快速链路健康且目标可信则继续基线跟踪, 否则退出追近。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import Config
from ..control.arbiter import ControlArbiter
from ..control.servo import ImageServoController
from ..control.target_state import TargetTracker
from ..io.recorder import RunRecorder, draw_overlay
from ..io.video_source import FrameSource, build_source
from ..memory.template_bank import TemplateBank
from ..metrics import SessionMetrics
from ..perception.corrector import SlowCorrector
from ..perception.detector import build_detector
from ..perception.embedder import build_embedder
from ..perception.tracker import ByteTracker
from ..prediction import build_predictor
from ..scheduling.corrector_scheduler import SlowLoopScheduler
from ..types import CorrectionStatus, TargetState, Track

__all__ = ["TrackingPipeline", "PipelineResult"]

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """一次运行的结果。"""

    metrics: SessionMetrics
    summary: Dict[str, object]
    output_dir: Optional[Path] = None
    frames: int = 0
    elapsed_s: float = 0.0
    final_mode: Optional[str] = None
    events: List[Dict] = field(default_factory=list)

    @property
    def success(self) -> Dict[str, bool]:
        return self.metrics.check_success(self.summary)


class TrackingPipeline:
    """把检测、跟踪、校正、调度与控制装配成可运行的闭环。"""

    def __init__(
        self,
        cfg: Config,
        *,
        source: Optional[FrameSource] = None,
        run_name: Optional[str] = None,
        enable_recording: bool = True,
    ) -> None:
        self.cfg = cfg
        self.source = source or build_source(cfg.source)
        self.image_size = (int(self.source.width), int(self.source.height))
        self.width, self.height = self.image_size
        self.frame_id = 0

        LOGGER.info("图像尺寸: %dx%d @ %.1f fps", self.width, self.height, self.source.fps)

        # 仿真帧源支持把控制指令回灌, 形成闭环(见 pandeng.sim)
        self._closed_loop = callable(getattr(self.source, "apply_command", None))
        self._depth_source = self.source if hasattr(self.source, "depth_error_m") else None
        if self._closed_loop:
            LOGGER.info("检测到闭环仿真载体: 控制指令将回灌到场景")

        # --- 构建感知 -----------------------------------------------------
        self.detector = build_detector(cfg.detector, self.image_size, source=self.source)
        self.tracker = ByteTracker(cfg.tracker, self.image_size)

        # --- 构建慢速环 ---------------------------------------------------
        self.corrector_cfg = cfg.corrector
        self.bank = TemplateBank(
            max_templates=int(cfg.memory.max_templates),
            min_template_score=float(cfg.memory.min_template_score),
            update_momentum=float(cfg.memory.update_momentum),
            consistency_gate=float(cfg.memory.consistency_gate),
            freeze_on_occlusion=bool(cfg.memory.freeze_on_occlusion),
            max_freeze_s=float(cfg.memory.max_freeze_s),
            min_update_interval_s=float(cfg.memory.min_update_interval_s),
        )
        self.corrector_enabled = bool(
            cfg.corrector.get_path("enabled", True)
        )
        if self.corrector_enabled:
            self.embedder = build_embedder(cfg.corrector)
            self.corrector = SlowCorrector(
                cfg.corrector, self.embedder, self.bank, self.image_size
            )
            self.scheduler = SlowLoopScheduler(cfg.scheduler, self.corrector, self.image_size)
        else:
            self.embedder = None
            self.corrector = None
            self.scheduler = None

        # --- 构建控制 -----------------------------------------------------
        self.target_tracker = TargetTracker(
            cfg.target, cfg.controller, self.image_size
        )
        self.controller = ImageServoController(cfg.controller, self.image_size)
        self.arbiter = ControlArbiter(cfg.controller)

        # --- 时序预测(第二阶段; 首版 backend=none) -------------------------
        pred_cfg = cfg.get_path("prediction", None)
        self.predictor = build_predictor(pred_cfg) if pred_cfg is not None else None
        limits = pred_cfg.get_path("limits", {}) if pred_cfg is not None else {}
        self.pred_max_yaw_hint = float(
            limits.get("max_yaw_hint", 0.25) if hasattr(limits, "get") else 0.25
        )
        self.pred_min_surge_scale = float(
            limits.get("min_surge_scale", 0.2) if hasattr(limits, "get") else 0.2
        )
        if self.predictor is not None:
            LOGGER.info("时序预测已启用: %s (horizon=%.2fs)", self.predictor.name, self.predictor.horizon_s)

        # --- 记录 ---------------------------------------------------------
        self.run_name = run_name or time.strftime("%Y%m%d_%H%M%S")
        rec = cfg.recorder
        out_root = rec.get_path("dir", None) or cfg.project.get_path("output_dir", "runs")
        self.recorder = RunRecorder(
            out_root,
            name=self.run_name,
            save_video=bool(rec.save_video),
            save_csv=bool(rec.save_csv),
            save_jsonl=bool(rec.save_jsonl),
            draw=bool(rec.draw),
            fps=float(self.source.fps),
            enabled=bool(rec.enabled) and enable_recording,
        )

        self.metrics = SessionMetrics(name=self.run_name)
        self._generation_seen = 0
        self._final_mode: Optional[str] = None
        self._stop_requested = False

        # --- 多序列支持 ---------------------------------------------------
        # `SequenceDirSource` 把每个视频当独立序列; 进新序列时必须重置全部
        # 时序状态, 否则轨迹 ID、目标代次与外观模板会跨视频泄漏。
        self._sequence_id: Optional[str] = None
        self._sequence_metrics: List[SessionMetrics] = []
        self._sequence_names: List[str] = []
        self._epoch = 0
        eval_cfg = cfg.get_path("evaluation", None)
        self._eval_iou = float(
            eval_cfg.get_path("iou_thresh", 0.5) if eval_cfg is not None else 0.5
        )
        self._eval_per_class = bool(
            eval_cfg.get_path("per_class", True) if eval_cfg is not None else True
        )
        # 帧源是否分序列(决定是否需要逐序列汇总与逐序列落盘)
        self._multi_sequence = hasattr(self.source, "num_sequences")

    # -- 生命周期 -----------------------------------------------------------
    def warmup(self) -> None:
        """用一张合成图预热检测器与特征提取器, 避免首帧计入时延统计。"""
        dummy = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        times = int(self.cfg.detector.get_path("warmup", 3))
        if times > 0:
            try:
                self.detector.warmup(dummy, times)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("检测器预热失败: %s", exc)
        if self.embedder is not None:
            try:
                self.embedder.embed(
                    dummy, [np.array([0.0, 0.0, 64.0, 64.0], dtype=np.float32)]
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("特征提取器预热失败: %s", exc)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.stop()
        self.recorder.close()
        if self.embedder is not None:
            self.embedder.close()
        self.detector.close()
        self.source.close()

    def request_stop(self) -> None:
        """供信号处理调用, 让主循环优雅退出。"""
        self._stop_requested = True

    # -- 主循环 -------------------------------------------------------------
    def run(self, max_frames: Optional[int] = None) -> PipelineResult:
        if self.scheduler is not None:
            self.scheduler.start()
        self.warmup()

        # 首轮固定深度: 以进场深度作为保持目标(方案第 3 节)。
        # 真实艇上该设定点由飞控/声纳给出, 本版不做深度闭环。
        if self._depth_source is not None:
            self._depth_source.set_depth_setpoint(None)

        started = time.monotonic()
        processed = 0
        try:
            while not self._stop_requested:
                if max_frames is not None and processed >= int(max_frames):
                    break
                item = self.source.read()
                if item is None:
                    break
                frame, capture_t = item
                self._maybe_start_sequence()
                self._process_frame(frame, capture_t)
                processed += 1
        except KeyboardInterrupt:  # pragma: no cover
            LOGGER.warning("收到中断信号, 正在退出")
        finally:
            elapsed = time.monotonic() - started
            if self.scheduler is not None:
                self.scheduler.stop()

        self._flush_sequence_metrics()

        scheduler_stats = (
            self.scheduler.stats.summary(elapsed) if self.scheduler is not None else None
        )
        # 多序列运行必须先把各序列合并再出汇总: 直接用 self.metrics 只会剩下
        # 最后一个视频的统计, 前面几十个序列的数据全部丢掉。
        if self._multi_sequence and self._sequence_metrics:
            aggregate = SessionMetrics(name=self.run_name)
            for item in self._sequence_metrics:
                aggregate.merge(item)
            summary = aggregate.summary(scheduler_stats)
        else:
            summary = self.metrics.summary(scheduler_stats)
        summary["elapsed_s"] = round(elapsed, 3)
        summary["processed_frames"] = processed
        summary["corrector_backend"] = str(
            self.cfg.corrector.get_path("backend", "disabled")
        ) if self.corrector_enabled else "disabled"
        summary["detector_backend"] = str(self.cfg.detector.get_path("backend", "ultralytics"))
        # 记录实际使用的外观骨干。若发生回退(is_fallback=True), 实验组 B/C/D/E
        # 的"关联改善"结论不可用于验收, 必须在汇总里一眼可见。
        if self.embedder is not None:
            summary["embedder_backend"] = self.embedder.name
            summary["embedder_backbone"] = getattr(self.embedder, "backbone_source", "n/a")
            summary["embedder_is_fallback"] = bool(
                getattr(self.embedder, "is_fallback", False)
            )
            summary["embedder_dim"] = int(getattr(self.embedder, "dim", 0))
        else:
            summary["embedder_backend"] = "disabled"
            summary["embedder_backbone"] = None
            summary["embedder_is_fallback"] = None
            summary["embedder_dim"] = None
        summary["final_mode"] = self._final_mode
        summary["template_count"] = len(self.bank)
        # 未做实时限速的回放不具备实时性参考价值, 单独标注
        summary["paced"] = (
            bool(getattr(self.source, "realtime_pacing", False))
            if hasattr(self.source, "realtime_pacing")
            else None
        )
        # 回放数据集时舵机不会真的动, 目标位置只反映视频本身的构图。
        # 此时"居中率"衡量的是素材而不是控制器, 判据不适用, 必须一眼可辨。
        summary["closed_loop"] = bool(self._closed_loop)
        summary["source_type"] = str(self.cfg.source.get_path("type", "video"))
        if self._multi_sequence and self._sequence_metrics:
            summary["num_sequences"] = len(self._sequence_metrics)
            summary["sequences"] = [item.summary(None) for item in self._sequence_metrics]
            summary["sequence_names"] = list(self._sequence_names)

        self.recorder.log_event("summary", **summary)
        self.recorder.close()

        # summary.json 与逐帧记录解耦: 即使关闭了视频/CSV 记录, 一次运行的
        # 指标汇总也必须落盘, 否则批量实验无法汇总(实验组 A~E 依赖该文件)。
        out_dir = self.recorder.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics.dump(out_dir / "summary.json", summary=summary)

        result = PipelineResult(
            metrics=self.metrics,
            summary=summary,
            output_dir=out_dir,
            frames=processed,
            elapsed_s=elapsed,
            final_mode=self._final_mode,
            events=list(self.recorder._events),
        )
        return result

    # -- 序列切换 -----------------------------------------------------------
    def _maybe_start_sequence(self) -> None:
        """检测帧源是否进入新序列, 是则重置全部时序状态。

        判定依据只有 `sequence_id` 变化这一条 —— 不依赖帧索引或帧数, 这样
        即使某个视频被 `max_frames` 截断也不影响正确性。
        """
        current = getattr(self.source, "sequence_id", None)
        if current == self._sequence_id:
            return
        index = int(getattr(self.source, "sequence_index", -1))
        self._begin_sequence(current, index)

    def _begin_sequence(self, sequence_id: Optional[str], index: int) -> None:
        # 首帧前不产生"切换"事件, 避免批量运行为每个序列凭空记一次
        if self._sequence_id is not None:
            self._flush_sequence_metrics()
            self._epoch += 1
            LOGGER.info("进入新序列 [%d]: %s", index + 1, sequence_id)

        self._sequence_id = sequence_id
        self.frame_id = 0
        self._generation_seen = 0
        self._final_mode = None

        # 跟踪与目标状态
        self.tracker.reset()
        self.target_tracker.reset()
        # 外观记忆: 代次变更即清空模板, 新视频第一帧不能匹配上一视频的模板
        self.bank.on_generation_change()
        # 控制与仲裁
        self.controller.reset()
        self.arbiter.reset()
        if self.predictor is not None:
            self.predictor.reset()
        # 调度器: 丢弃排队请求与在飞结果
        if self.scheduler is not None:
            self.scheduler.begin_sequence(self._epoch)

        self.metrics = SessionMetrics(name=sequence_id or self.run_name)
        self.metrics.detection.iou_thresh = self._eval_iou
        self.metrics.detection.per_class = self._eval_per_class
        self.recorder.start_sequence(sequence_id, index)
        if sequence_id is not None:
            self.recorder.log_event(
                "sequence_begin", sequence=sequence_id, index=index, epoch=self._epoch
            )

    def _flush_sequence_metrics(self) -> None:
        """把当前序列的指标存档; 空序列(被 max_frames=0 截断)不入档。"""
        if self.metrics.frames <= 0:
            return
        self.metrics.name = self._sequence_id or self.run_name
        self._sequence_metrics.append(self.metrics)
        self._sequence_names.append(self.metrics.name)

    # -- 单帧 ---------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray, capture_t: float) -> None:
        self.frame_id += 1
        now = time.monotonic()
        self.arbiter.report_perception(now)
        self.controller.notify_perception(now)

        # 1) 快速检测 + 跟踪
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame_id=self.frame_id)

        # 2) 目标状态
        state = self.target_tracker.update(
            frame_id=self.frame_id,
            timestamp=capture_t,
            tracks=tracks,
            detections=detections,
            now=now,
        )

        # 3) 目标代次变化: 清空跨个体记忆, 避免模板污染
        generation = self.target_tracker.selector.generation
        if generation != self._generation_seen:
            previous = self._generation_seen
            self._generation_seen = generation
            self.bank.on_generation_change()
            self.target_tracker.notify_target_reset()
            # 首次捕获目标(0 -> 1)不计入"误切目标", 否则每次试验都会
            # 凭空多出一次切换, 让 A~E 各组的切换指标失去区分度。
            if previous > 0:
                self.metrics.record_target_switch()
            self.recorder.log_event(
                "generation_change",
                frame_id=self.frame_id,
                generation=generation,
                previous_generation=previous,
                track_id=self.target_tracker.selector.track_id,
                is_initial=previous == 0,
            )

        # 4) 消费慢速环结果(非阻塞)
        correction_applied = False
        correction_status = ""
        correction_age: Optional[float] = None
        correction_latency: Optional[float] = None
        correction_score: Optional[float] = None
        gate_reason = ""
        template_frozen = self.bank.is_frozen()

        if self.scheduler is not None:
            result = self.scheduler.poll()
            if result is not None:
                decision = self.target_tracker.apply_correction(result, now=now)
                correction_applied = bool(decision.apply)
                correction_status = result.status.value
                correction_age = result.age_s(now)
                correction_latency = result.latency_s
                correction_score = result.drift_score
                gate_reason = decision.reason
                self.metrics.record_correction(
                    applied=correction_applied,
                    drift=result.status is CorrectionStatus.DRIFT,
                    template_frozen=template_frozen,
                )
                if result.status is CorrectionStatus.REDETECTED:
                    self.metrics.record_recovery()
                elif result.status in {CorrectionStatus.TIMEOUT, CorrectionStatus.OVERLOADED}:
                    self.metrics.record_slow_fault()
                self.recorder.log_event(
                    "correction",
                    frame_id=self.frame_id,
                    req_frame=result.frame_id,
                    status=result.status.value,
                    applied=correction_applied,
                    drift_score=round(result.drift_score, 4),
                    confidence=round(result.confidence, 4),
                    latency_s=round(result.latency_s, 5),
                    age_s=round(result.age_s(now), 5),
                    gate=decision.reason,
                    target_bbox=None if result.matched_bbox is None else result.matched_bbox.tolist(),
                )
                # 状态可能被校正如置信度/漂移, 重新取用
                state = self.target_tracker.state

        # 5) 提交新的校正请求(异步, 立即返回)
        #    一并给出当前目标的运动预测框, 供慢环做运动连续性打分
        if self.scheduler is not None and state is not None:
            self.scheduler.maybe_submit(
                frame_id=self.frame_id,
                timestamp=capture_t,
                state=state,
                detections=detections,
                image=frame,
                crossing=bool(self._crossing_hint(tracks, state)),
                predicted_bbox=self._predicted_bbox(tracks, state),
            )

        # 6) 时序预测(只有可见时才记录观测; 未来信息不进入输入)
        prediction = None
        if self.predictor is not None:
            self.predictor.observe(
                timestamp=now,
                ex=state.ex if state is not None else 0.0,
                ey=state.ey if state is not None else 0.0,
                visible=bool(state is not None and state.visible),
            )
            prediction = self.predictor.predict()
            if not prediction.valid:
                prediction = None
            else:
                # 受限修正: 在交给控制器前先按配置限幅
                prediction.yaw_hint = float(
                    np.clip(prediction.yaw_hint, -self.pred_max_yaw_hint, self.pred_max_yaw_hint)
                )
                prediction.surge_scale = float(
                    np.clip(prediction.surge_scale, self.pred_min_surge_scale, 1.0)
                )

        # 7) 控制
        command = self.controller.update(state, now=now, prediction=prediction)
        arbitration = self.arbiter.arbitrate(command, state)
        final = arbitration.command
        self._final_mode = final.mode.value

        # 仿真载体: 把仲裁后的指令回灌到场景, 形成闭环(仅在 simulated 帧源下生效)
        if self._closed_loop:
            self.source.apply_command(final)

        bridge_latency = time.monotonic() - capture_t

        # 8) 检测评估: 只对有标注的帧计分(无标注帧≠无目标)
        reference = (
            self.source.ground_truth()
            if hasattr(self.source, "ground_truth")
            else None
        )
        self.metrics.detection.update(detections, reference)

        # 9) 指标
        visible = state is not None and state.visible
        in_center = (
            SessionMetrics.is_in_central_region(state.bbox, self.width, self.height)
            if visible and state is not None
            else False
        )
        self.metrics.record_frame(
            now=now,
            visible=visible,
            in_central_region=in_center,
            bridge_latency=bridge_latency,
            depth_error=self._depth_source.depth_error_m if self._depth_source is not None else None,
        )

        # 9) 记录
        self._record(
            state=state,
            detections=detections,
            tracks=tracks,
            command=final,
            correction_status=correction_status,
            correction_applied=correction_applied,
            correction_age=correction_age,
            correction_latency=correction_latency,
            correction_score=correction_score,
            gate_reason=gate_reason,
            bridge_latency=bridge_latency,
            template_frozen=template_frozen,
        )

        # 10) 可视化
        if self.recorder.enabled and self.recorder.save_video and self.recorder.draw:
            health = self.arbiter.health
            overlay = draw_overlay(
                frame.copy(),
                state=state,
                tracks=tracks,
                detections=detections,
                command=final,
                scheduler=self.scheduler,
                extra={
                    "fps": f"{1.0 / max(1e-6, bridge_latency):.1f}",
                    "bridge_ms": f"{bridge_latency * 1000.0:.1f}",
                    "health": arbitration.health.value,
                },
            )
            self.recorder.write_frame(overlay)

    # -- 辅助 ---------------------------------------------------------------
    @staticmethod
    def _predicted_bbox(
        tracks: Sequence[Track], state: TargetState
    ) -> Optional[np.ndarray]:
        """取当前目标轨迹的运动预测框; 找不到(如鱼群虚拟轨迹)时返回 None。"""
        for track in tracks:
            if track.track_id == state.track_id:
                return track.predicted_bbox(1.0)
        return None

    def _crossing_hint(self, tracks, state: Optional[TargetState]) -> bool:
        """给调度器的"鱼体交叉"提示: 存在与目标框显著重叠的其他轨迹。"""
        if state is None or len(tracks) < 2:
            return False
        from ..utils.geometry import iou_matrix

        boxes = np.stack([np.asarray(t.xyxy, dtype=np.float32) for t in tracks])
        ious = iou_matrix(boxes, np.asarray(state.bbox, dtype=np.float32)[None, :]).ravel()
        return sum(1 for v in ious if v > 0.35) >= 2

    def _record(self, **kwargs) -> None:
        state: Optional[TargetState] = kwargs["state"]
        command = kwargs["command"]
        rows = {
            "sequence": self._sequence_id or "",
            "frame_id": self.frame_id,
            "t": kwargs["bridge_latency"],
            "visible": state.visible if state is not None else False,
            "track_id": state.track_id if state is not None else -1,
            "generation": state.generation if state is not None else -1,
            "x1": state.bbox[0] if state is not None else 0.0,
            "y1": state.bbox[1] if state is not None else 0.0,
            "x2": state.bbox[2] if state is not None else 0.0,
            "y2": state.bbox[3] if state is not None else 0.0,
            "ex": state.ex if state is not None else 0.0,
            "ey": state.ey if state is not None else 0.0,
            "area_ratio": state.area_ratio if state is not None else 0.0,
            "area_growth": state.area_growth if state is not None else 0.0,
            "det_score": state.score if state is not None else 0.0,
            "assoc_conf": state.association_confidence if state is not None else 0.0,
            "drift": state.drift_flag if state is not None else False,
            "recovered": state.recovered if state is not None else False,
            "group_size": state.group_size if state is not None else 0,
            "mode": command.mode.value,
            "surge": command.surge,
            "yaw": command.yaw,
            "heave": command.heave,
            "reason": command.reason,
            "n_detections": len(kwargs["detections"]),
            "n_tracks": len(kwargs["tracks"]),
            "correction_status": kwargs["correction_status"],
            "correction_applied": kwargs["correction_applied"],
            "correction_age_s": kwargs["correction_age"],
            "correction_latency_s": kwargs["correction_latency"],
            "correction_score": kwargs["correction_score"],
            "gate_reason": kwargs["gate_reason"],
            "bridge_latency_ms": kwargs["bridge_latency"] * 1000.0,
            "n_templates": len(self.bank),
            "templates_frozen": kwargs["template_frozen"],
        }
        self.recorder.log_frame(rows)
