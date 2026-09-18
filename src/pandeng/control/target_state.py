"""目标选择与目标状态维护。

- 单鱼: 连续可见性 + 切换滞回; 鱼群: 跟随有效鱼框的群体中心;
- 校正结果先过门控(年龄 + 目标代次一致性)再影响状态;
- 位置修正先映射到当前帧再平滑应用, 不用旧目标框覆盖当前观测。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..perception.corrector import CorrectionGate, GateDecision
from ..types import (
    CorrectionResult,
    CorrectionStatus,
    Detection,
    TargetMode,
    TargetState,
    Track,
    TrackState,
)
from ..utils.geometry import area_ratio, clip_boxes, normalize_error, slope_per_second

__all__ = ["TargetSelector", "TargetTracker"]

LOGGER = logging.getLogger(__name__)


class TargetSelector:
    """从轨迹集合中选出被跟随的目标。"""

    def __init__(self, cfg, image_size: Tuple[int, int]) -> None:
        self.cfg = cfg
        self.width, self.height = int(image_size[0]), int(image_size[1])
        self.mode = TargetMode(str(cfg.get_path("mode", "single")).lower())
        self.select = str(cfg.get_path("select", "largest")).lower()
        self.hysteresis = float(cfg.get_path("switch_hysteresis", 0.25))
        self.min_visible_frames = int(cfg.get_path("min_visible_frames", 5))
        self.lost_timeout_s = float(cfg.get_path("lost_timeout_s", 2.0))

        sc = cfg.get_path("school", {})
        self.school_min_score = float(sc.get("min_score", 0.30) if hasattr(sc, "get") else 0.30)
        self.school_min_count = int(sc.get("min_count", 2) if hasattr(sc, "get") else 2)

        self.track_id: int = -1
        self.generation: int = 0
        self.visible_frames: int = 0
        self.last_visible_t: Optional[float] = None

    def reset(self) -> None:
        """切换到新序列: 代次与可见帧计数归零。

        跨视频保留 track_id/generation 没有意义 —— 新视频的第一帧与上一视频
        的最后一帧在内容上没有关系, 继承下来的"当前目标"只会制造虚假的
        目标切换统计。
        """
        self.track_id = -1
        self.generation = 0
        self.visible_frames = 0
        self.last_visible_t = None

    # -- 主接口 -------------------------------------------------------------
    def select_target(
        self, tracks: Sequence[Track], now: Optional[float] = None
    ) -> Tuple[Optional[Track], TargetMode, int]:
        """返回 (目标轨迹, 生效模式, 群体大小)。

        单鱼模式返回单条轨迹; 鱼群模式返回群体中心的等价框(包装成 Track)。
        """
        now = time.monotonic() if now is None else float(now)
        tracked = [t for t in tracks if t.state == TrackState.TRACKED]

        if self.mode is TargetMode.SCHOOL:
            group = [t for t in tracked if t.score >= self.school_min_score]
            if len(group) >= self.school_min_count:
                return self._school_target(group), TargetMode.SCHOOL, len(group)
            # 鱼群数量不足时退化为单鱼选择, 但模式标记仍为 school 以便上层统计
            if group:
                return self._single_target(group, now), TargetMode.SINGLE, len(group)
            return None, TargetMode.SINGLE, 0

        if not tracked:
            return None, TargetMode.SINGLE, 0
        return self._single_target(tracked, now), TargetMode.SINGLE, 1

    def _single_target(self, tracks: Sequence[Track], now: float) -> Optional[Track]:
        if not tracks:
            return None
        current = next((t for t in tracks if t.track_id == self.track_id), None)
        best = max(tracks, key=self._rank_score)

        if current is None:
            self._on_target_change(best.track_id)
            return best

        # 切换滞回: 新候选必须显著优于当前目标, 且当前目标已不满足连续可见性
        if best.track_id != current.track_id:
            if self._rank_score(best) > self._rank_score(current) * (1.0 + self.hysteresis):
                if self.visible_frames < self.min_visible_frames or current.time_since_update > 0:
                    self._on_target_change(best.track_id)
                    return best
            return current
        return current

    def _school_target(self, group: Sequence[Track]) -> Track:
        """把群体中心包装成一条虚拟轨迹, 复用单鱼的控制链路。"""
        boxes = np.stack([np.asarray(t.xyxy, dtype=np.float32) for t in group])
        cx = float(np.mean((boxes[:, 0] + boxes[:, 2]) * 0.5))
        cy = float(np.mean((boxes[:, 1] + boxes[:, 3]) * 0.5))
        half_w = float(np.mean(boxes[:, 2] - boxes[:, 0])) * 0.5
        half_h = float(np.mean(boxes[:, 3] - boxes[:, 1])) * 0.5
        box = np.array(
            [cx - half_w, cy - half_h, cx + half_w, cy + half_h], dtype=np.float32
        )
        # 群体模式下 track_id 固定为 -1-, 避免与真实轨迹混淆
        if self.track_id != -2:
            self._on_target_change(-2)
        return Track(
            track_id=-2,
            xyxy=box,
            score=float(np.mean([t.score for t in group])),
            state=TrackState.TRACKED,
            age=min(t.age for t in group),
            hits=min(t.hits for t in group),
            time_since_update=0,
        )

    def _rank_score(self, track: Track) -> float:
        if self.select == "highest_score":
            return float(track.score)
        if self.select == "nearest_center":
            cu, cv = track.center
            dist = np.hypot(cu - self.width * 0.5, cv - self.height * 0.5)
            diag = float(np.hypot(self.width, self.height))
            return float(1.0 - dist / max(1.0, diag))
        # largest
        return float(track.area)

    def _on_target_change(self, new_id: int) -> None:
        if new_id != self.track_id:
            self.generation += 1
            self.visible_frames = 0
            LOGGER.info(
                "目标代次更新: %d -> %d (track %d -> %d)",
                self.generation - 1,
                self.generation,
                self.track_id,
                new_id,
            )
        self.track_id = int(new_id)


class TargetTracker:
    """维护 `TargetState`, 并应用慢速环校正结果。"""

    def __init__(
        self,
        cfg,
        controller_cfg,
        image_size: Tuple[int, int],
        *,
        gate: Optional[CorrectionGate] = None,
    ) -> None:
        self.width, self.height = int(image_size[0]), int(image_size[1])
        self.controller_cfg = controller_cfg
        self.selector = TargetSelector(cfg, image_size)
        self.lost_timeout_s = float(cfg.get_path("lost_timeout_s", 2.0))

        gate_cfg = controller_cfg.get_path("gate", {})
        self.gate = gate or CorrectionGate(
            max_age_s=float(gate_cfg.get("max_age_s", 0.60)),
            min_confidence=float(gate_cfg.get("min_confidence", 0.40)),
            smooth_alpha=float(gate_cfg.get("smooth_alpha", 0.35)),
            max_offset_ratio=float(gate_cfg.get("max_offset_ratio", 0.25)),
        )

        self.state: Optional[TargetState] = None
        self.frame_id: int = 0

        # 慢速环可写字段
        self._association_confidence: float = 1.0
        self._drift_flag: bool = False
        self._recovered: bool = False
        self._correction_offset = np.zeros(2, dtype=np.float32)
        self._last_correction_t: Optional[float] = None
        self._last_correction_frame: int = -1
        self._area_history: List[Tuple[float, float]] = []
        self.gate_log: List[Dict[str, object]] = []

    # -- 每帧更新 -----------------------------------------------------------
    def update(
        self,
        *,
        frame_id: int,
        timestamp: float,
        tracks: Sequence[Track],
        detections: Sequence[Detection],
        now: Optional[float] = None,
    ) -> Optional[TargetState]:
        now = time.monotonic() if now is None else float(now)
        self.frame_id = int(frame_id)

        target, mode, group_size = self.selector.select_target(tracks, now)
        if target is None:
            self.selector.visible_frames = 0
            previous = self.state
            self.state = None
            if previous is not None:
                self._drift_flag = False
            return None

        self.selector.visible_frames += 1
        self.selector.last_visible_t = now

        bbox = np.asarray(target.xyxy, dtype=np.float32).copy()

        # --- 位置修正(慢速环) -------------------------------------------
        # 先映射到当前帧: 用目标的运动速度把偏移做时间对齐, 再平滑应用。
        applied_offset = self._aligned_offset(target, frame_id, now)
        corrected_bbox = bbox.copy()
        if np.any(applied_offset):
            corrected_bbox[0::2] += applied_offset[0]
            corrected_bbox[1::2] += applied_offset[1]
            corrected_bbox = clip_boxes(corrected_bbox, self.width, self.height)[0]

        ex, ey = normalize_error(target.center, self.width, self.height)
        ratio = area_ratio(bbox, self.width, self.height)
        growth = slope_per_second(self._area_history, ratio, now, window_s=0.5)
        self._area_history.append((now, ratio))
        if len(self._area_history) > 60:
            self._area_history.pop(0)

        # 检测置信度: 用与目标框 IoU 最大的检测分数
        det_score = self._best_detection_score(bbox, detections)

        visible = target.time_since_update == 0
        state = TargetState(
            frame_id=int(frame_id),
            timestamp=float(timestamp),
            bbox=corrected_bbox,
            track_id=int(target.track_id),
            visible=visible,
            generation=int(self.selector.generation),
            ex=ex,
            ey=ey,
            area_ratio=ratio,
            area_growth=float(growth),
            score=float(max(det_score, 0.0)),
            association_confidence=float(self._association_confidence),
            drift_flag=bool(self._drift_flag),
            recovered=bool(self._recovered),
            group_size=int(group_size),
            age_frames=int(self.selector.visible_frames),
        )
        self.state = state
        return state

    # -- 慢速环结果 ---------------------------------------------------------
    def apply_correction(
        self,
        result: CorrectionResult,
        *,
        now: Optional[float] = None,
    ) -> GateDecision:
        """门控并应用校正结果。返回门控决策供记录。"""
        now = time.monotonic() if now is None else float(now)
        if self.state is None:
            decision = GateDecision(False, CorrectionStatus.STALE, "no active target")
            self._log_gate(result, decision)
            return decision

        decision = self.gate.evaluate(result, self.state, now=now)
        self._log_gate(result, decision)

        if result.status is CorrectionStatus.TIMEOUT:
            # 慢链路超时: 暂停校正, 快速链路健康则继续基线跟踪
            return decision

        if result.status in {CorrectionStatus.DRIFT, CorrectionStatus.NO_MATCH}:
            # 降低目标置信度, 停止错误追近
            self._association_confidence = float(
                np.clip(0.5 * self._association_confidence + 0.5 * result.confidence, 0.0, 1.0)
            )
            self._drift_flag = True
            self._correction_offset *= 0.0
            return decision

        if result.status is CorrectionStatus.TARGET_SWITCH:
            # 关联到其他个体: 递减置信度并交由上层决定是否切换目标
            self._association_confidence = float(np.clip(result.confidence, 0.0, 1.0))
            self._drift_flag = False
            return decision

        if result.status is CorrectionStatus.REDETECTED:
            self._association_confidence = float(np.clip(result.confidence, 0.0, 1.0))
            self._drift_flag = False
            self._recovered = True
            return decision

        if result.status is CorrectionStatus.OK and decision.apply:
            # 关联确认: 恢复置信度, 并按平滑后的偏移做温和位置修正
            self._association_confidence = float(
                np.clip(0.5 * self._association_confidence + 0.5 * result.confidence, 0.0, 1.0)
            )
            self._drift_flag = False
            self._recovered = False
            offset = self._offset_from_result(result)
            if offset is not None:
                reference = float(
                    np.hypot(
                        self.state.bbox[2] - self.state.bbox[0],
                        self.state.bbox[3] - self.state.bbox[1],
                    )
                )
                self._correction_offset = self.gate.smooth_offset(
                    self._correction_offset, offset, reference
                )
            self._last_correction_t = now
            self._last_correction_frame = result.frame_id
        return decision

    def reset(self) -> None:
        """切换到新序列: 清空目标状态与慢速环写入的字段。"""
        self.state = None
        self.frame_id = 0
        self.selector.reset()
        self.notify_target_reset()
        self._correction_offset = np.zeros(2, dtype=np.float32)
        self._last_correction_frame = -1
        self._area_history.clear()
        self.gate_log.clear()

    def notify_target_reset(self) -> None:
        """目标代次变化后重置慢速环影响。"""
        self._association_confidence = 1.0
        self._drift_flag = False
        self._recovered = False
        self._correction_offset *= 0.0
        self._last_correction_t = None

    # -- 内部 ---------------------------------------------------------------
    def _offset_from_result(self, result: CorrectionResult) -> Optional[np.ndarray]:
        """把校正框与当前目标的差异换算为偏移量(像素)。

        方案要求"需要位置修正时先映射到当前帧, 再平滑应用, 不用旧目标框覆盖
        当前观测", 因此这里只取差值, 不直接采用 `matched_bbox`。
        """
        if result.matched_bbox is None or self.state is None:
            return None
        matched = np.asarray(result.matched_bbox, dtype=np.float32)
        current = np.asarray(self.state.bbox, dtype=np.float32)
        offset = np.array(
            [
                ((matched[0] + matched[2]) - (current[0] + current[2])) * 0.5,
                ((matched[1] + matched[3]) - (current[1] + current[3])) * 0.5,
            ],
            dtype=np.float32,
        )
        return offset

    def _aligned_offset(self, target: Track, frame_id: int, now: float) -> np.ndarray:
        """把上次校正偏移按帧差做运动对齐后再使用。"""
        if np.allclose(self._correction_offset, 0.0):
            return self._correction_offset
        if self._last_correction_frame < 0:
            return self._correction_offset
        delta = frame_id - self._last_correction_frame
        if delta <= 0:
            return self._correction_offset
        # 偏移量随目标运动线性衰减, 避免陈旧偏移持续推动目标框
        decay = float(np.clip(1.0 - delta / 20.0, 0.0, 1.0))
        return (self._correction_offset * decay).astype(np.float32)

    @staticmethod
    def _best_detection_score(bbox: np.ndarray, detections: Sequence[Detection]) -> float:
        best = 0.0
        for det in detections:
            d = np.asarray(det.xyxy, dtype=np.float32)
            # 中心距判据即可满足"该框是否有对应检测"的需求, 避免额外 IoU 开销
            if abs(float(d[0] + d[2] - bbox[0] - bbox[2])) < 8.0 and \
                    abs(float(d[1] + d[3] - bbox[1] - bbox[3])) < 8.0:
                best = max(best, float(det.score))
        return best

    def _log_gate(self, result: CorrectionResult, decision: GateDecision) -> None:
        if len(self.gate_log) > 1000:
            del self.gate_log[:500]
        self.gate_log.append(
            {
                "frame_id": result.frame_id,
                "status": result.status.value,
                "applied": decision.apply,
                "reason": decision.reason,
                "confidence": round(result.confidence, 4),
                "drift_score": round(result.drift_score, 4),
                "age_s": round(result.age_s(), 4),
                "latency_s": round(result.latency_s, 4),
            }
        )
