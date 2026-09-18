"""低频校正: DINOv3 冻结骨干 + 目标记忆。

对应方案第 4 节。本模块**只做关联判断**, 不做控制:
- 输入: 一帧原图、有限数量候选框、快速环当前目标框与目标代次;
- 输出: `CorrectionResult`(关联置信度、漂移判定、重捕建议、模板更新结果)。

方案第 4.2 节的硬性约束在此实现:
* 快速链路不等待校正结果 —— 本模块在独立线程中被调用, 无阻塞等待;
* 结果携带原图时间、帧号、目标代次和置信度 —— 由 `CorrectionResult` 保证;
* 门控检查数据年龄与目标一致性 —— 由 `CorrectionGate` 在快速环侧执行;
* 不用旧目标框覆盖当前观测 —— 本模块只输出 `matched_bbox` 与置信度,
  位置修正在快速环内做运动补偿后平滑应用。
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..memory.template_bank import TemplateBank
from ..types import (
    CorrectionRequest,
    CorrectionResult,
    CorrectionStatus,
    Detection,
    TargetState,
)
from ..utils.geometry import clip_boxes, expand_box, iou_matrix
from .association import rank_candidates
from .embedder import BaseEmbedder

__all__ = ["SlowCorrector", "CorrectionGate", "GateDecision"]

LOGGER = logging.getLogger(__name__)


class SlowCorrector:
    """低频关联校正器。"""

    def __init__(
        self,
        cfg,
        embedder: BaseEmbedder,
        bank: TemplateBank,
        image_size: Tuple[int, int],
    ) -> None:
        """
        Args:
            cfg: 配置的 `corrector` 段。
            embedder: 外观特征提取器(DINOv3 冻结骨干)。
            bank: 目标记忆。
            image_size: (width, height)
        """
        self.cfg = cfg
        self.enabled = bool(cfg.get_path("enabled", True))
        self.embedder = embedder
        self.bank = bank
        self.width, self.height = int(image_size[0]), int(image_size[1])

        self.max_candidates = int(cfg.get_path("max_candidates", 8))
        self.crop_expand = float(cfg.get_path("crop_expand", 1.25))
        self.accept_thresh = float(cfg.get_path("accept_thresh", 0.62))
        self.reid_thresh = float(cfg.get_path("reid_thresh", 0.55))
        self.drift_thresh = float(cfg.get_path("drift_thresh", 0.45))
        self.switch_margin = float(cfg.get_path("switch_margin", 0.12))
        self.motion_scale = float(cfg.get_path("motion_scale", 1.5))
        self.cross_penalty = float(cfg.get_path("cross_penalty", 0.15))

        w = cfg.get_path("score_weights", {})
        self.weights = {
            "appearance": float(w.get("appearance", 0.60) if hasattr(w, "get") else 0.60),
            "position": float(w.get("position", 0.25) if hasattr(w, "get") else 0.25),
            "motion": float(w.get("motion", 0.15) if hasattr(w, "get") else 0.15),
        }

    # -- 主接口 -------------------------------------------------------------
    def correct(
        self,
        request: CorrectionRequest,
        *,
        predicted_bbox: Optional[np.ndarray] = None,
    ) -> CorrectionResult:
        """执行一次低频校正。

        Args:
            request: 校正请求(含原图与候选)。
            predicted_bbox: 快速环对当前帧的运动预测框, 用于运动连续性打分。
        """
        started = time.monotonic()
        if not self.enabled:
            return self._result(
                request, CorrectionStatus.DISABLED, started, message="corrector disabled"
            )

        image = request.image
        if image is None or getattr(image, "size", 0) == 0:
            return self._result(
                request, CorrectionStatus.NO_MATCH, started, message="empty image"
            )

        candidates = self._select_candidates(request)
        if not candidates:
            # 视野内没有候选: 不写模板, 交回快速环做搜索/退出判定
            return self._result(
                request, CorrectionStatus.NO_MATCH, started, message="no candidates"
            )

        boxes = np.stack([np.asarray(c.xyxy, dtype=np.float32) for c in candidates])
        feats = self._embed(image, boxes)

        current_bbox = self._current_bbox(candidates, request.target_bbox)
        crossing = self._detect_crossing(boxes, current_bbox)

        ranked = rank_candidates(
            boxes,
            candidate_features=[feats[i] for i in range(len(boxes))],
            template_features=[t.feature for t in self.bank.templates],
            predicted_bbox=predicted_bbox,
            current_bbox=current_bbox,
            weights=self.weights,
            motion_scale=self.motion_scale,
            crossing=crossing,
            cross_penalty=self.cross_penalty,
            candidate_scores=[float(c.score) for c in candidates],
        )
        best = ranked[0]
        scores = {s.index: s for s in ranked}

        status, confidence, message = self._decide(ranked, current_bbox, crossing)
        matched_bbox = boxes[best.index].copy()

        # --- 目标记忆更新 --------------------------------------------------
        template_updated = False
        template_frozen = False
        conflict = self._candidate_conflict(ranked)
        if status in {CorrectionStatus.OK, CorrectionStatus.REDETECTED}:
            decision = self.bank.try_add(
                feats[best.index],
                confidence=confidence,
                bbox=matched_bbox,
                candidates_conflicting=conflict,
                low_confidence=confidence < self.accept_thresh,
                occluded=current_bbox is None,
            )
            template_updated = decision.accepted
            template_frozen = decision.frozen
        elif status in {CorrectionStatus.DRIFT, CorrectionStatus.NO_MATCH}:
            # 漂移或未匹配: 冻结模板, 避免把背景或其他鱼写入记忆
            self.bank.freeze("drift" if status is CorrectionStatus.DRIFT else "no_match")
            template_frozen = True

        latency = time.monotonic() - started
        result = CorrectionResult(
            frame_id=request.frame_id,
            timestamp=request.timestamp,
            wall_time=time.monotonic(),
            status=status,
            target_track_id=request.target_track_id,
            generation=request.generation,
            confidence=float(confidence),
            matched_bbox=matched_bbox,
            matched_detection_index=int(best.index),
            drift_score=float(best.total),
            latency_s=float(latency),
            message=message,
            scores={f"cand{s.index}": s.as_dict() for s in ranked},
        )
        if status is CorrectionStatus.REDETECTED:
            LOGGER.info("慢速环重捕: %s", result.message)
        return result

    # -- 决策 ---------------------------------------------------------------
    def _decide(
        self,
        ranked: Sequence,
        current_bbox: Optional[np.ndarray],
        crossing: bool,
    ) -> Tuple[CorrectionStatus, float, str]:
        best = ranked[0]
        score = float(best.total)

        # 1) 完全没有候选与模板一致
        if score < self.drift_thresh:
            return (
                CorrectionStatus.DRIFT if current_bbox is not None else CorrectionStatus.NO_MATCH,
                score,
                f"best={score:.3f} < drift_thresh={self.drift_thresh:.2f}",
            )

        # 2) 当前目标本身与记忆一致
        if best.is_current_target:
            if score >= self.accept_thresh:
                return CorrectionStatus.OK, score, f"consistent ({score:.3f})"
            return (
                CorrectionStatus.DRIFT,
                score,
                f"current target suspicious ({score:.3f}) crossing={crossing}",
            )

        # 3) 当前目标不在候选里, 但另有候选与记忆高度一致 -> 目标重现或切换
        winner_index = best.index
        runner_up = next(
            (s for s in ranked if s.is_current_target), None
        )
        if runner_up is not None and score - runner_up.total < self.switch_margin:
            # 证据不足以区分: 不切换目标, 保守判为疑似漂移
            return (
                CorrectionStatus.DRIFT,
                score,
                f"ambiguous switch: {score:.3f} vs {runner_up.total:.3f}",
            )

        if score >= self.accept_thresh:
            status = (
                CorrectionStatus.REDETECTED
                if current_bbox is None
                else CorrectionStatus.TARGET_SWITCH
            )
            return status, score, f"matched cand{winner_index} ({score:.3f})"

        if score >= self.reid_thresh:
            return (
                CorrectionStatus.REDETECTED,
                score,
                f"weak reid cand{winner_index} ({score:.3f})",
            )

        return CorrectionStatus.DRIFT, score, f"no confident match ({score:.3f})"

    def _candidate_conflict(self, ranked: Sequence) -> bool:
        """两个候选分数接近即视为冲突: 此时不得更新模板。"""
        if len(ranked) < 2:
            return False
        return (ranked[0].total - ranked[1].total) < self.switch_margin

    # -- 内部 ---------------------------------------------------------------
    def _select_candidates(self, request: CorrectionRequest) -> List[Detection]:
        """只处理有限数量候选区域, 并按分数与位置先验裁剪。"""
        cands = [d for d in request.candidates if d.score > 0]
        if not cands:
            return []
        if len(cands) <= self.max_candidates:
            return list(cands)

        reference = request.target_bbox
        if reference is None:
            cands.sort(key=lambda d: d.score, reverse=True)
            return cands[: self.max_candidates]

        template_boxes = np.stack([np.asarray(d.xyxy, dtype=np.float32) for d in cands])
        ref = np.asarray(reference, dtype=np.float32)[None, :]
        ious = iou_matrix(template_boxes, ref).ravel()
        order = np.lexsort((-np.array([d.score for d in cands]), -ious))
        return [cands[i] for i in order[: self.max_candidates]]

    def _embed(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """对候选区域提取特征。为提升鱼体区域占比, 先按 crop_expand 外扩。"""
        expanded = np.stack(
            [
                expand_box(b, self.crop_expand, self.width, self.height)
                for b in clip_boxes(boxes, self.width, self.height)
            ]
        )
        return self.embedder.embed(image, expanded)

    @staticmethod
    def _current_bbox(
        candidates: Sequence[Detection], target_bbox: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """在候选集合中定位"当前目标框"的对应项, 找不到则返回 None。

        使用候选集合而非 request 本身, 保证位置/运动分量与候选处于同一帧。
        """
        if target_bbox is None:
            return None
        ref = np.asarray(target_bbox, dtype=np.float32)
        boxes = np.stack([np.asarray(c.xyxy, dtype=np.float32) for c in candidates])
        ious = iou_matrix(boxes, ref[None, :]).ravel()
        idx = int(np.argmax(ious))
        return boxes[idx] if ious[idx] > 0.3 else None

    @staticmethod
    def _detect_crossing(boxes: np.ndarray, current_bbox: Optional[np.ndarray]) -> bool:
        """鱼体交叉检测: 其他候选与当前目标显著重叠。"""
        if current_bbox is None or len(boxes) < 2:
            return False
        ref = np.asarray(current_bbox, dtype=np.float32)[None, :]
        ious = iou_matrix(boxes, ref).ravel()
        # 除自身外还有别的候选与当前目标重叠 > 0.35
        others = sum(1 for v in ious if v > 0.35)
        return others >= 2

    def _result(
        self,
        request: CorrectionRequest,
        status: CorrectionStatus,
        started: float,
        *,
        message: str = "",
        confidence: float = 0.0,
    ) -> CorrectionResult:
        return CorrectionResult(
            frame_id=request.frame_id,
            timestamp=request.timestamp,
            wall_time=time.monotonic(),
            status=status,
            target_track_id=request.target_track_id,
            generation=request.generation,
            confidence=confidence,
            latency_s=float(time.monotonic() - started),
            message=message,
        )


# ---------------------------------------------------------------------------
# 门控: 在快速环侧执行(方案 4.2 节)
# ---------------------------------------------------------------------------
class GateDecision:
    """门控结论。"""

    __slots__ = ("apply", "status", "reason", "position_weight")

    def __init__(self, apply: bool, status: CorrectionStatus, reason: str, position_weight: float = 0.0):
        self.apply = apply
        self.status = status
        self.reason = reason
        self.position_weight = position_weight

    def __repr__(self) -> str:  # pragma: no cover
        return f"GateDecision(apply={self.apply}, status={self.status.value}, reason={self.reason})"


class CorrectionGate:
    """校正结果门控。

    方案 4.2 节: "校正门控检查数据年龄与当前目标一致性; 过旧或目标已切换的
    结果直接丢弃。"
    """

    def __init__(
        self,
        *,
        max_age_s: float = 0.6,
        max_generation_delta: int = 0,
        min_confidence: float = 0.4,
        smooth_alpha: float = 0.35,
        max_offset_ratio: float = 0.25,
    ) -> None:
        self.max_age_s = float(max_age_s)
        self.max_generation_delta = int(max_generation_delta)
        self.min_confidence = float(min_confidence)
        self.smooth_alpha = float(smooth_alpha)
        self.max_offset_ratio = float(max_offset_ratio)

    def evaluate(
        self,
        result: CorrectionResult,
        state: TargetState,
        *,
        now: Optional[float] = None,
    ) -> GateDecision:
        now = time.monotonic() if now is None else now

        if result.status in {
            CorrectionStatus.STALE,
            CorrectionStatus.TIMEOUT,
            CorrectionStatus.OVERLOADED,
            CorrectionStatus.DISABLED,
        }:
            return GateDecision(False, result.status, f"status={result.status.value}")

        age = result.age_s(now)
        if age > self.max_age_s:
            return GateDecision(
                False, CorrectionStatus.STALE,
                f"result too old ({age:.2f}s > {self.max_age_s:.2f}s)",
            )

        # 目标代次一致性: 代次变化说明目标已切换, 旧结果全部作废
        generation_delta = abs(state.generation - result.generation)
        if generation_delta > self.max_generation_delta:
            return GateDecision(
                False, CorrectionStatus.STALE,
                f"generation changed ({result.generation} -> {state.generation})",
            )

        if result.target_track_id >= 0 and state.track_id >= 0 and \
                result.target_track_id != state.track_id:
            return GateDecision(
                False, CorrectionStatus.STALE,
                f"track id mismatch ({result.target_track_id} != {state.track_id})",
            )

        if result.status in {CorrectionStatus.OK, CorrectionStatus.REDETECTED} and \
                result.confidence < self.min_confidence:
            return GateDecision(
                False, result.status,
                f"confidence too low ({result.confidence:.2f})",
            )

        if result.status in {CorrectionStatus.OK, CorrectionStatus.REDETECTED}:
            return GateDecision(True, result.status, result.message, position_weight=0.0)

        # DRIFT / TARGET_SWITCH / NO_MATCH 只影响置信度与追近许可, 不产生位置修正
        return GateDecision(False, result.status, result.message)

    def smooth_offset(
        self,
        previous: Optional[np.ndarray],
        target: np.ndarray,
        reference_size: float,
    ) -> np.ndarray:
        """平滑并限幅位置偏移。方案要求"先映射到当前帧, 再平滑应用"。

        Args:
            reference_size: 目标框对角线长度, 用于把偏移限制在合理范围内。
        """
        offset = np.asarray(target, dtype=np.float32).ravel()[:2]
        limit = max(1.0, reference_size * self.max_offset_ratio)
        offset = np.clip(offset, -limit, limit)
        if previous is None:
            return (offset * self.smooth_alpha).astype(np.float32)
        blended = previous * (1.0 - self.smooth_alpha) + offset * self.smooth_alpha
        return np.clip(blended, -limit, limit).astype(np.float32)
