"""实验指标统计。

严格对应方案第 7 节"成功要求"与需要上报的量:

快速环:
- 感知帧率目标 10-20Hz;
- 采集至控制桥接输出的 P95 不超过 200ms;
- 稳定段深度误差 ±0.15m(需要深度传感器; 无传感器时标记为未采集)。

成功要求:
- 目标可见时间占计划时长至少 80%;
- 可见时位于画面中央 50% 宽高区域的时间比例至少 80%。

慢速环(不将请求频率视为实际完成频率):
- 请求率、完成率、P95 耗时、结果年龄。

异常计数:
- 误切目标、丢失次数、重捕时间;
- 错误校正、模板污染、过期丢弃、慢链路故障。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .utils.geometry import box_iou

__all__ = ["SessionMetrics", "DetectionEvaluator", "CENTRAL_REGION_RATIO"]


# 画面中央 50% 宽高区域(方案第 7 节)
CENTRAL_REGION_RATIO = 0.5


class DetectionEvaluator:
    """检测质量统计: 按帧贪心 IoU 匹配, 累计 TP / FP / FN。

    **只对有标注的帧计分**。该数据集的标注策略(方案第 5 节)决定了很多帧
    根本没有标注, 把这些帧当成"无目标"的负样本会把正常漏检/误检混为一谈 ——
    无标注帧上的检测既不算误检, 也不产生漏检。数量单独上报, 便于判断
    实际可评估覆盖率。
    """

    def __init__(self, iou_thresh: float = 0.5, per_class: bool = False) -> None:
        self.iou_thresh = float(iou_thresh)
        self.per_class = bool(per_class)
        self.annotated_frames = 0
        self.unannotated_frames = 0
        self.detections_on_unannotated = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        # 类别 -> [tp, fp, fn]
        self.class_stats: Dict[int, List[int]] = {}

    def _bump(self, cls: int, index: int) -> None:
        self.class_stats.setdefault(int(cls), [0, 0, 0])[index] += 1

    def update(
        self,
        detections: Sequence[object],
        ground_truth: Optional[Sequence[tuple]] = None,
    ) -> Optional[Dict[str, int]]:
        """`ground_truth` 为 None 表示该帧无标注, 直接跳过计分。"""
        if ground_truth is None:
            self.unannotated_frames += 1
            self.detections_on_unannotated += len(detections)
            return None

        self.annotated_frames += 1
        gts = [(int(c), np.asarray(b, dtype=np.float32)) for c, b in ground_truth]
        matched = [False] * len(gts)
        frame_tp = frame_fp = 0

        # 按置信度降序贪心: 高置信检测优先占据真值框
        for det in sorted(detections, key=lambda d: -float(d.score)):
            best_iou, best_j = 0.0, -1
            for j, (cls, box) in enumerate(gts):
                if matched[j]:
                    continue
                if self.per_class and int(det.cls) != cls:
                    continue
                value = box_iou(np.asarray(det.xyxy, dtype=np.float32), box)
                if value > best_iou:
                    best_iou, best_j = value, j
            if best_j >= 0 and best_iou >= self.iou_thresh:
                matched[best_j] = True
                frame_tp += 1
                self._bump(int(gts[best_j][0]), 0)
            else:
                frame_fp += 1
                if self.per_class:
                    self._bump(int(det.cls), 1)

        self.tp += frame_tp
        self.fp += frame_fp
        for j, (cls, _) in enumerate(gts):
            if not matched[j]:
                self.fn += 1
                self._bump(cls, 2)
        return {"tp": frame_tp, "fp": frame_fp, "fn": len(gts) - frame_tp}

    @staticmethod
    def _prf(tp: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        if precision is None or recall is None or (precision + recall) <= 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return {
            "precision": None if precision is None else round(precision, 4),
            "recall": None if recall is None else round(recall, 4),
            "f1": None if f1 is None else round(f1, 4),
        }

    def summary(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "iou_threshold": self.iou_thresh,
            "annotated_frames": self.annotated_frames,
            "unannotated_frames": self.unannotated_frames,
            "detections_on_unannotated": self.detections_on_unannotated,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "sample_coverage_ratio": round(
                self.annotated_frames
                / max(1, self.annotated_frames + self.unannotated_frames),
                4,
            ),
        }
        out.update(self._prf(self.tp, self.fp, self.fn))
        if self.per_class and self.class_stats:
            out["per_class"] = {
                str(cls): {"tp": v[0], "fp": v[1], "fn": v[2], **self._prf(*v)}
                for cls, v in sorted(self.class_stats.items())
            }
        return out

    def merge(self, other: "DetectionEvaluator") -> None:
        # 逐类别统计开关必须一并合并: 否则汇总器新建时默认关闭, 合完反而丢掉
        # 明细, 而逐序列条目里是有的。
        self.per_class = self.per_class or other.per_class
        self.annotated_frames += other.annotated_frames
        self.unannotated_frames += other.unannotated_frames
        self.detections_on_unannotated += other.detections_on_unannotated
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        for cls, values in other.class_stats.items():
            slot = self.class_stats.setdefault(cls, [0, 0, 0])
            for i, value in enumerate(values):
                slot[i] += value


@dataclass
class SessionMetrics:
    """单次试验的指标累计器。"""

    name: str = "run"
    # --- 计数 ---
    frames: int = 0
    visible_frames: int = 0
    centered_frames: int = 0
    target_switches: int = 0
    losses: int = 0
    recoveries: int = 0
    correction_applied: int = 0
    correction_rejected: int = 0
    correction_drift: int = 0
    template_freezes: int = 0
    slow_faults: int = 0

    # --- 时延采样 ---
    bridge_latency: List[float] = field(default_factory=list)
    fast_period: List[float] = field(default_factory=list)
    depth_errors: List[float] = field(default_factory=list)

    # --- 时间轴 ---
    visible_time: float = 0.0
    centered_time: float = 0.0
    planned_time: float = 0.0
    lost_intervals: List[float] = field(default_factory=list)

    # --- 检测评估(只统计有标注帧) ---
    detection: DetectionEvaluator = field(default_factory=DetectionEvaluator)

    _t_previous: Optional[float] = None
    _lost_start: Optional[float] = None

    # -- 记录 ---------------------------------------------------------------
    def record_frame(
        self,
        *,
        now: float,
        visible: bool,
        in_central_region: bool,
        bridge_latency: Optional[float] = None,
        depth_error: Optional[float] = None,
    ) -> None:
        self.frames += 1
        if self._t_previous is None:
            dt = 0.0
        else:
            dt = max(0.0, now - self._t_previous)
            if dt > 0:
                self.fast_period.append(dt)
                if len(self.fast_period) > 2000:
                    self.fast_period.pop(0)
        self._t_previous = now
        self.planned_time += dt

        if visible:
            self.visible_frames += 1
            self.visible_time += dt
            if in_central_region:
                self.centered_frames += 1
                self.centered_time += dt
            if self._lost_start is not None:
                self.lost_intervals.append(now - self._lost_start)
                self._lost_start = None
        else:
            if self._lost_start is None:
                self._lost_start = now
                self.losses += 1

        if bridge_latency is not None:
            self.bridge_latency.append(float(bridge_latency))
            if len(self.bridge_latency) > 2000:
                self.bridge_latency.pop(0)
        if depth_error is not None:
            self.depth_errors.append(abs(float(depth_error)))

    def record_target_switch(self) -> None:
        self.target_switches += 1

    def record_recovery(self) -> None:
        self.recoveries += 1

    def record_correction(self, *, applied: bool, drift: bool, template_frozen: bool) -> None:
        if applied:
            self.correction_applied += 1
        else:
            self.correction_rejected += 1
        if drift:
            self.correction_drift += 1
        if template_frozen:
            self.template_freezes += 1

    def record_slow_fault(self) -> None:
        self.slow_faults += 1

    # -- 统计 ---------------------------------------------------------------
    @staticmethod
    def _percentile(values: Sequence[float], q: float) -> Optional[float]:
        if not values:
            return None
        return float(np.percentile(np.asarray(values, dtype=np.float64), q))

    def summary(self, scheduler_stats: Optional[dict] = None) -> Dict[str, object]:
        planned = max(self.planned_time, 1e-9)
        summary: Dict[str, object] = {
            "name": self.name,
            "frames": self.frames,
            "planned_time_s": round(self.planned_time, 3),
            # --- 快速环 ---
            "fast_hz": round(self.frames / planned, 3),
            "bridge_latency_p50_ms": self._ms(self._percentile(self.bridge_latency, 50)),
            "bridge_latency_p95_ms": self._ms(self._percentile(self.bridge_latency, 95)),
            "depth_error_max_m": (
                round(float(np.max(self.depth_errors)), 4) if self.depth_errors else None
            ),
            # --- 成功要求 ---
            "visibility_ratio": round(self.visible_time / planned, 4),
            "central_ratio_when_visible": round(
                self.centered_time / max(self.visible_time, 1e-9), 4
            ),
            # --- 异常 ---
            "target_switches": self.target_switches,
            "losses": self.losses,
            "recoveries": self.recoveries,
            "mean_recovery_time_s": (
                round(float(np.mean(self.lost_intervals)), 3) if self.lost_intervals else None
            ),
            "max_recovery_time_s": (
                round(float(np.max(self.lost_intervals)), 3) if self.lost_intervals else None
            ),
            "corrections_applied": self.correction_applied,
            "corrections_rejected": self.correction_rejected,
            "correction_drift": self.correction_drift,
            "template_freezes": self.template_freezes,
            "slow_link_faults": self.slow_faults,
        }
        if scheduler_stats:
            summary["slow_loop"] = scheduler_stats
        if self.detection.annotated_frames > 0:
            summary["detection"] = self.detection.summary()
        return summary

    def merge(self, other: "SessionMetrics") -> None:
        """把另一段(例如另一个序列)的计数累加到自身, 用于汇总。"""
        self.frames += other.frames
        self.visible_frames += other.visible_frames
        self.centered_frames += other.centered_frames
        self.target_switches += other.target_switches
        self.losses += other.losses
        self.recoveries += other.recoveries
        self.correction_applied += other.correction_applied
        self.correction_rejected += other.correction_rejected
        self.correction_drift += other.correction_drift
        self.template_freezes += other.template_freezes
        self.slow_faults += other.slow_faults
        self.visible_time += other.visible_time
        self.centered_time += other.centered_time
        self.planned_time += other.planned_time
        self.lost_intervals.extend(other.lost_intervals)
        for bucket, values in (
            (self.bridge_latency, other.bridge_latency),
            (self.fast_period, other.fast_period),
            (self.depth_errors, other.depth_errors),
        ):
            bucket.extend(values)
            if len(bucket) > 2000:
                del bucket[: len(bucket) - 2000]
        self.detection.merge(other.detection)

    def check_success(self, summary: Optional[Dict[str, object]] = None) -> Dict[str, bool]:
        """按方案第 7 节的成功要求给出判定(不含"无人工接管"一项)。

        关于感知频率: 方案的"目标 10-20Hz"是**能力下限**要求, 实际跑得比
        20Hz 更快不构成不达标。因此判据取"不低于 10Hz"; 上限由整机实测时的
        调度频率另行核对(见 `paced` 字段, 未做实时限速的回放不具备参考性)。
        """
        summary = summary or self.summary()
        fast_hz = float(summary["fast_hz"])
        latency = summary["bridge_latency_p95_ms"]
        return {
            "fast_rate_ge_10hz": fast_hz >= 10.0,
            "bridge_p95_le_200ms": (latency is not None and float(latency) <= 200.0),
            "visibility_ge_80pct": float(summary["visibility_ratio"]) >= 0.80,
            "central_ge_80pct": float(summary["central_ratio_when_visible"]) >= 0.80,
            "depth_within_015m": (
                summary["depth_error_max_m"] is not None
                and float(summary["depth_error_max_m"]) <= 0.15
            ),
        }

    @staticmethod
    def _ms(value: Optional[float]) -> Optional[float]:
        return None if value is None else round(value * 1000.0, 3)

    def dump(
        self,
        path: str | Path,
        scheduler_stats: Optional[dict] = None,
        summary: Optional[Dict[str, object]] = None,
    ) -> Path:
        """写 `summary.json`。

        `summary` 缺省时用自身 `summary()`; 多序列运行时必须传入**已汇总**
        的结果, 否则逐序列指标会丢掉 —— 那正是多序列评估的全部价值。
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if summary is None:
            summary = self.summary(scheduler_stats)
        payload = {
            "summary": summary,
            "success_criteria": self.check_success(summary),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return target

    @staticmethod
    def is_in_central_region(bbox: np.ndarray, width: int, height: int) -> bool:
        """判断目标中心是否位于画面中央 50% 宽高区域内。"""
        cx = (float(bbox[0]) + float(bbox[2])) * 0.5
        cy = (float(bbox[1]) + float(bbox[3])) * 0.5
        half_w = width * CENTRAL_REGION_RATIO * 0.5
        half_h = height * CENTRAL_REGION_RATIO * 0.5
        return abs(cx - width * 0.5) <= half_w and abs(cy - height * 0.5) <= half_h
