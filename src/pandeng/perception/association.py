"""候选关联打分。

对应方案第 4.1 节:
"低频链路对当前候选提取特征, 结合外观相似度、预测位置和运动连续性判断关联,
 不单独依靠特征相似度切换目标。"

因此这里把三种证据显式拆开, 分别计算后再融合, 并保留各分量以便:
- 写入记录文件, 供离线分析"错误校正"与"目标误切";
- 在鱼体交叉时降低位置分量权重(交叉时位置先验不可靠)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..utils.geometry import box_iou, normalize_error

__all__ = ["AssociationScores", "score_candidate", "rank_candidates"]


@dataclass
class AssociationScores:
    """单个候选的关联证据。"""

    index: int
    appearance: float
    position: float
    motion: float
    total: float
    is_current_target: bool = False

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "index": self.index,
            "appearance": round(self.appearance, 4),
            "position": round(self.position, 4),
            "motion": round(self.motion, 4),
            "total": round(self.total, 4),
            "current": self.is_current_target,
        }


def _motion_score(
    candidate: np.ndarray,
    predicted: Optional[np.ndarray],
    scale: float,
) -> float:
    """运动连续性: 候选中心与预测中心的距离按目标尺度归一化后指数衰减。

    使用指数形式而非硬阈值, 使"位置略微跳变"平滑降权而不是直接判死,
    便于在处理鱼体交叉时仍保留外观证据的作用。
    """
    if predicted is None:
        return 0.5  # 无预测信息时给中性分, 不引入偏向
    cx, cy = (candidate[0] + candidate[2]) * 0.5, (candidate[1] + candidate[3]) * 0.5
    px, py = (predicted[0] + predicted[2]) * 0.5, (predicted[1] + predicted[3]) * 0.5
    dist = float(np.hypot(cx - px, cy - py))
    diag = float(np.hypot(predicted[2] - predicted[0], predicted[3] - predicted[1]))
    sigma = max(1.0, diag * max(1e-3, scale))
    return float(np.exp(-dist / sigma))


def score_candidate(
    *,
    index: int,
    candidate_bbox: np.ndarray,
    candidate_feature: Optional[np.ndarray],
    template_features: Sequence[np.ndarray],
    predicted_bbox: Optional[np.ndarray],
    current_bbox: Optional[np.ndarray],
    weights: dict[str, float],
    motion_scale: float = 1.5,
    crossing: bool = False,
    cross_penalty: float = 0.15,
    candidate_score: float = 0.0,
) -> AssociationScores:
    """计算单个候选的融合关联得分。

    Args:
        candidate_feature: 已 L2 归一化的外观特征; 若为 None 则外观分取中性值。
        template_features: 目标记忆中的模板特征集合。
        predicted_bbox: 快速环对当前帧的预测框(用于位置与运动分量)。
        current_bbox: 快速环本帧的目标框, 用于标记"该候选是否就是当前目标"。
        crossing: 是否处于鱼体交叉场景, 为真时降低位置分量权重。
        candidate_score: 候选的检测置信度。**仅在目标记忆为空时**作为外观分量
            的代理 —— 冷启动阶段没有模板可比, 若一律给中性分 0.5, 融合总分
            永远低于建档门槛, 模板将永远无法建立(引导死锁)。此时改用检测
            置信度衡量"该区域是一个稳定且可信的鱼体", 与方案第 4.1 节
            "目标连续稳定可见时建立少量高置信度外观模板"一致。
    """
    # --- 外观 -------------------------------------------------------------
    if candidate_feature is not None and len(template_features) > 0:
        sims = [float(np.dot(np.asarray(t, dtype=np.float32).ravel(), candidate_feature))
                for t in template_features]
        appearance = float(np.clip(max(sims), -1.0, 1.0))
        appearance = (appearance + 1.0) * 0.5  # 映射到 [0, 1]
    elif len(template_features) == 0:
        appearance = float(np.clip(candidate_score, 0.0, 1.0))
    else:
        appearance = 0.5

    # --- 位置 -------------------------------------------------------------
    reference = current_bbox if current_bbox is not None else predicted_bbox
    position = box_iou(np.asarray(candidate_bbox, dtype=np.float32), np.asarray(reference, dtype=np.float32)) \
        if reference is not None else 0.5

    # --- 运动连续性 -------------------------------------------------------
    motion = _motion_score(
        np.asarray(candidate_bbox, dtype=np.float32), predicted_bbox, motion_scale
    )

    is_current = False
    if current_bbox is not None:
        is_current = box_iou(
            np.asarray(candidate_bbox, dtype=np.float32),
            np.asarray(current_bbox, dtype=np.float32),
        ) > 0.5

    w = dict(weights)
    if crossing:
        # 交叉时位置先验不可靠, 把位置权重让给外观与运动
        shift = min(float(w.get("position", 0.25)), float(cross_penalty))
        w["position"] = float(w.get("position", 0.25)) - shift
        w["appearance"] = float(w.get("appearance", 0.6)) + shift * 0.5
        w["motion"] = float(w.get("motion", 0.15)) + shift * 0.5

    total_weight = sum(max(0.0, float(v)) for v in w.values()) or 1.0
    total = (
        max(0.0, float(w.get("appearance", 0.0))) * appearance
        + max(0.0, float(w.get("position", 0.0))) * position
        + max(0.0, float(w.get("motion", 0.0))) * motion
    ) / total_weight

    return AssociationScores(
        index=int(index),
        appearance=appearance,
        position=float(position),
        motion=float(motion),
        total=float(np.clip(total, 0.0, 1.0)),
        is_current_target=bool(is_current),
    )


def rank_candidates(
    candidates: Sequence[np.ndarray],
    *,
    candidate_features: Optional[Sequence[Optional[np.ndarray]]] = None,
    template_features: Sequence[np.ndarray] = (),
    predicted_bbox: Optional[np.ndarray] = None,
    current_bbox: Optional[np.ndarray] = None,
    weights: Optional[dict[str, float]] = None,
    motion_scale: float = 1.5,
    crossing: bool = False,
    cross_penalty: float = 0.15,
    candidate_scores: Optional[Sequence[float]] = None,
) -> list[AssociationScores]:
    """对全部候选打分并按总分降序返回。"""
    weights = weights or {"appearance": 0.6, "position": 0.25, "motion": 0.15}
    feats = list(candidate_features) if candidate_features is not None else [None] * len(candidates)
    det_scores = list(candidate_scores) if candidate_scores is not None else [0.0] * len(candidates)
    scores = [
        score_candidate(
            index=i,
            candidate_bbox=np.asarray(box, dtype=np.float32),
            candidate_feature=feats[i],
            template_features=template_features,
            predicted_bbox=predicted_bbox,
            current_bbox=current_bbox,
            weights=weights,
            motion_scale=motion_scale,
            crossing=crossing,
            cross_penalty=cross_penalty,
            candidate_score=float(det_scores[i]),
        )
        for i, box in enumerate(candidates)
    ]
    scores.sort(key=lambda s: s.total, reverse=True)
    return scores


def edge_margin(bbox: np.ndarray, width: int, height: int) -> float:
    """目标中心到画面边缘的最小归一化边距, 用于"接近视场边缘"的事件触发。

    返回 0 表示中心已到边缘, 0.5 表示位于画面正中央。
    """
    ex, ey = normalize_error(
        ((float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5),
        width,
        height,
    )
    return float(min(0.5 - abs(ex) * 0.5, 0.5 - abs(ey) * 0.5))
