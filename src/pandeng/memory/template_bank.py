"""目标记忆(模板库)。

对应方案第 4.1 节:
- 目标连续稳定可见时, 提取鱼体区域特征, 建立少量高置信度外观模板;
- 模板只在高置信度一致匹配时更新;
- 遮挡、低置信度和候选冲突期间冻结模板, 避免把背景或其他鱼写入记忆。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["Template", "TemplateBank", "TemplateUpdateDecision"]

LOGGER = logging.getLogger(__name__)


@dataclass
class Template:
    """单个外观模板。"""

    feature: np.ndarray
    created_at: float
    score: float                 # 建立/更新时的关联置信度
    hits: int = 1
    last_used: float = 0.0
    aspect_ratio: float = 1.0    # 建立时的框宽高比, 用于位置先验

    def similarity(self, feature: np.ndarray) -> float:
        """余弦相似度, 特征均已 L2 归一化。"""
        return float(np.dot(self.feature, feature))


@dataclass
class TemplateUpdateDecision:
    accepted: bool
    reason: str
    best_similarity: float = 0.0
    frozen: bool = False


class TemplateBank:
    """少量高置信度模板的集合。"""

    def __init__(
        self,
        max_templates: int = 5,
        min_template_score: float = 0.75,
        update_momentum: float = 0.2,
        consistency_gate: float = 0.70,
        freeze_on_occlusion: bool = True,
        max_freeze_s: float = 1.5,
        min_update_interval_s: float = 0.5,
    ) -> None:
        self.max_templates = int(max_templates)
        self.min_template_score = float(min_template_score)
        self.update_momentum = float(update_momentum)
        self.consistency_gate = float(consistency_gate)
        self.freeze_on_occlusion = bool(freeze_on_occlusion)
        self.max_freeze_s = float(max_freeze_s)
        self.min_update_interval_s = float(min_update_interval_s)

        self.templates: List[Template] = []
        self._frozen_until: float = 0.0
        self._frozen_reason: str = ""
        self._last_update: float = 0.0
        self.generation: int = 0

    # -- 状态 ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.templates)

    @property
    def is_empty(self) -> bool:
        return not self.templates

    def is_frozen(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        return now < self._frozen_until

    @property
    def frozen_reason(self) -> str:
        return self._frozen_reason if self.is_frozen() else ""

    def freeze(self, reason: str, duration_s: Optional[float] = None) -> None:
        """冻结模板更新。遮挡、低置信度、候选冲突期间调用。"""
        duration = self.max_freeze_s if duration_s is None else float(duration_s)
        until = time.monotonic() + duration
        if until > self._frozen_until:
            self._frozen_until = until
            self._frozen_reason = reason
            LOGGER.debug("模板冻结 %.2fs: %s", duration, reason)

    def unfreeze(self) -> None:
        self._frozen_until = 0.0
        self._frozen_reason = ""

    def on_generation_change(self) -> None:
        """目标代次变化(确认切换或重捕)后清空记忆, 避免跨个体污染。"""
        self.templates.clear()
        self.unfreeze()
        self._last_update = 0.0

    # -- 匹配 ---------------------------------------------------------------
    def best_similarity(self, feature: np.ndarray) -> Tuple[float, int]:
        """返回与模板集合的最佳相似度及其索引; 无模板时返回 (0, -1)。"""
        if not self.templates:
            return 0.0, -1
        sims = [t.similarity(feature) for t in self.templates]
        idx = int(np.argmax(sims))
        return float(sims[idx]), idx

    def mean_similarity(self, feature: np.ndarray) -> float:
        if not self.templates:
            return 0.0
        return float(np.mean([t.similarity(feature) for t in self.templates]))

    # -- 更新 ---------------------------------------------------------------
    def try_add(
        self,
        feature: np.ndarray,
        *,
        confidence: float,
        bbox: Optional[np.ndarray] = None,
        now: Optional[float] = None,
        candidates_conflicting: bool = False,
        low_confidence: bool = False,
        occluded: bool = False,
    ) -> TemplateUpdateDecision:
        """尝试建立或更新模板。

        只有在"高置信度一致匹配"时才更新; 遮挡/低置信度/候选冲突期间冻结。
        """
        now = time.monotonic() if now is None else now
        feature = np.asarray(feature, dtype=np.float32).ravel()

        if self.freeze_on_occlusion and (occluded or low_confidence or candidates_conflicting):
            reason = (
                "occlusion" if occluded else
                "low_confidence" if low_confidence else
                "candidate_conflict"
            )
            self.freeze(reason)
            return TemplateUpdateDecision(False, f"frozen:{reason}", frozen=True)

        if self.is_frozen(now):
            return TemplateUpdateDecision(False, f"frozen:{self.frozen_reason}", frozen=True)

        if confidence < self.min_template_score:
            return TemplateUpdateDecision(False, "below_template_score")

        if bbox is not None:
            x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
            aspect = float((x2 - x1) / max(1e-6, y2 - y1))
        else:
            aspect = 1.0

        if not self.templates:
            self.templates.append(
                Template(feature=feature, created_at=now, score=confidence,
                         hits=1, last_used=now, aspect_ratio=aspect)
            )
            self._last_update = now
            return TemplateUpdateDecision(True, "created", 1.0)

        best_sim, best_idx = self.best_similarity(feature)
        if best_sim < self.consistency_gate:
            # 与既有记忆不一致: 冻结而非写入, 避免模板污染
            self.freeze("inconsistent")
            return TemplateUpdateDecision(False, "inconsistent", best_sim, frozen=True)

        if now - self._last_update < self.min_update_interval_s:
            return TemplateUpdateDecision(False, "rate_limited", best_sim)

        # 滑动更新已有模板
        target = self.templates[best_idx]
        m = self.update_momentum
        merged = (1.0 - m) * target.feature + m * feature
        norm = float(np.linalg.norm(merged))
        target.feature = (merged / max(norm, 1e-9)).astype(np.float32)
        target.score = confidence
        target.hits += 1
        target.last_used = now
        target.aspect_ratio = 0.8 * target.aspect_ratio + 0.2 * aspect
        self._last_update = now

        if len(self.templates) < self.max_templates and best_sim < self.consistency_gate + 0.15:
            # 多样性不足时补充一个新模板(仍受 max_templates 限制)
            self.templates.append(
                Template(feature=feature, created_at=now, score=confidence,
                         hits=1, last_used=now, aspect_ratio=aspect)
            )
            return TemplateUpdateDecision(True, "updated+appended", best_sim)

        return TemplateUpdateDecision(True, "updated", best_sim)

    def prune(self, now: Optional[float] = None, max_age_s: float = 120.0) -> None:
        """清理长期未使用的模板, 保持模板数量有限。"""
        now = time.monotonic() if now is None else now
        self.templates = [t for t in self.templates if now - t.last_used <= max_age_s]
        if len(self.templates) > self.max_templates:
            self.templates.sort(key=lambda t: (t.score, t.hits), reverse=True)
            self.templates = self.templates[: self.max_templates]

    def snapshot(self) -> List[dict]:
        return [
            {
                "score": t.score,
                "hits": t.hits,
                "created_at": t.created_at,
                "last_used": t.last_used,
                "aspect_ratio": t.aspect_ratio,
            }
            for t in self.templates
        ]
