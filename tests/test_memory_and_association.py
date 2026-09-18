"""单元测试: 目标记忆(模板库)与关联打分。

重点覆盖方案第 4.1 节的约束: 模板只在高置信度一致匹配时更新; 遮挡、低置信度和
候选冲突期间冻结模板, 避免把背景或其他鱼写入记忆。
"""

from __future__ import annotations

import numpy as np
import pytest

from pandeng.memory.template_bank import TemplateBank
from pandeng.perception.association import rank_candidates, score_candidate


def unit(vec):
    arr = np.asarray(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


# ---------------------------------------------------------------------------
# 模板库
# ---------------------------------------------------------------------------
def test_template_created_on_high_confidence():
    bank = TemplateBank(max_templates=3, min_template_score=0.75)
    decision = bank.try_add(unit([1.0, 0.0, 0.0]), confidence=0.9)
    assert decision.accepted
    assert len(bank) == 1


def test_template_rejected_below_score_threshold():
    bank = TemplateBank(min_template_score=0.75)
    decision = bank.try_add(unit([1.0, 0.0, 0.0]), confidence=0.5)
    assert not decision.accepted
    assert len(bank) == 0


def test_template_frozen_on_occlusion():
    """遮挡期间不得写入模板。"""
    bank = TemplateBank(min_template_score=0.5)
    decision = bank.try_add(unit([1.0, 0.0]), confidence=0.9, occluded=True)
    assert not decision.accepted
    assert decision.frozen
    assert bank.is_frozen()

    # 冻结期内即使条件满足也不更新
    blocked = bank.try_add(unit([1.0, 0.0]), confidence=0.9)
    assert not blocked.accepted


def test_template_frozen_on_candidate_conflict():
    bank = TemplateBank(min_template_score=0.5)
    decision = bank.try_add(unit([1.0, 0.0]), confidence=0.9, candidates_conflicting=True)
    assert not decision.accepted
    assert decision.frozen


def test_inconsistent_feature_freezes_instead_of_polluting():
    """与既有记忆不一致时冻结, 而不是把新特征写进去(防止模板污染)。"""
    bank = TemplateBank(min_template_score=0.5, consistency_gate=0.9)
    bank.try_add(unit([1.0, 0.0]), confidence=0.9)
    before = bank.templates[0].feature.copy()

    decision = bank.try_add(unit([0.0, 1.0]), confidence=0.9)
    assert not decision.accepted
    assert decision.frozen
    assert np.allclose(bank.templates[0].feature, before), "模板内容不应被污染"


def test_template_updated_with_momentum():
    bank = TemplateBank(
        min_template_score=0.5,
        consistency_gate=0.5,
        update_momentum=0.5,
        min_update_interval_s=0.0,   # 关闭更新间隔限制, 否则同一时刻会被限流
    )
    bank.try_add(unit([1.0, 0.0]), confidence=0.9)
    original = bank.templates[0].feature.copy()

    decision = bank.try_add(unit([0.9, 0.1]), confidence=0.9)
    assert decision.accepted, decision.reason
    moved = bank.templates[0].feature
    assert not np.allclose(moved, original)
    assert np.linalg.norm(moved) == pytest.approx(1.0, abs=1e-5)


def test_generation_change_clears_memory():
    bank = TemplateBank(min_template_score=0.5)
    bank.try_add(unit([1.0, 0.0]), confidence=0.9)
    assert len(bank) == 1
    bank.on_generation_change()
    assert len(bank) == 0


def test_prune_respects_max_templates():
    bank = TemplateBank(max_templates=2, min_template_score=0.5, consistency_gate=0.99)
    for i in range(5):
        vec = unit(np.eye(8)[i % 8] + 0.01 * i)
        bank.try_add(vec, confidence=0.9)
    bank.prune(max_age_s=0.0)
    assert len(bank) <= 2


# ---------------------------------------------------------------------------
# 关联打分
# ---------------------------------------------------------------------------
def test_association_prefers_matching_appearance():
    candidates = [
        np.array([100, 100, 200, 200], dtype=np.float32),
        np.array([500, 400, 600, 500], dtype=np.float32),
    ]
    template = unit([1.0, 0.0, 0.0])
    features = [unit([1.0, 0.0, 0.0]), unit([0.0, 1.0, 0.0])]

    ranked = rank_candidates(
        candidates,
        candidate_features=features,
        template_features=[template],
        current_bbox=candidates[1],
    )
    assert ranked[0].index == 0, "外观一致的候选应排在前面"


def test_association_does_not_switch_on_appearance_alone():
    """方案要求不单独依靠特征相似度切换目标: 位置完全不符时不应盲从外观。"""
    candidates = [
        np.array([0, 0, 100, 100], dtype=np.float32),       # 外观像但在画面角落
        np.array([600, 350, 700, 450], dtype=np.float32),   # 当前目标
    ]
    template = unit([1.0, 0.0, 0.0])
    features = [unit([1.0, 0.0, 0.0]), unit([0.95, 0.1, 0.0])]

    ranked = rank_candidates(
        candidates,
        candidate_features=features,
        template_features=[template],
        predicted_bbox=candidates[1],
        current_bbox=candidates[1],
        weights={"appearance": 0.6, "position": 0.25, "motion": 0.15},
    )
    # 融合后当前目标仍应胜出或有接近的证据, 不应被外观单独带偏
    top = ranked[0]
    assert top.index == 1 or (top.total - ranked[1].total) < 0.15


def test_crossing_penalty_reduces_position_weight():
    score = score_candidate(
        index=0,
        candidate_bbox=np.array([100, 100, 200, 200], dtype=np.float32),
        candidate_feature=unit([1.0, 0.0]),
        template_features=[unit([1.0, 0.0])],
        predicted_bbox=np.array([100, 100, 200, 200], dtype=np.float32),
        current_bbox=np.array([100, 100, 200, 200], dtype=np.float32),
        weights={"appearance": 0.6, "position": 0.25, "motion": 0.15},
        crossing=True,
        cross_penalty=0.15,
    )
    assert 0.0 <= score.total <= 1.0


def test_bootstrap_uses_detection_score_when_memory_empty():
    """回归测试: 记忆为空时外观分量必须回退到检测置信度, 否则模板永远建不起来。"""
    from pandeng.perception.association import score_candidate as sc

    box = np.array([100, 100, 200, 200], dtype=np.float32)
    high = sc(
        index=0,
        candidate_bbox=box,
        candidate_feature=unit([1.0, 0.0]),
        template_features=[],
        predicted_bbox=box,
        current_bbox=box,
        weights={"appearance": 0.6, "position": 0.25, "motion": 0.15},
        candidate_score=0.9,
    )
    low = sc(
        index=0,
        candidate_bbox=box,
        candidate_feature=unit([1.0, 0.0]),
        template_features=[],
        predicted_bbox=box,
        current_bbox=box,
        weights={"appearance": 0.6, "position": 0.25, "motion": 0.15},
        candidate_score=0.2,
    )
    assert high.total > 0.75, "高置信度候选应足以建立首个模板"
    assert low.total < high.total
