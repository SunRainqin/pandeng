"""感知层: 检测(YOLO11n)、跟踪(ByteTrack)、特征提取(DINOv3)、低频校正。"""

from __future__ import annotations

from .association import AssociationScores, edge_margin, rank_candidates, score_candidate
from .corrector import CorrectionGate, GateDecision, SlowCorrector
from .detector import BaseDetector, MockDetector, OnnxDetector, YoloDetector, build_detector
from .embedder import BaseEmbedder, DinoV3Embedder, MockEmbedder, build_embedder
from .tracker import ByteTracker, STrack

__all__ = [
    "AssociationScores",
    "BaseDetector",
    "BaseEmbedder",
    "ByteTracker",
    "CorrectionGate",
    "DinoV3Embedder",
    "GateDecision",
    "MockDetector",
    "MockEmbedder",
    "OnnxDetector",
    "STrack",
    "SlowCorrector",
    "YoloDetector",
    "build_detector",
    "build_embedder",
    "edge_margin",
    "rank_candidates",
    "score_candidate",
]
