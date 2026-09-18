"""时序预测(第二阶段研究内容)。

首版交付不含本模块的模型, 但提供接口与恒速基线, 使实验组 D 可先行运行。
"""

from __future__ import annotations

from .predictors import (
    BasePredictor,
    ConstantVelocityPredictor,
    LearnedPredictor,
    PredictionResult,
    build_predictor,
)

__all__ = [
    "BasePredictor",
    "ConstantVelocityPredictor",
    "LearnedPredictor",
    "PredictionResult",
    "build_predictor",
]
