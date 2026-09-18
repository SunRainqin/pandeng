"""时序预测接口。

对应方案第 6 节。首版交付不含时序预测, 但实验组 D/E 需要该接口:

- 输出: 短时目标图像位置、预测置信度、转向修正和减速建议;
- 艇端部署: 小型时序模型异步推理, 不直接加载大型教师;
- 约束: 输入仅使用当前及历史观测; 预测只提供受限修正, 目标丢失时仍禁止
  盲目前进(由控制器与仲裁器共同保证, 见 `pandeng.control`)。

`ConstantVelocityPredictor` 是方案实验组 D 的基线; `LearnedPredictor` 是
第二阶段(V-JEPA 2.1 教师 -> 蒸馏轻量时序头)的接入点。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "PredictionResult",
    "BasePredictor",
    "ConstantVelocityPredictor",
    "LearnedPredictor",
    "build_predictor",
]

LOGGER = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """短时预测输出。"""

    ex: float = 0.0            # 预测的归一化水平位置
    ey: float = 0.0            # 预测的归一化垂直位置
    confidence: float = 0.0
    horizon_s: float = 0.0
    yaw_hint: float = 0.0      # 受限转向修正建议
    surge_scale: float = 1.0   # 减速建议(乘性)
    will_leave_view: bool = False
    source: str = "none"
    valid: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "ex": round(self.ex, 4),
            "ey": round(self.ey, 4),
            "confidence": round(self.confidence, 4),
            "horizon_s": round(self.horizon_s, 3),
            "yaw_hint": round(self.yaw_hint, 4),
            "surge_scale": round(self.surge_scale, 4),
            "will_leave_view": self.will_leave_view,
            "source": self.source,
            "valid": self.valid,
        }


class BasePredictor(ABC):
    """预测器接口。"""

    name: str = "base"

    def __init__(self, *, horizon_s: float = 0.5, history_len: int = 30) -> None:
        self.horizon_s = float(horizon_s)
        self.history: Deque[Tuple[float, float, float]] = deque(maxlen=int(history_len))

    def observe(self, *, timestamp: float, ex: float, ey: float, visible: bool = True) -> None:
        """记录一次观测(仅使用当前及历史观测, 不使用未来信息)。"""
        if visible:
            self.history.append((float(timestamp), float(ex), float(ey)))

    def reset(self) -> None:
        self.history.clear()

    @abstractmethod
    def predict(self) -> PredictionResult:
        """给出短时预测。"""

    def close(self) -> None:  # pragma: no cover
        pass


class ConstantVelocityPredictor(BasePredictor):
    """恒速预测基线(实验组 D)。

    对历史 (t, ex, ey) 做最小二乘线性拟合, 外推 `horizon_s`。这是方案第 6 节
    明确要求的对比基线之一("恒速预测及无预测的 DINOv3 双环")。
    """

    name = "constant_velocity"

    def __init__(
        self,
        *,
        horizon_s: float = 0.5,
        history_len: int = 30,
        min_samples: int = 5,
        leave_view_margin: float = 0.85,
    ) -> None:
        super().__init__(horizon_s=horizon_s, history_len=history_len)
        self.min_samples = int(min_samples)
        self.leave_view_margin = float(leave_view_margin)

    def predict(self) -> PredictionResult:
        if len(self.history) < self.min_samples:
            return PredictionResult(source=self.name, valid=False)

        arr = np.asarray(self.history, dtype=np.float64)
        t = arr[:, 0] - arr[0, 0]
        if t[-1] <= 1e-3:
            return PredictionResult(source=self.name, valid=False)

        ex_slope, ex_intercept = np.polyfit(t, arr[:, 1], 1)
        ey_slope, ey_intercept = np.polyfit(t, arr[:, 2], 1)

        t_future = t[-1] + self.horizon_s
        ex_pred = float(ex_slope * t_future + ex_intercept)
        ey_pred = float(ey_slope * t_future + ey_intercept)

        # 置信度: 样本越多越可信, 拟合残差越大越不可信
        residual = float(
            np.mean(
                np.abs(arr[:, 1] - (ex_slope * t + ex_intercept))
                + np.abs(arr[:, 2] - (ey_slope * t + ey_intercept))
            )
        )
        confidence = float(
            np.clip(len(self.history) / (2.0 * self.min_samples), 0.0, 1.0)
            * np.clip(1.0 - residual * 4.0, 0.0, 1.0)
        )

        will_leave = abs(ex_pred) > 1.0 or abs(ey_pred) > self.leave_view_margin
        return PredictionResult(
            ex=ex_pred,
            ey=ey_pred,
            confidence=confidence,
            horizon_s=self.horizon_s,
            yaw_hint=0.0,
            surge_scale=0.5 if will_leave else 1.0,
            will_leave_view=bool(will_leave),
            source=self.name,
            valid=True,
        )


class LearnedPredictor(BasePredictor):
    """第二阶段: 学习的时序预测头。

    方案第 6 节要求岸端用 V-JEPA 2.1 提取教师特征训练/蒸馏轻量时序头, 艇端
    只部署小型模型。本类定义了接入点与输入/输出契约, 权重缺失时给出明确
    提示而不静默降级 —— 静默降级会让实验组 E 的结论失去意义。
    """

    name = "learned"

    def __init__(
        self,
        weights: str | Path,
        *,
        horizon_s: float = 0.5,
        history_len: int = 30,
        device: str = "cuda:0",
    ) -> None:
        super().__init__(horizon_s=horizon_s, history_len=history_len)
        self.weights_path = Path(weights)
        self.device = device
        self.model = self._load()
        self._extra_history: Deque[np.ndarray] = deque(maxlen=history_len)

    def _load(self):
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"未找到时序预测权重: {self.weights_path}\n"
                "第二阶段流程: 岸端用 V-JEPA 2.1 提取时序教师特征 -> 训练/蒸馏轻量"
                "时序头 -> 导出 TorchScript/ONNX 后再在艇端异步推理。\n"
                "若只想先跑通实验组 D, 请使用 prediction.backend=constant_velocity。"
            )
        import torch

        model = torch.jit.load(str(self.weights_path), map_location=self.device)
        model.eval()
        LOGGER.info("已加载时序预测模型: %s", self.weights_path)
        return model

    def observe_features(self, features: np.ndarray) -> None:
        """接入视频特征(由 DINOv3 或专门的时序特征提取器提供)。"""
        self._extra_history.append(np.asarray(features, dtype=np.float32).ravel())

    def predict(self) -> PredictionResult:
        if len(self.history) < 2 or not self._extra_history:
            return PredictionResult(source=self.name, valid=False)

        import torch

        track = np.asarray(self.history, dtype=np.float32)
        features = np.stack(list(self._extra_history))
        with torch.no_grad():
            batch = {
                "track": torch.from_numpy(track)[None].to(self.device),
                "features": torch.from_numpy(features)[None].to(self.device),
                "horizon": torch.tensor([[self.horizon_s]], device=self.device),
            }
            out = self.model(**batch)
        ex, ey, conf = (float(out["ex"]), float(out["ey"]), float(out["confidence"]))
        return PredictionResult(
            ex=ex,
            ey=ey,
            confidence=conf,
            horizon_s=self.horizon_s,
            yaw_hint=float(out.get("yaw_hint", 0.0)),
            surge_scale=float(out.get("surge_scale", 1.0)),
            will_leave_view=bool(abs(ex) > 1.0 or abs(ey) > 1.0),
            source=self.name,
            valid=True,
        )


def build_predictor(cfg) -> Optional[BasePredictor]:
    """按配置构建预测器; `backend=none` 时返回 None。"""
    backend = str(cfg.get_path("backend", "none")).lower()
    if backend in {"none", "off", "disabled"}:
        return None
    horizon = float(cfg.get_path("horizon_s", 0.5))
    history_len = int(cfg.get_path("history_len", 30))
    if backend == "constant_velocity":
        return ConstantVelocityPredictor(
            horizon_s=horizon,
            history_len=history_len,
            min_samples=int(cfg.get_path("min_samples", 5)),
            leave_view_margin=float(cfg.get_path("leave_view_margin", 0.85)),
        )
    if backend == "learned":
        return LearnedPredictor(
            cfg.get_path("weights", "weights/temporal_head.pt"),
            horizon_s=horizon,
            history_len=history_len,
            device=str(cfg.get_path("device", "cuda:0")),
        )
    raise ValueError(f"未知的预测后端: {backend}")
