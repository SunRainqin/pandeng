"""快速检测器: YOLO11n。

对应方案第 2 节"快速检测 YOLO11n 预训练权重, 公开鱼类数据微调, 目标 10-20Hz"
与第 7 节"检测器采用 ONNX -> TensorRT FP16 部署"。

后端说明:
- `ultralytics`: 开发与训练阶段使用, 直接加载 `.pt`。
- `onnx` / `tensorrt`: Orin 端部署使用导出的引擎。
- `mock`: 无权重/无数据时用于打通链路的合成检测器。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..types import Detection

__all__ = ["BaseDetector", "YoloDetector", "OnnxDetector", "MockDetector", "build_detector"]

LOGGER = logging.getLogger(__name__)


class BaseDetector(ABC):
    """检测器接口。"""

    name: str = "base"

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """在 BGR 图像上执行检测。"""

    def warmup(self, image: np.ndarray, times: int = 3) -> None:
        for _ in range(max(0, int(times))):
            self.detect(image)

    def close(self) -> None:  # pragma: no cover - 资源释放
        pass


class YoloDetector(BaseDetector):
    """基于 ultralytics 的 YOLO11n 检测器。"""

    name = "yolo11n"

    def __init__(
        self,
        weights: str | Path,
        *,
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.5,
        device: str = "cuda:0",
        half: bool = True,
        max_det: int = 20,
        classes: Optional[Sequence[int]] = None,
        fallback_weights: Optional[str | Path] = None,
    ) -> None:
        try:
            from ultralytics import YOLO  # 延迟导入, 避免无 CUDA 环境直接失败
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "未安装 ultralytics, 请执行: pip install ultralytics"
            ) from exc

        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = str(device)
        self.half = bool(half)
        self.max_det = int(max_det)
        self.classes = list(classes) if classes else None
        self.fallback_weights = str(fallback_weights) if fallback_weights else None
        self.weights = self._resolve_weights(weights)
        self.model = YOLO(self.weights)
        self._warned_generic = False

    def _resolve_weights(self, weights: str | Path) -> str:
        path = Path(weights)
        if path.is_file():
            return str(path)
        if self.fallback_weights:
            fb = Path(self.fallback_weights)
            if fb.is_file():
                LOGGER.warning(
                    "微调权重 %s 不存在, 回退到 %s。注意: 通用 COCO 类别不含鱼, "
                    "该回退仅用于链路自检。",
                    path,
                    fb,
                )
                return str(fb)
            # ultralytics 可自动下载同名权重(如 yolo11n.pt)
            if fb.suffix == ".pt":
                LOGGER.warning("本地未找到权重, 交由 ultralytics 解析: %s", fb)
                return fb.name
        LOGGER.warning("权重不存在, 交由 ultralytics 解析: %s", path)
        return str(path)

    def detect(self, image: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half and self.device.startswith("cuda"),
            max_det=self.max_det,
            classes=self.classes,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float32)
        clses = boxes.cls.detach().cpu().numpy().astype(np.int32)
        if not self._warned_generic and Path(self.weights).name == "yolo11n.pt":
            LOGGER.warning(
                "当前使用通用 COCO 权重, 输出类别不是鱼类标签, 请勿用于指标验收。"
            )
            self._warned_generic = True
        for box, score, cls in zip(xyxy, scores, clses):
            detections.append(Detection(xyxy=box, score=float(score), cls=int(cls)))
        return detections


class OnnxDetector(BaseDetector):
    """ONNX / TensorRT 端侧检测器。

    仅做前处理/后处理封装, 便于在 Orin 上复用同一套上层逻辑。
    输入: 1x3xHxW float32, 输出: (N, 6) = [x1, y1, x2, y2, score, cls]。
    """

    name = "onnx"

    def __init__(
        self,
        engine_path: str | Path,
        *,
        imgsz: int = 640,
        conf: float = 0.35,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise ImportError("未安装 onnxruntime, 请执行: pip install onnxruntime-gpu") from exc

        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.session = ort.InferenceSession(
            str(engine_path),
            providers=list(providers) if providers else None,
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image: np.ndarray) -> List[Detection]:
        import cv2

        h, w = image.shape[:2]
        scale = min(self.imgsz / max(1, w), self.imgsz / max(1, h))
        resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        outputs = self.session.run(None, {self.input_name: blob[None]})[0]
        rows = np.asarray(outputs).reshape(-1, np.asarray(outputs).shape[-1])
        detections: List[Detection] = []
        for row in rows:
            score = float(row[4])
            if score < self.conf:
                continue
            box = row[:4].astype(np.float32) / scale
            detections.append(Detection(xyxy=box, score=score, cls=int(row[5]) if len(row) > 5 else 0))
        return detections


class MockDetector(BaseDetector):
    """合成检测器: 生成一条在画面内做正弦运动的"鱼"。

    用途: 在没有数据集与权重时验证快慢双环、调度与控制的完整链路。
    """

    name = "mock"

    def __init__(
        self,
        image_size: tuple[int, int] = (1280, 720),
        *,
        fps: float = 30.0,
        score: float = 0.85,
        box_size: float = 0.10,   # 相对图像高度
        period_s: float = 6.0,
        noise: float = 0.004,
        dropout_prob: float = 0.0,
        seed: int = 0,
        extra_targets: int = 0,
    ) -> None:
        self.width, self.height = int(image_size[0]), int(image_size[1])
        self.fps = float(fps)
        self.score = float(score)
        self.box_size = float(box_size)
        self.period_s = float(period_s)
        self.noise = float(noise)
        self.dropout_prob = float(dropout_prob)
        self.extra_targets = int(extra_targets)
        self.rng = np.random.default_rng(int(seed))
        self._t0: Optional[float] = None
        self._frame = 0

    def detect(self, image: np.ndarray) -> List[Detection]:
        if self._t0 is None:
            self._t0 = time.monotonic()
        t = self._frame / max(1e-6, self.fps)
        self._frame += 1

        if self.dropout_prob > 0 and self.rng.random() < self.dropout_prob:
            return []

        h = self.box_size * self.height * (1.0 + 0.15 * np.sin(2 * np.pi * t / (self.period_s * 2)))
        w = h * 2.4
        cx = self.width * 0.5 + 0.32 * self.width * np.sin(2 * np.pi * t / self.period_s)
        cy = self.height * 0.5 + 0.10 * self.height * np.cos(2 * np.pi * t / self.period_s)
        cx += self.rng.normal(0, self.noise * self.width)
        cy += self.rng.normal(0, self.noise * self.height)

        dets = [self._make(cx, cy, w, h, self.score)]
        for i in range(self.extra_targets):
            phase = (i + 1) * 1.7
            ex = self.width * 0.5 + 0.30 * self.width * np.sin(2 * np.pi * t / self.period_s + phase)
            ey = self.height * 0.5 + 0.18 * self.height * np.cos(2 * np.pi * t / (self.period_s * 1.3) + phase)
            dets.append(self._make(ex, ey, w * 0.9, h * 0.9, self.score * 0.8))
        return dets

    def _make(self, cx: float, cy: float, w: float, h: float, score: float) -> Detection:
        half_w, half_h = w * 0.5, h * 0.5
        box = np.array(
            [
                np.clip(cx - half_w, 0, self.width),
                np.clip(cy - half_h, 0, self.height),
                np.clip(cx + half_w, 0, self.width),
                np.clip(cy + half_h, 0, self.height),
            ],
            dtype=np.float32,
        )
        return Detection(xyxy=box, score=float(score), cls=0)


def build_detector(cfg, image_size: tuple[int, int], *, source=None) -> BaseDetector:
    """按配置构建检测器。

    Args:
        source: 可选帧源。`backend == "scenario"` 时从仿真场景读取真值框,
            用于把控制问题与检测问题解耦(见 `pandeng.sim`)。
    """
    backend = str(cfg.backend).lower()
    if backend == "scenario":
        if source is None or not hasattr(source, "scenario"):
            raise ValueError("backend=scenario 需要以 simulated 类型帧源作为输入")
        from ..sim.simulator import ScriptedDetector

        return ScriptedDetector(
            source.scenario,
            score=float(cfg.get_path("scenario.score", 0.88)),
            score_jitter=float(cfg.get_path("scenario.score_jitter", 0.05)),
            dropout_prob=float(cfg.get_path("scenario.dropout_prob", 0.0)),
            seed=int(cfg.get_path("scenario.seed", 0)),
        )
    if backend == "mock":
        return MockDetector(
            image_size=image_size,
            fps=float(cfg.get_path("mock.fps", 30.0)),
            score=float(cfg.get_path("mock.score", 0.85)),
            box_size=float(cfg.get_path("mock.box_size", 0.10)),
            dropout_prob=float(cfg.get_path("mock.dropout_prob", 0.0)),
            extra_targets=int(cfg.get_path("mock.extra_targets", 0)),
            seed=int(cfg.get_path("mock.seed", 0)),
        )
    if backend in {"onnx", "tensorrt"}:
        engine = cfg.get_path("engine") or cfg.get_path("weights")
        return OnnxDetector(
            engine,
            imgsz=int(cfg.imgsz),
            conf=float(cfg.conf),
        )
    if backend == "ultralytics":
        return YoloDetector(
            cfg.weights,
            imgsz=int(cfg.imgsz),
            conf=float(cfg.conf),
            iou=float(cfg.iou),
            device=str(cfg.device),
            half=bool(cfg.half),
            max_det=int(cfg.max_det),
            classes=cfg.get_path("classes", None),
            fallback_weights=cfg.get_path("fallback_weights", None),
        )
    raise ValueError(f"未知的检测器后端: {backend}")
