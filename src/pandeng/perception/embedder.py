"""视觉基座特征提取: DINOv3。

对应方案第 2 节"DINOv3 ViT-S/16, 约 21M 参数, 冻结骨干"与第 4 节
低频校正链路的目标外观特征提取。

约束:
- 骨干冻结, 任务期间不执行梯度训练(方案第 1 节);
- 只对有限数量候选区域、限制输入尺寸做前向(方案第 2 节)。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from ..utils.geometry import clip_boxes

__all__ = [
    "BaseEmbedder",
    "DinoV3Embedder",
    "MockEmbedder",
    "build_embedder",
]
LOGGER = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """特征提取接口。输出必须为 L2 归一化向量, 便于直接做余弦相似度。"""

    dim: int = 0
    name: str = "base"
    # 实际使用的骨干来源与"是否回退到替代骨干"。这两个字段会被写入
    # summary.json, 避免用错误的骨干跑出看起来正常的实验结论。
    backbone_source: str = "n/a"
    is_fallback: bool = False

    @abstractmethod
    def embed(self, image: np.ndarray, boxes: Sequence[np.ndarray]) -> np.ndarray:
        """提取若干区域的特征。

        Args:
            image: BGR 图像。
            boxes: 目标框列表, 格式 xyxy。

        Returns:
            形状 (len(boxes), dim) 的 float32 数组, 每行 L2 归一化。
        """

    def close(self) -> None:  # pragma: no cover
        pass

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9)


class DinoV3Embedder(BaseEmbedder):
    """DINOv3 ViT-S/16 冻结骨干特征。

    无网环境使用 `hf_dir` 指向 `scripts/download_dinov3.py` 产出的本地快照;
    否则会回退到 torchvision ViT 骨干, 此时 `is_fallback` 为 True, 该状态下的
    实验结论不可用于验收。
    """

    name = "dinov3"

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        model_name: str = "dinov3_vits16",
        hub_repo: str = "facebookresearch/dinov3",
        repo_dir: str | Path | None = None,
        hf_dir: str | Path | None = None,
        impl: str = "auto",
        device: str = "cuda:0",
        input_size: int = 224,
        fp16: bool = True,
        freeze_backbone: bool = True,
        allow_fallback: bool = True,
    ) -> None:
        import torch

        from .dinov3_loader import load_backbone

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or "cpu" in device else "cpu")
        self.input_size = int(input_size)
        self.fp16 = bool(fp16) and self.device.type == "cuda"

        result = load_backbone(
            model_name,
            hub_repo=hub_repo,
            repo_dir=repo_dir,
            hf_dir=hf_dir,
            weights=weights,
            impl=impl,
            freeze=freeze_backbone,
            allow_fallback=allow_fallback,
        )
        self.model = result.model
        self.backbone_source = result.source
        self.is_fallback = bool(result.is_fallback)
        self.load_detail = result.detail

        self.model.to(self.device).eval()
        self.dim = int(self._probe_dim())
        LOGGER.info(
            "外观骨干就绪: 来源=%s 维度=%d 设备=%s 回退=%s",
            self.backbone_source, self.dim, self.device, self.is_fallback,
        )
        if self.is_fallback:
            LOGGER.warning(
                "当前使用的是回退骨干(%s), 外观判别力与 DINOv3 不同。"
                "请勿据此得出实验组 B/C/D/E 的结论。",
                self.backbone_source,
            )

        # 归一化参数优先取上游 preprocessor 配置, 避免"换了预处理但代码仍用
        # ImageNet 统计量"这种静默退化。
        mean = result.image_mean or (0.485, 0.456, 0.406)
        std = result.image_std or (0.229, 0.224, 0.225)
        if result.image_mean is None and not self.is_fallback:
            LOGGER.info("上游未提供 preprocessor 归一化配置, 使用 ImageNet 统计量。")
        self._mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        self._std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)

    def _probe_dim(self) -> int:
        with self.torch.no_grad():
            dummy = self.torch.zeros(1, 3, self.input_size, self.input_size, device=self.device)
            out = self._forward(dummy)
        return int(out.shape[-1])

    # -- 前向 ---------------------------------------------------------------
    def _forward(self, batch):
        torch = self.torch
        if self.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model(batch)
            return out.float()
        return self.model(batch)

    def embed(self, image: np.ndarray, boxes: Sequence[np.ndarray]) -> np.ndarray:
        import cv2

        if len(boxes) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        boxes_arr = clip_boxes(np.asarray(boxes, dtype=np.float32), image.shape[1], image.shape[0])
        crops: List[np.ndarray] = []
        for x1, y1, x2, y2 in boxes_arr:
            crop = image[int(y1) : int(y2), int(x1) : int(x2)]
            if crop.size == 0:
                crop = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            crop = cv2.resize(crop, (self.input_size, self.input_size))
            crops.append(crop)

        batch = np.stack(crops).astype(np.float32) / 255.0
        batch = batch[:, :, :, ::-1].transpose(0, 3, 1, 2).copy()  # BGR -> RGB, NCHW
        tensor = self.torch.from_numpy(batch).to(self.device)
        tensor = (tensor - self._mean) / self._std

        with self.torch.no_grad():
            features = self._forward(tensor)
        if features.ndim > 2:
            features = features.flatten(1)
        return self._normalize(features.detach().cpu().numpy())


class MockEmbedder(BaseEmbedder):
    """无权重时的外观描述子(HSV 直方图 + 梯度方向直方图)。

    用于在缺少 DINOv3 权重时打通慢速环链路与单元测试。它的判别力弱于
    DINOv3, 不能用于指标验收, 仅作占位。
    """

    name = "mock"

    bins_h: int = 12
    bins_s: int = 8
    bins_v: int = 8
    bins_grad: int = 12

    def __init__(self, input_size: int = 64) -> None:
        self.input_size = int(input_size)
        self.dim = self.bins_h + self.bins_s + self.bins_v + self.bins_grad
        self.backbone_source = "mock:hsv+grad"

    def embed(self, image: np.ndarray, boxes: Sequence[np.ndarray]) -> np.ndarray:
        import cv2

        if len(boxes) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        boxes_arr = clip_boxes(np.asarray(boxes, dtype=np.float32), image.shape[1], image.shape[0])
        feats = []
        for x1, y1, x2, y2 in boxes_arr:
            crop = image[int(y1) : int(y2), int(x1) : int(x2)]
            if crop.size == 0:
                feats.append(np.zeros(self.dim, dtype=np.float32))
                continue
            crop = cv2.resize(crop, (self.input_size, self.input_size))
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hists = [
                cv2.calcHist([hsv], [0], None, [self.bins_h], [0, 180]).flatten(),
                cv2.calcHist([hsv], [1], None, [self.bins_s], [0, 256]).flatten(),
                cv2.calcHist([hsv], [2], None, [self.bins_v], [0, 256]).flatten(),
            ]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx**2 + gy**2)
            angle = (np.rad2deg(np.arctan2(gy, gx)) % 180.0) / 15.0
            hist_grad = np.histogram(
                angle, bins=self.bins_grad, range=(0, 12), weights=mag
            )[0]
            hists.append(hist_grad.astype(np.float32))
            vector = np.concatenate(hists).astype(np.float32)
            feats.append(vector)
        return self._normalize(np.stack(feats))


def build_embedder(cfg) -> BaseEmbedder:
    """按配置构建特征提取器。"""
    backend = str(cfg.backend).lower()
    if backend == "mock":
        return MockEmbedder(input_size=int(cfg.get_path("input_size", 64)))
    if backend == "dinov3":
        return DinoV3Embedder(
            cfg.get_path("weights", None),
            model_name=str(cfg.get_path("model", "dinov3_vits16")),
            hub_repo=str(cfg.get_path("hub_repo", "facebookresearch/dinov3")),
            repo_dir=cfg.get_path("repo_dir", None),
            hf_dir=cfg.get_path("hf_dir", None),
            impl=str(cfg.get_path("impl", "auto")),
            device=str(cfg.get_path("device", "cuda:0")),
            input_size=int(cfg.get_path("input_size", 224)),
            fp16=bool(cfg.get_path("fp16", True)),
            freeze_backbone=bool(cfg.get_path("freeze_backbone", True)),
            allow_fallback=bool(cfg.get_path("allow_fallback", True)),
        )
    raise ValueError(f"未知的特征提取后端: {backend}")
