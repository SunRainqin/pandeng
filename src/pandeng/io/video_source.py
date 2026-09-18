"""图像输入源。

- `video`: 视频文件回放
- `camera`: 摄像头
- `image_dir`: 单个目录内的图像序列
- `sequence_dir`: 每个视频一个子目录的序列集合(逐帧标注可选), 用于
  连续序列上的检测—跟踪评估
- `synthetic` / `simulated`: 合成画面与闭环仿真
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FrameSource",
    "VideoSource",
    "CameraSource",
    "ImageDirSource",
    "SequenceDirSource",
    "SyntheticSource",
    "build_source",
]

LOGGER = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class FrameSource(ABC):
    """帧源接口。

    多序列源额外提供 `sequence_id` / `sequence_index` / `frame_annotated` /
    `ground_truth()`; 单序列源保持默认值(None / 0 / False), 上层据此判断是否
    需要按序列重置状态。
    """

    width: int = 0
    height: int = 0
    fps: float = 30.0

    #: 当前序列标识; 单序列源为 None。值发生变化即表示进入了新序列。
    sequence_id: Optional[str] = None
    #: 当前序列序号(从 0 开始)
    sequence_index: int = 0
    #: 当前帧是否有可用标注。**无标注的帧不等于负样本**, 评估时必须跳过。
    frame_annotated: bool = False

    @abstractmethod
    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        """返回 (BGR 图像, 采集时间 monotonic); 全部结束时返回 None。"""

    def ground_truth(self) -> Optional[List[Tuple[int, np.ndarray]]]:
        """当前帧的真值框 `[(类号, xyxy), ...]`; 无标注时返回 None。"""
        return None

    def sequence_meta(self) -> Dict[str, object]:
        """当前序列的元信息(写入记录, 便于回溯)。"""
        return {}

    def close(self) -> None:  # pragma: no cover
        pass

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()



class VideoSource(FrameSource):
    """视频文件回放。"""

    def __init__(
        self,
        path: str | Path,
        *,
        loop: bool = False,
        max_frames: int = 0,
        realtime_pacing: bool = False,
        fallback_size: Tuple[int, int] = (1280, 720),
    ) -> None:
        import cv2

        self.cv2 = cv2
        self.path = Path(path)
        self.loop = bool(loop)
        self.max_frames = int(max_frames)
        self.realtime_pacing = bool(realtime_pacing)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"视频不存在: {self.path}。可执行 `python scripts/make_demo_video.py` 生成示例视频。"
            )
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开视频: {self.path}")

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or fallback_size[0]
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or fallback_size[1]
        src_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.fps = src_fps if 1.0 < src_fps < 240.0 else 30.0
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.index = 0
        self._start = time.monotonic()

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        if 0 < self.max_frames <= self.index:
            return None
        ok, frame = self.capture.read()
        if not ok:
            if not self.loop:
                return None
            self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            self.index = 0
            ok, frame = self.capture.read()
            if not ok:
                return None
        timestamp = time.monotonic()
        if self.realtime_pacing:
            target = self._start + self.index / self.fps
            wait = target - timestamp
            if wait > 0:
                time.sleep(min(wait, 1.0))
            timestamp = time.monotonic()
        self.index += 1
        return frame, timestamp

    def close(self) -> None:
        self.capture.release()


class CameraSource(FrameSource):
    """摄像头实时采集。"""

    def __init__(
        self,
        index: int = 0,
        *,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        max_frames: int = 0,
        backend: Optional[str] = None,
    ) -> None:
        import cv2

        self.cv2 = cv2
        self.capture = (
            cv2.VideoCapture(int(index), getattr(cv2, backend))
            if backend
            else cv2.VideoCapture(int(index))
        )
        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开摄像头 {index}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.capture.set(cv2.CAP_PROP_FPS, float(fps))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        self.fps = float(fps)
        self.max_frames = int(max_frames)
        self.index = 0

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        if 0 < self.max_frames <= self.index:
            return None
        ok, frame = self.capture.read()
        if not ok:
            return None
        self.index += 1
        return frame, time.monotonic()

    def close(self) -> None:
        self.capture.release()


class ImageDirSource(FrameSource):
    """单个目录内的图像序列。"""

    SUFFIXES = IMAGE_SUFFIXES

    def __init__(
        self,
        directory: str | Path,
        *,
        fps: float = 30.0,
        max_frames: int = 0,
        loop: bool = False,
    ) -> None:
        import cv2

        self.cv2 = cv2
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise NotADirectoryError(f"目录不存在: {self.directory}")
        self.files: List[Path] = sorted(
            p for p in self.directory.iterdir() if p.suffix.lower() in self.SUFFIXES
        )
        if not self.files:
            raise FileNotFoundError(f"目录中未找到图像: {self.directory}")
        self.fps = float(fps)
        self.loop = bool(loop)
        self.max_frames = int(max_frames)
        self.index = 0
        sample = cv2.imread(str(self.files[0]))
        self.height, self.width = (sample.shape[0], sample.shape[1]) if sample is not None else (720, 1280)

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        if 0 < self.max_frames <= self.index:
            return None
        if self.index >= len(self.files):
            if not self.loop:
                return None
            self.index = 0
        frame = self.cv2.imread(str(self.files[self.index]))
        self.index += 1
        if frame is None:
            return None
        return frame, time.monotonic()

    def close(self) -> None:
        pass


class SequenceDirSource(FrameSource):
    """每个视频一个子目录的序列集合, 逐帧遍历全部序列。

    布局(与 `dataset_tracking` 一致):

        <root>/<image_subdir>/<split>/<video>/<frame>.png
        <root>/<label_subdir>/<split>/<video>/<frame>.txt   # 可选, 逐帧对齐

    标注文件为 YOLO 归一化格式 `cls cx cy w h`, 空文件表示该帧**无标注**。
    注意: 无标注帧不能当作"无目标"负样本(漏标帧会被误判成误检), 因此
    `ground_truth()` 只对有标注的帧返回真值, `frame_annotated` 为 False 的帧
    必须从检测评估中排除。

    每个视频目录构成一个独立序列: `sequence_id` 在进入新视频时变化, 上层据此
    重置跟踪器、目标记忆与控制器状态。**这一点是必需的** —— 否则轨迹 ID、
    目标代次与外观模板会跨视频泄漏, 视频内部的连续性保证也就失去意义。
    """

    def __init__(
        self,
        root: str | Path,
        *,
        splits: Sequence[str] = ("val", "test"),
        image_subdir: str = "images",
        label_subdir: Optional[str] = "labels",
        fps: float = 30.0,
        max_frames: int = 0,
        max_sequences: int = 0,
        realtime_pacing: bool = False,
        require_labels: bool = False,
    ) -> None:
        import cv2

        self.cv2 = cv2
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise NotADirectoryError(f"序列根目录不存在: {self.root}")

        self.fps = float(fps)
        self.max_frames = int(max_frames)
        self.max_sequences = int(max_sequences)
        self.realtime_pacing = bool(realtime_pacing)
        self.label_subdir = label_subdir

        self.sequences = self._discover(splits, image_subdir, label_subdir, require_labels)
        if not self.sequences:
            raise FileNotFoundError(
                f"在 {self.root} 下未找到匹配的序列。期望布局: "
                f"{image_subdir}/<split>/<video>/*"
            )

        self.sequence_index = -1
        self.sequence_id: Optional[str] = None
        self.frame_annotated = False
        self._frames: List[Path] = []
        self._labels: Optional[Path] = None
        self._index = 0
        self._gt: Optional[List[Tuple[int, np.ndarray]]] = None
        self._start: Optional[float] = None

        first = cv2.imread(str(self.sequences[0][2][0])) if self.sequences[0][2] else None
        self.height, self.width = (
            (first.shape[0], first.shape[1]) if first is not None else (540, 960)
        )
        LOGGER.info(
            "序列集合就绪: %d 个序列, 根目录 %s", len(self.sequences), self.root
        )

    # -- 发现 ---------------------------------------------------------------
    def _discover(
        self,
        splits: Sequence[str],
        image_subdir: str,
        label_subdir: Optional[str],
        require_labels: bool,
    ) -> List[Tuple[str, str, List[Path]]]:
        found: List[Tuple[str, str, List[Path]]] = []
        for split in splits:
            image_root = self.root / image_subdir / split
            if not image_root.is_dir():
                LOGGER.warning("split 目录不存在, 跳过: %s", image_root)
                continue
            for video_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
                frames = sorted(
                    p for p in video_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
                )
                if not frames:
                    LOGGER.warning("序列目录为空, 跳过: %s", video_dir)
                    continue
                if require_labels and label_subdir:
                    label_dir = self.root / label_subdir / split / video_dir.name
                    if not label_dir.is_dir():
                        LOGGER.warning("缺少标注目录, 跳过: %s", label_dir)
                        continue
                found.append((split, video_dir.name, frames))
        if self.max_sequences > 0:
            found = found[: self.max_sequences]
        return found

    # -- 序列切换 -----------------------------------------------------------
    def _advance_sequence(self) -> bool:
        """切换到下一个序列; 无更多序列时返回 False。"""
        self.sequence_index += 1
        if self.sequence_index >= len(self.sequences):
            return False
        split, name, frames = self.sequences[self.sequence_index]
        self.sequence_id = f"{split}/{name}"
        self._frames = frames
        self._index = 0
        self._start = None
        self._labels = None
        if self.label_subdir:
            candidate = self.root / self.label_subdir / split / name
            self._labels = candidate if candidate.is_dir() else None
        LOGGER.info(
            "[%d/%d] 序列 %s: %d 帧%s",
            self.sequence_index + 1,
            len(self.sequences),
            self.sequence_id,
            len(frames),
            "" if self._labels else " (无标注)",
        )
        return True

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        if self._index >= len(self._frames):
            if not self._advance_sequence():
                return None
        if 0 < self.max_frames <= self._index:
            # 当前序列截断, 直接进入下一个序列
            self._index = len(self._frames)
            return self.read()

        path = self._frames[self._index]
        self._index += 1
        frame = self.cv2.imread(str(path))
        if frame is None:
            LOGGER.warning("无法读取图像, 跳过: %s", path)
            return self.read()

        self._gt = self._load_labels(path)
        self.frame_annotated = self._gt is not None

        now = time.monotonic()
        if self.realtime_pacing:
            if self._start is None:
                self._start = now
            target = self._start + (self._index - 1) / max(1e-6, self.fps)
            wait = target - now
            if wait > 0:
                time.sleep(min(wait, 1.0))
            now = time.monotonic()
        return frame, now

    # -- 标注 ---------------------------------------------------------------
    def _load_labels(self, image_path: Path) -> Optional[List[Tuple[int, np.ndarray]]]:
        """读取逐帧 YOLO 标注。无标注文件或空文件 -> None。"""
        if self._labels is None:
            return None
        label_path = self._labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            return None
        try:
            lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines()]
        except OSError:
            return None
        boxes: List[Tuple[int, np.ndarray]] = []
        height, width = float(self.height), float(self.width)
        for line in lines:
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            boxes.append(
                (
                    cls,
                    np.array(
                        [
                            (cx - bw * 0.5) * width,
                            (cy - bh * 0.5) * height,
                            (cx + bw * 0.5) * width,
                            (cy + bh * 0.5) * height,
                        ],
                        dtype=np.float32,
                    ),
                )
            )
        return boxes or None

    def ground_truth(self) -> Optional[List[Tuple[int, np.ndarray]]]:
        return self._gt

    def sequence_meta(self) -> Dict[str, object]:
        return {
            "sequence": self.sequence_id,
            "index": self.sequence_index,
            "frames": len(self._frames),
            "has_labels": self._labels is not None,
            "source_frame": str(self._frames[self._index - 1].name)
            if 0 < self._index <= len(self._frames)
            else None,
        }

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    def close(self) -> None:
        pass


class SyntheticSource(FrameSource):
    """合成画面: 与 MockDetector 的"鱼"轨迹一致, 便于链路自检。"""

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        max_frames: int = 0,
        period_s: float = 6.0,
        box_size: float = 0.10,
        extra_targets: int = 0,
        seed: int = 0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.max_frames = int(max_frames)
        self.period_s = float(period_s)
        self.box_size = float(box_size)
        self.extra_targets = int(extra_targets)
        self.rng = np.random.default_rng(int(seed))
        self.index = 0

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        import cv2

        if 0 < self.max_frames <= self.index:
            return None
        t = self.index / max(1e-6, self.fps)
        self.index += 1

        # 背景: 渐变水色 + 噪声, 模拟水下光照
        yy = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None]
        base = np.zeros((self.height, self.width, 3), dtype=np.float32)
        base[:, :, 0] = 120.0 - 60.0 * yy + 8.0
        base[:, :, 1] = 90.0 - 40.0 * yy + 6.0
        base[:, :, 2] = 40.0 - 15.0 * yy + 4.0
        base += self.rng.normal(0.0, 3.0, base.shape).astype(np.float32)
        frame = np.clip(base, 0, 255).astype(np.uint8)

        cv2.ellipse(
            frame,
            (int(self.width * 0.5), int(self.height * 0.42)),
            (int(self.width * 0.42), int(self.height * 0.30)),
            0, 0, 360, (70, 55, 30), -1,
        )

        for box in self._boxes(t):
            x1, y1, x2, y2 = box.astype(int)
            cv2.ellipse(
                frame,
                ((x1 + x2) // 2, (y1 + y2) // 2),
                (max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2)),
                0, 0, 360, (60, 180, 235), -1,
            )
            cv2.ellipse(
                frame,
                ((x1 + x2) // 2 + max(1, (x2 - x1) // 6), (y1 + y2) // 2),
                (max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)),
                0, 0, 360, (30, 30, 30), -1,
            )
        return frame, time.monotonic()

    def _boxes(self, t: float) -> List[np.ndarray]:
        boxes: List[np.ndarray] = []
        h = self.box_size * self.height * (1.0 + 0.15 * np.sin(2 * np.pi * t / (self.period_s * 2)))
        w = h * 2.4
        cx = self.width * 0.5 + 0.32 * self.width * np.sin(2 * np.pi * t / self.period_s)
        cy = self.height * 0.5 + 0.10 * self.height * np.cos(2 * np.pi * t / self.period_s)
        boxes.append(np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32))
        for i in range(self.extra_targets):
            phase = (i + 1) * 1.7
            ecx = self.width * 0.5 + 0.30 * self.width * np.sin(2 * np.pi * t / self.period_s + phase)
            ecy = self.height * 0.5 + 0.18 * self.height * np.cos(
                2 * np.pi * t / (self.period_s * 1.3) + phase
            )
            boxes.append(
                np.array(
                    [ecx - w * 0.45, ecy - h * 0.45, ecx + w * 0.45, ecy + h * 0.45],
                    dtype=np.float32,
                )
            )
        return boxes

    def close(self) -> None:
        pass


def _split_list(value) -> List[str]:
    """把 `val,test` / `[val, test]` 统一成列表。"""
    if value is None:
        return ["val", "test"]
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return [str(s) for s in value]


def build_source(cfg) -> FrameSource:
    """按配置构建帧源。"""
    kind = str(cfg.get_path("type", "video")).lower()
    if kind == "sequence_dir":
        # root 与 path 互为别名, 便于直接用 --source 指定
        root = cfg.get_path("root", None) or cfg.get_path("path", None)
        if not root:
            raise ValueError("source.type=sequence_dir 需要 source.root(或 source.path)")
        return SequenceDirSource(
            root,
            splits=_split_list(cfg.get_path("splits", None)),
            image_subdir=str(cfg.get_path("image_subdir", "images")),
            label_subdir=cfg.get_path("label_subdir", "labels"),
            fps=float(cfg.get_path("fps", 30.0)),
            max_frames=int(cfg.get_path("max_frames", 0)),
            max_sequences=int(cfg.get_path("max_sequences", 0)),
            realtime_pacing=bool(cfg.get_path("realtime_pacing", False)),
            require_labels=bool(cfg.get_path("require_labels", False)),
        )
    if kind == "video":
        return VideoSource(
            cfg.get_path("path", "data/demo/demo.mp4"),
            loop=bool(cfg.get_path("loop", False)),
            max_frames=int(cfg.get_path("max_frames", 0)),
            realtime_pacing=bool(cfg.get_path("realtime_pacing", False)),
        )
    if kind == "camera":
        return CameraSource(
            int(cfg.get_path("index", 0)),
            width=int(cfg.get_path("width", 1280)),
            height=int(cfg.get_path("height", 720)),
            fps=float(cfg.get_path("fps", 30.0)),
            max_frames=int(cfg.get_path("max_frames", 0)),
        )
    if kind == "image_dir":
        return ImageDirSource(
            cfg.get_path("path", "data/images"),
            fps=float(cfg.get_path("fps", 30.0)),
            max_frames=int(cfg.get_path("max_frames", 0)),
            loop=bool(cfg.get_path("loop", False)),
        )
    if kind == "synthetic":
        return SyntheticSource(
            width=int(cfg.get_path("width", 1280)),
            height=int(cfg.get_path("height", 720)),
            fps=float(cfg.get_path("fps", 30.0)),
            max_frames=int(cfg.get_path("max_frames", 0)),
            extra_targets=int(cfg.get_path("mock.extra_targets", 0)),
            seed=int(cfg.get_path("mock.seed", 0)),
        )
    if kind == "simulated":
        from ..sim.simulator import SimulatedScenario, SimulatedSource

        scenario = SimulatedScenario(
            width=int(cfg.get_path("width", 1280)),
            height=int(cfg.get_path("height", 720)),
            fps=float(cfg.get_path("fps", 30.0)),
            fov_h_deg=float(cfg.get_path("sim.fov_h_deg", 70.0)),
            fov_v_deg=float(cfg.get_path("sim.fov_v_deg", 45.0)),
            max_yaw_rate=float(cfg.get_path("sim.max_yaw_rate", 0.6)),
            max_speed=float(cfg.get_path("sim.max_speed", 0.8)),
            max_depth_rate=float(cfg.get_path("sim.max_depth_rate", 0.35)),
            fish_size_m=float(cfg.get_path("sim.fish_size_m", 0.35)),
            n_fish=int(cfg.get_path("sim.n_fish", 1)),
            seed=int(cfg.get_path("mock.seed", 0)),
            noise_std_px=float(cfg.get_path("sim.noise_std_px", 1.5)),
            dropout_prob=float(cfg.get_path("sim.dropout_prob", 0.0)),
            initial_distance=float(cfg.get_path("sim.initial_distance", 6.0)),
            initial_bearing_deg=float(cfg.get_path("sim.initial_bearing_deg", 18.0)),
            initial_depth_offset=float(cfg.get_path("sim.initial_depth_offset", 1.0)),
        )
        return SimulatedSource(
            scenario,
            max_frames=int(cfg.get_path("max_frames", 0)),
            realtime_pacing=bool(cfg.get_path("realtime_pacing", True)),
        )
    raise ValueError(f"未知的帧源类型: {kind}")
