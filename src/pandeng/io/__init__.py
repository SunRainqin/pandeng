"""输入输出: 帧源与记录器。"""

from __future__ import annotations

from .recorder import RunRecorder, draw_overlay
from .video_source import (
    CameraSource,
    FrameSource,
    ImageDirSource,
    SequenceDirSource,
    SyntheticSource,
    VideoSource,
    build_source,
)

__all__ = [
    "CameraSource",
    "FrameSource",
    "ImageDirSource",
    "RunRecorder",
    "SequenceDirSource",
    "SyntheticSource",
    "VideoSource",
    "build_source",
    "draw_overlay",
]
