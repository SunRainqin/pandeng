"""记录器: 视频叠加、逐帧 CSV、事件 JSONL。

方案第 7 节要求记录"错误校正、模板污染、过期丢弃和慢链路故障", 因此除逐帧
CSV 外, 单独输出事件流 JSONL, 便于离线统计与人工复核。
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["RunRecorder"]

LOGGER = logging.getLogger(__name__)

FRAME_FIELDS: List[str] = [
    "sequence",
    "frame_id",
    "t",
    "visible",
    "track_id",
    "generation",
    "x1", "y1", "x2", "y2",
    "ex", "ey",
    "area_ratio", "area_growth",
    "det_score",
    "assoc_conf",
    "drift",
    "recovered",
    "group_size",
    "mode",
    "surge", "yaw", "heave",
    "reason",
    "n_detections",
    "n_tracks",
    "correction_status",
    "correction_applied",
    "correction_age_s",
    "correction_latency_s",
    "correction_score",
    "gate_reason",
    "bridge_latency_ms",
    "n_templates",
    "templates_frozen",
]


class RunRecorder:
    """把一次试验落盘。"""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        name: str = "run",
        save_video: bool = True,
        save_csv: bool = True,
        save_jsonl: bool = True,
        draw: bool = True,
        fps: float = 30.0,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.out_dir = Path(out_dir) / name
        self.name = name
        self.save_video = bool(save_video)
        self.save_csv = bool(save_csv)
        self.save_jsonl = bool(save_jsonl)
        self.draw = bool(draw)
        self.fps = float(fps)

        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._jsonl_file = None
        self._video_writer = None
        self._events: List[Dict] = []
        self._sequence: Optional[str] = None

        if not self.enabled:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.save_csv:
            self._csv_file = (self.out_dir / "frames.csv").open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=FRAME_FIELDS)
            self._csv_writer.writeheader()

        if self.save_jsonl:
            self._jsonl_file = (self.out_dir / "events.jsonl").open("w", encoding="utf-8")

        LOGGER.info("记录目录: %s", self.out_dir)

    # -- 逐帧 ---------------------------------------------------------------
    def log_frame(self, row: Dict[str, object]) -> None:
        if not self.enabled or self._csv_writer is None:
            return
        missing = [k for k in FRAME_FIELDS if k not in row]
        if missing:
            raise KeyError(f"记录行缺少字段: {missing}")
        self._csv_writer.writerow({k: _fmt(row.get(k)) for k in FRAME_FIELDS})

    # -- 事件 ---------------------------------------------------------------
    def log_event(self, kind: str, **payload) -> None:
        if not self.enabled or self._jsonl_file is None:
            return
        record = {"t": round(time.monotonic(), 5), "kind": kind}
        record.update(payload)
        self._events.append(record)
        self._jsonl_file.write(json.dumps(record, ensure_ascii=False, default=_default) + "\n")
        self._jsonl_file.flush()

    @property
    def sequence(self) -> Optional[str]:
        """当前序列标签; 未设置时为空字符串, 便于直接写入 CSV。"""
        return self._sequence

    # -- 视频 ---------------------------------------------------------------
    def start_sequence(self, name: Optional[str], index: int = 0) -> None:
        """切换到新序列: 关闭旧视频并重命名, 后续帧写入独立文件。

        每个视频单独一个 mp4, 而不是把 29 个序列拼成一条长视频 —— 拼接
        后的画面在序列边界没有任何视觉连续性, 无法用于人工复核。
        """
        # 必须先回收再改标签: 否则刚写完的上一段视频会被冠上下一段的名字。
        self._release_video()
        self._sequence = name
        if name:
            self.log_event("sequence_start", sequence=name, index=index)

    def _release_video(self) -> None:
        if self._video_writer is None:
            return
        self._video_writer.release()
        self._video_writer = None
        done = self.out_dir / "overlay.mp4"
        if done.is_file() and self._sequence:
            tag = _sanitize(self._sequence)
            try:
                done.replace(self.out_dir / f"overlay_{tag}.mp4")
            except OSError:  # pragma: no cover - 文件系统权限等
                LOGGER.warning("重命名叠加视频失败: %s", done)
        elif done.is_file():
            # 单一视频缺失序列标签时的兼容路径
            pass

    # -- 逐帧视频 -----------------------------------------------------------
    def write_frame(self, frame: np.ndarray) -> None:
        if not self.enabled or not (self.save_video and self.draw):
            return
        if self._video_writer is None:
            import cv2

            h, w = frame.shape[:2]
            path = self.out_dir / "overlay.mp4"
            self._video_writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
            )
        self._video_writer.write(frame)

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None
        self._release_video()


def _sanitize(name: str) -> str:
    """把 `val/video_001` 转为适合做文件名后缀的形式。"""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
COLOR_TRACK = (200, 200, 200)
COLOR_TARGET = (60, 220, 60)
COLOR_DRIFT = (40, 40, 240)
COLOR_CAND = (240, 180, 60)
COLOR_TEXT = (255, 255, 255)


def draw_overlay(
    frame: np.ndarray,
    *,
    state=None,
    tracks: Sequence = (),
    detections: Sequence = (),
    command=None,
    scheduler=None,
    extra: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    """在画面上绘制轨迹、目标框、控制量与慢链路状态。"""
    import cv2

    for track in tracks:
        if state is not None and track.track_id == state.track_id:
            continue
        x1, y1, x2, y2 = np.asarray(track.xyxy, dtype=int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_TRACK, 1)

    for det in detections:
        x1, y1, x2, y2 = np.asarray(det.xyxy, dtype=int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_CAND, 1)

    if state is not None:
        color = COLOR_DRIFT if state.drift_flag else COLOR_TARGET
        x1, y1, x2, y2 = np.asarray(state.bbox, dtype=int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 中央 50% 区域
        h, w = frame.shape[:2]
        cx1, cy1 = int(w * 0.25), int(h * 0.25)
        cx2, cy2 = int(w * 0.75), int(h * 0.75)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (120, 120, 60), 1)

        lines = [
            f"track={state.track_id} gen={state.generation} vis={int(state.visible)}",
            f"ex={state.ex:+.2f} ey={state.ey:+.2f} area={state.area_ratio:.3f} grow={state.area_growth:+.3f}",
            f"conf={state.association_confidence:.2f} drift={int(state.drift_flag)} "
            f"group={state.group_size}",
        ]
    else:
        lines = ["no target"]

    if command is not None:
        lines.append(
            f"mode={command.mode.value} surge={command.surge:+.2f} "
            f"yaw={command.yaw:+.2f} heave={command.heave:+.2f}"
        )
        lines.append(f"why={command.reason}")

    if scheduler is not None:
        lines.append(
            f"slow: req={scheduler.stats.submitted} done={scheduler.stats.completed} "
            f"to={scheduler.stats.timeouts} inflight={int(scheduler.inflight)}"
        )

    if extra:
        lines.extend(f"{k}={v}" for k, v in extra.items())

    y = 22
    for text in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
        y += 22
    return frame


def _fmt(value) -> object:
    if isinstance(value, (np.floating, float)):
        return round(float(value), 5)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return " ".join(f"{float(v):.2f}" for v in value.ravel())
    if isinstance(value, bool):
        return int(value)
    return value


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return str(value)
