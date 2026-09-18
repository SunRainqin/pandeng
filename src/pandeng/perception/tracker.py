"""ByteTrack 短时跟踪器。

对应方案第 2 节"短时跟踪 ByteTrack, 随检测更新"与第 3 节"YOLO 输出鱼框和
置信度, ByteTrack 关联连续帧"。

实现要点(与原始 ByteTrack 一致):
1. 高分检测做第一轮关联(使用 IoU 与检测分数的融合代价);
2. 低分检测做第二轮关联(仅用 IoU), 用于遮挡后恢复;
3. 未匹配的已跟踪轨迹转入 lost 状态, 保留 track_buffer 帧;
4. 未匹配高分检测在分数超过 new_track_thresh 时新建轨迹。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..types import Detection, Track, TrackState
from ..utils.geometry import iou_matrix, xyxy_to_xywh, xywh_to_xyxy
from ..utils.kalman import KalmanFilterXYAH

__all__ = ["ByteTracker", "STrack"]


class STrack:
    """单条轨迹的内部表示。

    `track_id` 由所属 `ByteTracker` 分配, 不使用类级全局计数器 —— 全局计数器
    会让同一进程内的多个跟踪器实例(多路视频、批量实验)共享编号, 日志与
    "目标代次"统计都会失真, 也让单元测试彼此耦合。
    """

    def __init__(
        self,
        xyxy: np.ndarray,
        score: float,
        kf: KalmanFilterXYAH,
        frame_id: int,
        track_id: int,
    ) -> None:
        self.track_id = int(track_id)

        self.kf = kf
        self.mean, self.covariance = kf.initiate(xyxy_to_xywh(xyxy))
        self.score = float(score)
        self.start_frame = int(frame_id)
        self.frame_id = int(frame_id)
        self.state = TrackState.TRACKED
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.is_activated = False
        self.history: List[np.ndarray] = []

    # -- 属性 ---------------------------------------------------------------
    @property
    def xyxy(self) -> np.ndarray:
        return xywh_to_xyxy(self.mean[:4])

    @property
    def velocity(self) -> np.ndarray:
        """框四个角的每帧位移, 由状态中的速度分量推导。"""
        vcx, vcy, va, vh = self.mean[4:8]
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array(
            [
                vcx - 0.5 * (va * h + a * vh),
                vcy - 0.5 * vh,
                vcx + 0.5 * (va * h + a * vh),
                vcy + 0.5 * vh,
            ],
            dtype=np.float32,
        )

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            xyxy=self.xyxy.astype(np.float32),
            score=self.score,
            state=self.state,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            velocity=self.velocity,
        )

    # -- 生命周期 -----------------------------------------------------------
    def predict(self) -> None:
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        if self.state == TrackState.TRACKED:
            self.hits = self.hits  # 命中次数只在 update 时增加

    def update(self, detection: Detection, frame_id: int) -> None:
        self.frame_id = int(frame_id)
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, xyxy_to_xywh(detection.xyxy)
        )
        self.score = float(detection.score)
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.TRACKED
        self.history.append(self.xyxy.astype(np.float32))
        if len(self.history) > 30:
            self.history.pop(0)

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


class ByteTracker:
    """ByteTrack 跟踪器封装。"""

    def __init__(self, cfg, image_size: Tuple[int, int]) -> None:
        """
        Args:
            cfg: `configs/default.yaml` 中的 `tracker` 段。
            image_size: (width, height)
        """
        self.track_high_thresh = float(cfg.track_high_thresh)
        self.track_low_thresh = float(cfg.track_low_thresh)
        self.new_track_thresh = float(cfg.new_track_thresh)
        self.match_thresh = float(cfg.match_thresh)
        self.track_buffer = int(cfg.track_buffer)
        self.min_box_area = float(cfg.min_box_area)
        self.width, self.height = int(image_size[0]), int(image_size[1])

        kalman_cfg = cfg.get_path("kalman", {})
        std_wp = float(kalman_cfg.get("std_weight_position", 0.05))
        std_wv = float(kalman_cfg.get("std_weight_velocity", 0.00625))
        self.kf = KalmanFilterXYAH(std_weight_position=std_wp, std_weight_velocity=std_wv)

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0
        self._next_id = 1

    def _allocate_id(self) -> int:
        track_id = self._next_id
        self._next_id += 1
        return track_id

    # -- 主接口 -------------------------------------------------------------
    @property
    def tracks(self) -> List[Track]:
        return [t.to_track() for t in self.tracked_stracks if t.state == TrackState.TRACKED]

    def update(
        self, detections: Sequence[Detection], frame_id: Optional[int] = None
    ) -> List[Track]:
        """用当前帧检测更新轨迹, 返回本帧处于 tracked 状态的轨迹。"""
        self.frame_id = self.frame_id + 1 if frame_id is None else int(frame_id)

        dets = [d for d in detections if self._valid(d)]
        high = [d for d in dets if d.score >= self.track_high_thresh]
        low = [d for d in dets if self.track_low_thresh <= d.score < self.track_high_thresh]

        for track in self.tracked_stracks + self.lost_stracks:
            track.predict()

        # 第一轮: 已跟踪轨迹 <-> 高分检测
        pool = [t for t in self.tracked_stracks if t.state == TrackState.TRACKED]
        pool += [t for t in self.lost_stracks if t.time_since_update <= self.track_buffer]
        matched, unmatched_tracks, unmatched_high = self._associate(
            pool, high, thresh=self.match_thresh, fuse_score=True
        )
        for ti, di in matched:
            pool[ti].update(high[di], self.frame_id)

        # 第二轮: 未匹配轨迹 <-> 低分检测(遮挡恢复)
        remaining = [pool[i] for i in unmatched_tracks]
        matched2, unmatched_tracks2, _ = self._associate(
            remaining, low, thresh=0.5, fuse_score=False
        )
        for ti, di in matched2:
            remaining[ti].update(low[di], self.frame_id)

        # 未匹配轨迹: tracked -> lost; 超过 buffer -> removed
        for i in unmatched_tracks2:
            track = remaining[i]
            if track.state == TrackState.TRACKED:
                track.mark_lost()
            elif track.state == TrackState.LOST and track.time_since_update > self.track_buffer:
                track.mark_removed()

        # 未匹配高分检测: 新建轨迹
        for di in unmatched_high:
            det = high[di]
            if det.score < self.new_track_thresh:
                continue
            track = STrack(det.xyxy, det.score, self.kf, self.frame_id, self._allocate_id())
            track.is_activated = True
            self.tracked_stracks.append(track)

        self._reindex()
        return self.tracks

    def reset(self) -> None:
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        self._next_id = 1

    def find_by_id(self, track_id: int) -> Optional[STrack]:
        for track in self.tracked_stracks + self.lost_stracks:
            if track.track_id == track_id:
                return track
        return None

    def force_reactivate(self, track_id: int, bbox: np.ndarray, score: float) -> Optional[Track]:
        """慢速环重捕后强制恢复某条轨迹(不新建 ID, 以便保持目标代次语义)。

        方案第 4.1 节要求"目标重现"时重新确认后恢复跟随, 这里通过复用原
        track_id 实现, 目标代次由上层递增。
        """
        track = self.find_by_id(track_id)
        if track is None:
            return None
        track.mean, track.covariance = self.kf.initiate(xyxy_to_xywh(bbox))
        track.score = float(score)
        track.state = TrackState.TRACKED
        track.time_since_update = 0
        track.hits += 1
        if track not in self.tracked_stracks:
            self.tracked_stracks.append(track)
        if track in self.lost_stracks:
            self.lost_stracks.remove(track)
        return track.to_track()

    # -- 内部 ---------------------------------------------------------------
    def _valid(self, det: Detection) -> bool:
        x1, y1, x2, y2 = np.asarray(det.xyxy, dtype=np.float64)
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            return False
        if (x2 - x1) * (y2 - y1) < self.min_box_area:
            return False
        return True

    def _associate(
        self,
        tracks: Sequence[STrack],
        detections: Sequence[Detection],
        thresh: float,
        fuse_score: bool,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """匈牙利匹配。代价 = 1 - IoU (可选与检测分数融合)。"""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.stack([t.xyxy for t in tracks])
        det_boxes = np.stack([np.asarray(d.xyxy, dtype=np.float32) for d in detections])
        ious = iou_matrix(track_boxes, det_boxes)

        if fuse_score:
            scores = np.array([d.score for d in detections], dtype=np.float32)[None, :]
            similarity = ious * scores
        else:
            similarity = ious
        cost = 1.0 - similarity

        rows, cols = linear_sum_assignment(cost)
        matched: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))
        for r, c in zip(rows, cols):
            if cost[r, c] > thresh or ious[r, c] <= 0.0:
                continue
            matched.append((int(r), int(c)))
            unmatched_tracks.discard(int(r))
            unmatched_dets.discard(int(c))
        return matched, sorted(unmatched_tracks), sorted(unmatched_dets)

    def _reindex(self) -> None:
        """收敛轨迹池: 合并/去重 tracked 与 lost, 剔除 removed。

        注意 lost 轨迹一旦在本帧重新关联上检测, 其状态会被置回 TRACKED,
        此时必须从 lost 池搬回 tracked 池; 否则该轨迹会在两个池子里都找
        不到, 表现为"目标 ID 突然换了"(曾经的隐蔽缺陷)。
        """
        tracked: List[STrack] = []
        lost: List[STrack] = []
        seen: set[int] = set()

        for track in self.tracked_stracks + self.lost_stracks:
            if track.track_id in seen:
                continue
            seen.add(track.track_id)
            if track.state == TrackState.REMOVED:
                continue
            if track.state == TrackState.TRACKED:
                tracked.append(track)
            elif track.time_since_update > self.track_buffer:
                track.mark_removed()
            else:
                lost.append(track)

        self.removed_stracks.extend(
            t for t in self.tracked_stracks + self.lost_stracks
            if t.state == TrackState.REMOVED
        )
        # 新建但尚未确认命中的轨迹只在本帧保留
        self.tracked_stracks = [
            t for t in tracked if t.time_since_update == 0 or t.hits > 1
        ]
        self.lost_stracks = lost
