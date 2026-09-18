"""单元测试: 多序列数据集回放。

覆盖三件事:

1. `SequenceDirSource` 能按视频目录切分序列, 并正确读取逐帧 YOLO 标注;
2. **无标注帧不等于负样本** —— 检测评估只对有标注的帧计分;
3. 进入新序列时跟踪器、目标记忆、控制器与调度器全部重置, `frame_id` 归零。
   这是"每个视频一个文件夹以保证视频内部连续性"能成立的前提: 连续性只在
   **视频内部**有意义, 视频之间必须彻底断开, 否则轨迹 ID 与外观模板会跨视频
   泄漏, 统计出来的目标切换次数与重捕时间都是假的。
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from pandeng.config import load_config
from pandeng.io.recorder import FRAME_FIELDS
from pandeng.io.video_source import SequenceDirSource, build_source
from pandeng.metrics import DetectionEvaluator, SessionMetrics
from pandeng.types import Detection

WIDTH, HEIGHT = 160, 120


# ---------------------------------------------------------------------------
# 构造测试数据集
# ---------------------------------------------------------------------------
def _write_label(path, boxes) -> None:
    """boxes: [(cls, cx, cy, w, h)] 归一化坐标; 空列表写空文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(" ".join(str(v) for v in row) for row in boxes)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _make_dataset(root, layout) -> None:
    """layout: {split: {video: [(annotated: bool), ...]}}"""
    for split, videos in layout.items():
        for video, flags in videos.items():
            for index, annotated in enumerate(flags):
                image = np.full((HEIGHT, WIDTH, 3), 40 + index * 10, dtype=np.uint8)
                image_path = root / "images" / split / video / f"frame_{index:04d}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_path), image)
                boxes = [(0, 0.5, 0.5, 0.25, 0.25)] if annotated else []
                _write_label(
                    root / "labels" / split / video / f"frame_{index:04d}.txt", boxes
                )


@pytest.fixture()
def dataset(tmp_path):
    _make_dataset(
        tmp_path,
        {
            "val": {
                "video_a": [True, True, False],
                "video_b": [True, False],
            },
            "test": {"video_c": [True]},
        },
    )
    return tmp_path


# ---------------------------------------------------------------------------
# SequenceDirSource
# ---------------------------------------------------------------------------
def test_sequence_source_discovers_videos_per_split(dataset):
    source = SequenceDirSource(dataset)
    assert source.num_sequences == 3
    assert [name for _, name, _ in source.sequences] == ["video_a", "video_b", "video_c"]
    assert [(split, name) for split, name, _ in source.sequences] == [
        ("val", "video_a"),
        ("val", "video_b"),
        ("test", "video_c"),
    ]


def test_sequence_source_filters_splits_and_limits_count(dataset):
    only_test = SequenceDirSource(dataset, splits=("test",))
    assert only_test.num_sequences == 1
    assert only_test.sequences[0][1] == "video_c"

    limited = SequenceDirSource(dataset, max_sequences=2)
    assert limited.num_sequences == 2


def test_sequence_id_changes_once_per_video_and_frames_stream(dataset):
    source = SequenceDirSource(dataset)
    seen: list[str] = []
    frames = 0
    while True:
        item = source.read()
        if item is None:
            break
        frames += 1
        if not seen or seen[-1] != source.sequence_id:
            seen.append(source.sequence_id)

    assert frames == 6, "应逐帧遍历全部序列, 不遗漏也不重复"
    assert seen == ["val/video_a", "val/video_b", "test/video_c"]


def test_ground_truth_is_none_for_unannotated_and_missing_labels(dataset):
    source = SequenceDirSource(dataset)
    flags: list[bool] = []
    while True:
        item = source.read()
        if item is None:
            break
        flags.append(source.frame_annotated)
        if source.frame_annotated:
            gt = source.ground_truth()
            assert gt is not None and len(gt) == 1
            cls, box = gt[0]
            assert cls == 0
            # 归一化中心 0.5/0.5, 边长 0.25 -> 像素框
            np.testing.assert_allclose(
                box, [WIDTH * 0.375, HEIGHT * 0.375, WIDTH * 0.625, HEIGHT * 0.625], atol=1e-4
            )
        else:
            assert source.ground_truth() is None

    # video_a: 2 有标注 + 1 无标注; video_b: 1 + 1; video_c: 1
    assert flags == [True, True, False, True, False, True]


def test_labelless_sequence_is_kept_for_continuity(tmp_path):
    """整段无标注的视频目录仍必须保留 —— 否则视频连续性就被破坏。"""
    _make_dataset(tmp_path, {"val": {"video_a": [True]}})
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image_path = tmp_path / "images" / "val" / "video_b" / "frame_0000.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), image)

    source = SequenceDirSource(tmp_path)
    assert source.num_sequences == 2
    while True:
        item = source.read()
        if item is None:
            break
        if source.sequence_id == "val/video_b":
            assert source.frame_annotated is False
            assert source.ground_truth() is None


def test_build_source_accepts_sequence_dir(dataset):
    cfg = load_config(
        overrides=[
            "source.type=sequence_dir",
            f"source.root={dataset}",
            "source.splits=val",
            "source.max_sequences=1",
        ]
    )
    source = build_source(cfg.source)
    assert isinstance(source, SequenceDirSource)
    assert source.num_sequences == 1
    source.close()


def test_build_source_accepts_path_alias(tmp_path):
    _make_dataset(tmp_path, {"val": {"video_a": [True]}})
    cfg = load_config(
        overrides=["source.type=sequence_dir", f"source.path={tmp_path}"]
    )
    assert isinstance(build_source(cfg.source), SequenceDirSource)


def test_build_source_requires_root():
    cfg = load_config(
        overrides=["source.type=sequence_dir", "source.root=null", "source.path=null"]
    )
    with pytest.raises(ValueError):
        build_source(cfg.source)


# ---------------------------------------------------------------------------
# 无标注帧不是负样本
# ---------------------------------------------------------------------------
def _det(box, score=0.9, cls=0) -> Detection:
    return Detection(xyxy=np.asarray(box, dtype=np.float32), score=score, cls=cls)


def test_detection_evaluator_skips_unannotated_frames():
    evaluator = DetectionEvaluator(iou_thresh=0.5)
    perfect = _det([10, 10, 40, 40])
    gt = [(0, np.array([10, 10, 40, 40], dtype=np.float32))]

    evaluator.update([perfect], gt)
    # 无标注帧上有一个"多余的"检测, 但它不算误检
    assert evaluator.update([perfect, _det([100, 100, 120, 120])], None) is None

    summary = evaluator.summary()
    assert summary["annotated_frames"] == 1
    assert summary["unannotated_frames"] == 1
    assert summary["detections_on_unannotated"] == 2
    assert (summary["tp"], summary["fp"], summary["fn"]) == (1, 0, 0)
    assert summary["precision"] == 1.0
    assert summary["recall"] == 1.0
    assert summary["sample_coverage_ratio"] == 0.5


def test_detection_evaluator_counts_tp_fp_fn():
    evaluator = DetectionEvaluator(iou_thresh=0.5)
    evaluator.update(
        [_det([10, 10, 40, 40]), _det([100, 100, 130, 130])],
        [(0, np.array([10, 10, 40, 40], dtype=np.float32))],
    )
    summary = evaluator.summary()
    assert (summary["tp"], summary["fp"], summary["fn"]) == (1, 1, 0)

    evaluator.update([], [(0, np.array([0, 0, 10, 10], dtype=np.float32))])
    summary = evaluator.summary()
    assert (summary["tp"], summary["fp"], summary["fn"]) == (1, 1, 1)
    assert summary["recall"] == pytest.approx(0.5)


def test_detection_evaluator_low_iou_is_miss_and_false_positive():
    evaluator = DetectionEvaluator(iou_thresh=0.5)
    evaluator.update(
        [_det([50, 50, 60, 60])],
        [(0, np.array([10, 10, 40, 40], dtype=np.float32))],
    )
    summary = evaluator.summary()
    assert (summary["tp"], summary["fp"], summary["fn"]) == (0, 1, 1)


def test_detection_evaluator_greedy_prefers_highest_score():
    evaluator = DetectionEvaluator(iou_thresh=0.5)
    gt = [(0, np.array([10, 10, 40, 40], dtype=np.float32))]
    evaluator.update([_det([10, 10, 40, 40], score=0.4), _det([10, 10, 40, 40], score=0.9)], gt)
    summary = evaluator.summary()
    # 只有一个真值框: 高分检测匹配成功, 低分检测成为误检
    assert (summary["tp"], summary["fp"], summary["fn"]) == (1, 1, 0)


def test_detection_evaluator_merge():
    left = DetectionEvaluator()
    right = DetectionEvaluator()
    gt = [(0, np.array([10, 10, 40, 40], dtype=np.float32))]
    left.update([_det([10, 10, 40, 40])], gt)
    right.update([], gt)
    left.merge(right)
    summary = left.summary()
    assert (summary["tp"], summary["fp"], summary["fn"]) == (1, 0, 1)
    assert summary["annotated_frames"] == 2


# ---------------------------------------------------------------------------
# 会话指标合并
# ---------------------------------------------------------------------------
def test_session_metrics_merge_preserves_ratios():
    left = SessionMetrics(name="a")
    right = SessionMetrics(name="b")
    now = time.monotonic()
    for offset in range(5):
        left.record_frame(
            now=now + offset * 0.1, visible=True, in_central_region=True, bridge_latency=0.01
        )
    for offset in range(5):
        right.record_frame(
            now=now + offset * 0.1, visible=False, in_central_region=False, bridge_latency=0.02
        )
    left.record_target_switch()
    left.merge(right)
    summary = left.summary()
    assert summary["frames"] == 10
    assert summary["target_switches"] == 1
    assert summary["visibility_ratio"] == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# 调度器序列纪元
# ---------------------------------------------------------------------------
class _DummyCorrector:
    """可控制耗时的假校正器: 用于制造"校正结果仍在飞"的局面。"""

    def __init__(self, latency_s: float = 0.0) -> None:
        self.latency_s = float(latency_s)
        self.calls = 0

    def correct(self, request, *, predicted_bbox=None):
        from pandeng.types import CorrectionResult, CorrectionStatus

        self.calls += 1
        if self.latency_s:
            time.sleep(self.latency_s)
        now = time.monotonic()
        return CorrectionResult(
            frame_id=request.frame_id,
            timestamp=request.timestamp,
            wall_time=now,
            status=CorrectionStatus.OK,
            target_track_id=request.target_track_id,
            generation=request.generation,
            confidence=0.9,
            latency_s=self.latency_s,
        )


def _scheduler(corrector=None, **overrides):
    """构建一个只依赖本文件辅助函数的调度器。"""
    from pandeng.config import Config
    from pandeng.scheduling.corrector_scheduler import SlowLoopScheduler

    cfg = load_config()
    data = cfg.scheduler.to_dict()
    for key_path, value in overrides.items():
        parts = key_path.split(".")
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return SlowLoopScheduler(Config(data), corrector or _DummyCorrector(), (WIDTH, HEIGHT))


def _state(frame_id: int) -> "TargetState":
    from pandeng.types import TargetState

    return TargetState(
        frame_id=frame_id,
        timestamp=time.monotonic(),
        bbox=np.array([40.0, 30.0, 80.0, 60.0], dtype=np.float32),
        track_id=1,
        visible=True,
        generation=1,
        score=0.9,
    )


def _detections():
    return [Detection(xyxy=np.array([40.0, 30.0, 80.0, 60.0], dtype=np.float32), score=0.9)]


def test_scheduler_begin_sequence_drops_stale_inflight_result():
    corrector = _DummyCorrector(latency_s=0.25)
    scheduler = _scheduler(corrector, interval_frames=1, warmup_skip_frames=0, queue_size=1)
    scheduler.start()
    try:
        image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        scheduler.maybe_submit(
            frame_id=1,
            timestamp=time.monotonic(),
            state=_state(1),
            detections=_detections(),
            image=image,
        )
        time.sleep(0.05)  # 让工作线程开始处理, 校正仍在飞
        scheduler.begin_sequence(1)
        assert scheduler.epoch == 1
        assert scheduler.poll() is None, "排队中的请求必须被丢弃"
        time.sleep(0.4)  # 等旧校正跑完
        assert scheduler.poll() is None, "上一序列的在飞结果不得进入新序列"
    finally:
        scheduler.stop()


def test_scheduler_request_carries_current_epoch():
    scheduler = _scheduler(interval_frames=1, warmup_skip_frames=0)
    scheduler.begin_sequence(7)
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    scheduler.maybe_submit(
        frame_id=1,
        timestamp=time.monotonic(),
        state=_state(1),
        detections=_detections(),
        image=image,
    )
    request = scheduler._queue.get_nowait()
    assert request.epoch == 7


def test_scheduler_begin_sequence_resets_trigger_state():
    scheduler = _scheduler(interval_frames=5, warmup_skip_frames=0)
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for frame_id in range(1, 4):
        scheduler.maybe_submit(
            frame_id=frame_id,
            timestamp=time.monotonic(),
            state=_state(frame_id),
            detections=_detections(),
            image=image,
        )
    scheduler.begin_sequence(1)
    # 新序列的 frame_id 从 1 重新开始, 周期节拍也必须跟着重锚
    assert (
        scheduler.maybe_submit(
            frame_id=1,
            timestamp=time.monotonic(),
            state=_state(1),
            detections=_detections(),
            image=image,
        )
        is None
    )


# ---------------------------------------------------------------------------
# 记录器序列字段
# ---------------------------------------------------------------------------
def test_recorder_declares_sequence_column():
    assert FRAME_FIELDS[0] == "sequence"


def test_recorder_rotates_video_per_sequence(tmp_path):
    from pandeng.io.recorder import RunRecorder

    recorder = RunRecorder(tmp_path, name="run", fps=10.0)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    recorder.start_sequence("val/video_a", 0)
    recorder.write_frame(frame)
    recorder.start_sequence("val/video_b", 1)
    recorder.write_frame(frame)
    recorder.close()

    files = sorted(p.name for p in (tmp_path / "run").glob("overlay*.mp4"))
    assert files == ["overlay_val_video_a.mp4", "overlay_val_video_b.mp4"]


# ---------------------------------------------------------------------------
# 端到端: 多序列回放
# ---------------------------------------------------------------------------
def _sequence_config(root, out_dir=None, **extra):
    overrides = [
        "source.type=sequence_dir",
        f"source.root={root}",
        "source.splits=val,test",
        "detector.backend=mock",
        "corrector.backend=mock",
    ]
    if out_dir is not None:
        # 测试不得污染仓库根目录下的 runs/
        overrides.append(f"recorder.dir={out_dir}")
    overrides.extend(f"{k}={v}" for k, v in extra.items())
    return load_config(overrides=overrides)


def test_pipeline_resets_state_between_sequences(tmp_path, dataset):
    from pandeng.pipeline.tracking_loop import TrackingPipeline

    cfg = _sequence_config(dataset, out_dir=tmp_path / "runs")
    pipeline = TrackingPipeline(cfg, run_name="test_seq", enable_recording=True)
    try:
        result = pipeline.run()
    finally:
        pipeline.close()

    # 三个序列各自独立统计
    names = result.summary["sequence_names"]
    assert names == ["val/video_a", "val/video_b", "test/video_c"]
    expected_frames = {"val/video_a": 3, "val/video_b": 2, "test/video_c": 1}
    for item in result.summary["sequences"]:
        assert item["frames"] == expected_frames[item["name"]]

    # frame_id 必须逐序列归零 —— 这是"视频内部连续、视频之间断开"的直接证据
    rows = (result.output_dir / "frames.csv").read_text(encoding="utf-8").strip().splitlines()
    seen: dict[str, list[int]] = {}
    for row in rows[1:]:
        fields = row.split(",")
        seen.setdefault(fields[0], []).append(int(fields[1]))
    assert set(seen) == set(names)
    for sequence, ids in seen.items():
        assert ids == list(range(1, len(ids) + 1)), f"{sequence} 的 frame_id 未归零: {ids}"

    # 换了序列就得换 track_id: 轨迹编号不能跨视频延续
    first_ids = {}
    for row in rows[1:]:
        fields = row.split(",")
        first_ids.setdefault(fields[0], int(fields[4]))
    assert set(first_ids.values()) == {1}

    # 调度器纪元随序列递增
    assert result.summary["num_sequences"] == 3


def test_pipeline_records_per_sequence_detection_scores(tmp_path, dataset):
    from pandeng.pipeline.tracking_loop import TrackingPipeline

    cfg = _sequence_config(dataset, out_dir=tmp_path / "runs")
    pipeline = TrackingPipeline(cfg, run_name="test_seq_det", enable_recording=False)
    try:
        result = pipeline.run()
    finally:
        pipeline.close()

    sequences = {item["name"]: item for item in result.summary["sequences"]}
    video_a = sequences["val/video_a"]["detection"]
    # video_a 的 3 帧里有 2 帧带标注, 1 帧没有
    assert video_a["annotated_frames"] == 2
    assert video_a["unannotated_frames"] == 1
    assert video_a["detections_on_unannotated"] > 0
    assert video_a["tp"] == 0, "mock 检测框与真值不重合, 不该计为命中"


def test_pipeline_sequence_accepts_max_frames_per_sequence(tmp_path, dataset):
    """max_frames 在 sequence_dir 下是"每个序列"的上限, 不是全局上限。"""
    from pandeng.pipeline.tracking_loop import TrackingPipeline

    cfg = _sequence_config(dataset, out_dir=tmp_path / "runs", **{"source.max_frames": 1})
    pipeline = TrackingPipeline(cfg, run_name="test_seq_cap", enable_recording=False)
    try:
        result = pipeline.run()
    finally:
        pipeline.close()

    assert result.summary["num_sequences"] == 3
    assert [item["frames"] for item in result.summary["sequences"]] == [1, 1, 1]
    assert result.summary["processed_frames"] == 3


def test_summary_json_persists_per_sequence_and_aggregate(tmp_path, dataset):
    """多序列结果必须落盘。只看 CLI stdout 等于没有可归档的评估结果。"""
    import json

    from pandeng.pipeline.tracking_loop import TrackingPipeline

    cfg = _sequence_config(dataset, out_dir=tmp_path / "runs")
    pipeline = TrackingPipeline(cfg, run_name="test_seq_json", enable_recording=False)
    try:
        result = pipeline.run()
    finally:
        pipeline.close()

    payload = json.loads((tmp_path / "runs" / "test_seq_json" / "summary.json").read_text())
    summary = payload["summary"]
    assert summary["num_sequences"] == 3
    assert [s["name"] for s in summary["sequences"]] == result.summary["sequence_names"]
    # 汇总是各序列之和, 不是最后一个序列的数字
    assert summary["frames"] == sum(s["frames"] for s in summary["sequences"])
    assert summary["frames"] == 6
    assert "success_criteria" in payload


def test_single_video_run_keeps_flat_overlay_name(tmp_path):
    """非序列帧源不能被序列逻辑污染: 仍然输出 overlay.mp4。"""
    import numpy as np

    from pandeng.io.recorder import RunRecorder

    recorder = RunRecorder(tmp_path, name="run", fps=10.0)
    recorder.write_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    recorder.close()
    assert (tmp_path / "run" / "overlay.mp4").is_file()
    assert not list((tmp_path / "run").glob("overlay_*.mp4"))


def test_sequence_run_does_not_expose_num_sequences_for_flat_sources(tmp_path):
    """视频帧源不应出现乘性的逐序列字段。"""
    from pandeng.config import load_config
    from pandeng.pipeline.tracking_loop import TrackingPipeline

    cfg = load_config(
        overrides=[
            "detector.backend=mock",
            "corrector.backend=mock",
            "source.type=simulated",
            "source.max_frames=20",
            f"recorder.dir={tmp_path / 'runs'}",
        ]
    )
    pipeline = TrackingPipeline(cfg, run_name="test_flat", enable_recording=False)
    try:
        summary = pipeline.run().summary
    finally:
        pipeline.close()
    assert "sequences" not in summary
    assert "num_sequences" not in summary


def test_detection_merge_keeps_per_class_detail():
    """汇总器默认关闭逐类别, 合并时必须继承开关, 否则汇总里反而没有明细。"""
    left = DetectionEvaluator(iou_thresh=0.5, per_class=False)
    right = DetectionEvaluator(iou_thresh=0.5, per_class=True)
    gt = [(3, np.array([10, 10, 40, 40], dtype=np.float32))]
    right.update([_det([10, 10, 40, 40], cls=3)], gt)
    left.merge(right)
    assert left.per_class is True
    summary = left.summary()
    assert "per_class" in summary
    assert summary["per_class"]["3"]["tp"] == 1
