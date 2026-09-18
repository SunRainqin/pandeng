"""校验项目内的数据准备脚本与已生成的数据集一致。

`scripts/dataset/` 下的两个脚本是数据集准备的**权威副本**。它们必须能从
项目内独立运行，并且重新推导出的划分、帧清单与已生成数据集完全一致 ——
否则项目里带的那份就只是"看起来像"的文档，不能作为重建依据。

本测试只做纯计算部分（解析 CSV、按源视频划分、标签坐标换算），不抽帧，
因此不需要 ffmpeg，也只花几秒；数据集或视频目录不存在时自动跳过。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_SCRIPT_DIR = REPO_ROOT / "scripts" / "dataset"
DATA_ROOT = Path("/data2/zyq/datasets/brackish-dataset")
TRACKING_ROOT = DATA_ROOT / "dataset_tracking"

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "annotations" / "annotations_AAU").is_dir(),
    reason="本机没有 Brackish 源数据, 无法对照校验",
)


def _load(name: str, path: Path):
    """按文件路径加载脚本模块(`scripts/dataset` 不在 sys.path 上)。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prepare():
    if str(DATASET_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(DATASET_SCRIPT_DIR))
    yolo = _load("prepare_yolo_dataset_under_test", DATASET_SCRIPT_DIR / "prepare_yolo_dataset.py")
    tracking = _load(
        "prepare_tracking_dataset_under_test",
        DATASET_SCRIPT_DIR / "prepare_tracking_dataset.py",
    )
    return yolo, tracking


def test_scripts_are_self_contained(prepare):
    """策略二必须能 import 到策略一, 且类别表随项目分发。"""
    yolo, _ = prepare
    assert yolo.FRAME_RE.match("2019-03-19_17-02-04to2019-03-19_17-02-12_1-0001.png")
    assert (DATASET_SCRIPT_DIR / "Brackish.names").is_file()
    assert (DATASET_SCRIPT_DIR / "README.md").is_file()
    assert yolo.read_classes(DATA_ROOT / "scripts" / "Brackish.names") == {
        "fish": 0, "small_fish": 1, "crab": 2, "shrimp": 3, "jellyfish": 4, "starfish": 5,
    }


def test_class_names_fallback_when_data_dir_lacks_them(prepare):
    """数据目录还没准备好时, 类别表必须从项目内取到。

    这是把脚本放进项目的主要理由之一: 数据集下载之前就能拿到类别定义。
    """
    yolo, _ = prepare
    classes = yolo.read_classes(Path("/nonexistent/scripts/Brackish.names"))
    assert classes["fish"] == 0 and len(classes) == 6
    assert yolo.FALLBACK_NAMES == DATASET_SCRIPT_DIR / "Brackish.names"


def test_split_and_label_coordinates_match_shipped_manifest(prepare):
    """重新推导的划分/帧数必须与已生成的 `sequences.json` 逐条一致。"""
    if not TRACKING_ROOT.is_dir():
        pytest.skip("本机没有 dataset_tracking")

    yolo, _ = prepare
    classes = yolo.read_classes(DATA_ROOT / "scripts" / "Brackish.names")
    annotations = yolo.read_annotations(DATA_ROOT / "annotations" / "annotations_AAU")
    assignment = yolo.split_videos(DATA_ROOT / "dataset" / "videos", annotations)

    shipped = json.loads((TRACKING_ROOT / "sequences.json").read_text(encoding="utf-8"))
    assert shipped, "sequences.json 不应为空"

    expected = {}
    for filename in annotations:
        stem = yolo.FRAME_RE.match(filename).group("video")
        if assignment.get(stem) in ("val", "test"):
            expected.setdefault(stem, []).append(filename)

    for record in shipped:
        stem = record["video"]
        assert record["split"] == assignment[stem], f"{stem} 的划分不一致"
        assert record["annotated_frame_count"] == len(expected[stem]), f"{stem} 的标注帧数不一致"
        assert (record["output_width"], record["output_height"]) == (960, 540)

    assert {record["video"] for record in shipped} == set(expected)


def test_label_coordinates_use_960x540_not_the_avi_size(prepare):
    """标签坐标系是 960×540, 再按 1920×1080 缩一次就会错位。"""
    yolo, _ = prepare
    classes = yolo.read_classes(DATA_ROOT / "scripts" / "Brackish.names")
    # 满框 -> 中心 0.5/0.5, 边长 1.0; 若误按 1920×1080 缩放会变成 0.25 左右
    box = yolo.yolo_box(("fish", 0.0, 0.0, 960.0, 540.0), classes, 1920, 1080)
    assert box == f"{classes['fish']} 0.500000 0.500000 1.000000 1.000000"

    with pytest.raises(ValueError, match="Unexpected source video size"):
        yolo.yolo_box(("fish", 0.0, 0.0, 960.0, 540.0), classes, 1280, 720)


def test_tracking_manifest_matches_shipped_classes_and_counts():
    """已生成数据集的类别表与逐序列统计必须自洽(项目 README 引用了这些数字)。"""
    if not TRACKING_ROOT.is_dir():
        pytest.skip("本机没有 dataset_tracking")

    classes = json.loads((TRACKING_ROOT / "classes.json").read_text(encoding="utf-8"))
    assert classes == {
        "fish": 0, "small_fish": 1, "crab": 2, "shrimp": 3, "jellyfish": 4, "starfish": 5,
    }

    records = json.loads((TRACKING_ROOT / "sequences.json").read_text(encoding="utf-8"))
    by_split = defaultdict(int)
    for record in records:
        by_split[record["split"]] += 1
    assert dict(by_split) == {"val": 13, "test": 16}
    assert sum(r["frame_count"] for r in records) == 4965
    assert sum(r["annotated_frame_count"] for r in records) == 3669
    # 每个序列都必须有帧(否则序列本身没有意义)
    assert all(r["frame_count"] > 0 for r in records)
    # 标注密度在序列之间极不均匀: 稀疏的几个只标了 7/121 帧, 而 14/29 个
    # 序列是逐帧全标。这正是"无标注帧不能当负样本"的经验依据 —— 如果把空标签
    # 当负样本, 那 7/121 的序列会凭空产生上百个误检。
    assert any(r["annotated_frame_count"] < r["frame_count"] for r in records)
    assert min(r["annotated_frame_count"] / r["frame_count"] for r in records) < 0.10
    assert sum(1 for r in records if r["annotated_frame_count"] == r["frame_count"]) == 14


def test_yolo_split_statistics_match_documented_numbers(prepare):
    """策略一的划分统计必须与文档声明一致(README 与 scripts/dataset/README.md)。

    这条用例的价值在于: 它是"项目里那份副本确实能重建出已发布数据"的证据。
    任何人改动脚本里的划分比例、去重规则或坐标换算, 这里都会立刻失败。
    """
    yolo, _ = prepare
    classes = yolo.read_classes(DATA_ROOT / "scripts" / "Brackish.names")
    annotations = yolo.read_annotations(DATA_ROOT / "annotations" / "annotations_AAU")
    assignment = yolo.split_videos(DATA_ROOT / "dataset" / "videos", annotations)

    frames: dict[str, int] = defaultdict(int)
    boxes: dict[str, int] = defaultdict(int)
    videos: dict[str, set[str]] = defaultdict(set)
    for filename, values in annotations.items():
        stem = yolo.FRAME_RE.match(filename).group("video")
        split = assignment[stem]
        videos[split].add(stem)
        frames[split] += 1
        for item in values:
            yolo.yolo_box(item, classes, 1920, 1080)
            boxes[split] += 1

    assert {s: len(videos[s]) for s in yolo.SPLITS} == {"train": 60, "val": 13, "test": 16}
    assert {s: frames[s] for s in yolo.SPLITS} == {"train": 8775, "val": 1378, "test": 2291}
    assert {s: boxes[s] for s in yolo.SPLITS} == {"train": 23982, "val": 2199, "test": 9384}
    assert sum(boxes.values()) == 35565
