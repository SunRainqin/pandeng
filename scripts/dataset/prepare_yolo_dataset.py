"""Prepare a source-video-isolated YOLO dataset from the Brackish annotations.

The released AAU CSV files are frame-level splits.  This script deliberately
ignores those split names and assigns complete source videos to train/val/test
so that adjacent frames from one video cannot leak across splits.

本文件是 `pandeng` 项目内的权威副本(见 `scripts/dataset/README.md`), 与数据
目录下的同名脚本保持一致; 相对原版仅增加了一处 `Brackish.names` 回退查找,
使得数据集尚未下载时也能拿到类别表。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


FRAME_RE = re.compile(r"^(?P<video>.+)-(?P<frame>\d{4})\.png$")
SPLITS = ("train", "val", "test")
OUTPUT_WIDTH = 960
OUTPUT_HEIGHT = 540
# AAU/COCO annotations are defined on the resized frames produced by
# frameExtractor.py, not on the 1920x1080 AVI pixel coordinates.
LABEL_WIDTH = 960
LABEL_HEIGHT = 540

#: 类别表随项目分发; 数据目录下没有 `scripts/Brackish.names` 时回退到这里。
FALLBACK_NAMES = Path(__file__).resolve().parent / "Brackish.names"


def read_classes(path: Path) -> dict[str, int]:
    if not path.is_file() and FALLBACK_NAMES.is_file():
        path = FALLBACK_NAMES
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names:
        raise ValueError(f"No classes found in {path}")
    return {name: index for index, name in enumerate(names)}


def read_annotations(annotation_dir: Path) -> dict[str, list[tuple[str, float, float, float, float]]]:
    annotations: dict[str, list[tuple[str, float, float, float, float]]] = defaultdict(list)
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for path in sorted(annotation_dir.glob("*.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter=";"):
                key = tuple(row[field] for field in (
                    "Filename",
                    "Annotation tag",
                    "Upper left corner X",
                    "Upper left corner Y",
                    "Lower right corner X",
                    "Lower right corner Y",
                ))
                if key in seen:
                    continue
                seen.add(key)
                match = FRAME_RE.match(row["Filename"])
                if not match:
                    raise ValueError(f"Unexpected frame filename: {row['Filename']}")
                annotations[row["Filename"]].append(
                    (
                        row["Annotation tag"],
                        float(row["Upper left corner X"]),
                        float(row["Upper left corner Y"]),
                        float(row["Lower right corner X"]),
                        float(row["Lower right corner Y"]),
                    )
                )
    return annotations


def split_videos(video_dir: Path, annotations: dict[str, list[tuple]]) -> dict[str, str]:
    by_class: dict[str, list[str]] = defaultdict(list)
    for video in sorted(video_dir.rglob("*.avi")):
        stem = video.stem
        if any(name.startswith(stem + "-") for name in annotations):
            by_class[video.parent.name].append(stem)

    assignment: dict[str, str] = {}
    for stems in by_class.values():
        for index, stem in enumerate(sorted(stems)):
            fraction = (index + 1) / len(stems)
            split = "train" if fraction <= 0.70 else "val" if fraction <= 0.85 else "test"
            assignment[stem] = split
    return assignment


def yolo_box(
    annotation: tuple[str, float, float, float, float],
    classes: dict[str, int],
    source_width: int,
    source_height: int,
) -> str:
    label, left, top, right, bottom = annotation
    if label not in classes:
        raise ValueError(f"Unknown annotation class: {label}")
    if (source_width, source_height) != (1920, 1080):
        raise ValueError(
            f"Unexpected source video size {(source_width, source_height)}; "
            "update the label coordinate mapping explicitly."
        )
    left, right = sorted((
        max(0.0, min(float(LABEL_WIDTH), left)),
        max(0.0, min(float(LABEL_WIDTH), right)),
    ))
    top, bottom = sorted((
        max(0.0, min(float(LABEL_HEIGHT), top)),
        max(0.0, min(float(LABEL_HEIGHT), bottom)),
    ))
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bounding box for {label}: {(left, top, right, bottom)}")
    return (
        f"{classes[label]} {(left + right) / (2 * LABEL_WIDTH):.6f} "
        f"{(top + bottom) / (2 * LABEL_HEIGHT):.6f} "
        f"{width / LABEL_WIDTH:.6f} {height / LABEL_HEIGHT:.6f}"
    )


def extract_video_frames(video: Path, wanted: list[str], destination: Path) -> None:
    with tempfile.TemporaryDirectory(prefix=f"{video.stem}-", dir=destination.parent) as temp:
        temp_dir = Path(temp)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
             "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}", "-vsync", "0",
             str(temp_dir / "%04d.png")],
            check=True,
        )
        for filename in wanted:
            frame_number = int(FRAME_RE.match(filename).group("frame"))
            source = temp_dir / f"{frame_number:04d}.png"
            if not source.exists():
                raise FileNotFoundError(f"Frame {frame_number} not found in {video}")
            shutil.copy2(source, destination / filename)


def source_video_size(video: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
        check=True,
        capture_output=True,
        text=True,
    )
    width, height = (int(value) for value in result.stdout.strip().split(","))
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Brackish 数据集根目录(含 annotations/ 与 dataset/videos/)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "dataset_yolo").resolve()
    if output == root or output == root / "dataset":
        raise ValueError("Refusing to write into the source dataset directory")

    classes = read_classes(root / "scripts" / "Brackish.names")
    annotations = read_annotations(root / "annotations" / "annotations_AAU")
    video_dir = root / "dataset" / "videos"
    videos = {video.stem: video for video in video_dir.rglob("*.avi")}
    assignment = split_videos(video_dir, annotations)
    if not assignment:
        raise ValueError("No annotated source videos were found")

    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    by_video: dict[str, list[str]] = defaultdict(list)
    for filename in annotations:
        video_stem = FRAME_RE.match(filename).group("video")
        if video_stem in assignment:
            by_video[video_stem].append(filename)

    video_manifest = []
    for video_stem, filenames in sorted(by_video.items()):
        split = assignment[video_stem]
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        source_width, source_height = source_video_size(videos[video_stem])
        extract_video_frames(videos[video_stem], sorted(filenames), image_dir)
        for filename in filenames:
            label_path = label_dir / f"{Path(filename).stem}.txt"
            label_path.write_text(
                "\n".join(
                    yolo_box(item, classes, source_width, source_height)
                    for item in annotations[filename]
                ) + "\n"
            )
        video_manifest.append({
            "video": video_stem,
            "source": str(videos[video_stem].relative_to(root)),
            "source_class": videos[video_stem].parent.name,
            "split": split,
            "source_width": source_width,
            "source_height": source_height,
            "output_width": OUTPUT_WIDTH,
            "output_height": OUTPUT_HEIGHT,
            "annotated_frame_count": len(filenames),
        })

    yaml = [
        f"path: {output}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(classes)}",
        "names:",
        *[f"  - {name}" for name in classes],
    ]
    (output / "data.yaml").write_text("\n".join(yaml) + "\n")
    (output / "video_splits.json").write_text(json.dumps(video_manifest, indent=2) + "\n")
    with (output / "video_splits.txt").open("w") as stream:
        for split in SPLITS:
            stream.write(f"[{split}]\n")
            for item in video_manifest:
                if item["split"] == split:
                    stream.write(
                        f'{item["video"]}\t{item["source_class"]}\t'
                        f'{item["annotated_frame_count"]} frames\t{item["source"]}\n'
                    )
            stream.write("\n")
    print(f"Prepared {len(annotations)} frames from {len(assignment)} source videos at {output}")


if __name__ == "__main__":
    main()
