"""Prepare contiguous validation/test sequences for detector-tracker evaluation.

本文件是 `pandeng` 项目内的权威副本(见 `scripts/dataset/README.md`), 与数据
目录下的同名脚本保持一致。依赖同目录的 `prepare_yolo_dataset.py`。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from prepare_yolo_dataset import (
    FRAME_RE,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    read_annotations,
    read_classes,
    source_video_size,
    split_videos,
    yolo_box,
)


def extract_all_frames(video: Path, destination: Path) -> list[Path]:
    with tempfile.TemporaryDirectory(prefix=f"{video.stem}-", dir=destination.parent) as temp:
        temp_dir = Path(temp)
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
                "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}", "-vsync", "0",
                str(temp_dir / "%04d.png"),
            ],
            check=True,
        )
        frames = sorted(temp_dir.glob("*.png"))
        output_frames = []
        for frame in frames:
            output = destination / f"{video.stem}-{frame.stem}.png"
            shutil.copy2(frame, output)
            output_frames.append(output)
        return output_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export complete val/test video sequences with frame-aligned YOLO labels."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Brackish 数据集根目录(含 annotations/ 与 dataset/videos/)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: dataset_tracking)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "dataset_tracking").resolve()
    if output == root or output == root / "dataset":
        raise ValueError("Refusing to write into the source dataset directory")

    classes = read_classes(root / "scripts" / "Brackish.names")
    annotations = read_annotations(root / "annotations" / "annotations_AAU")
    video_dir = root / "dataset" / "videos"
    videos = {video.stem: video for video in video_dir.rglob("*.avi")}
    assignment = split_videos(video_dir, annotations)
    by_video: dict[str, list[str]] = defaultdict(list)
    for filename in annotations:
        video_stem = FRAME_RE.match(filename).group("video")
        if assignment.get(video_stem) in ("val", "test"):
            by_video[video_stem].append(filename)

    records = []
    for video_stem in sorted(by_video):
        split = assignment[video_stem]
        video = videos[video_stem]
        image_dir = output / "images" / split / video_stem
        label_dir = output / "labels" / split / video_stem
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        source_width, source_height = source_video_size(video)
        frames = extract_all_frames(video, image_dir)
        frame_annotations = {name: values for name, values in annotations.items()}
        annotated_count = 0
        for image_path in frames:
            label_path = label_dir / f"{image_path.stem}.txt"
            values = frame_annotations.get(f"{image_path.stem}.png", [])
            label_path.write_text(
                "\n".join(
                    yolo_box(item, classes, source_width, source_height)
                    for item in values
                ) + ("\n" if values else "")
            )
            annotated_count += bool(values)
        records.append({
            "video": video_stem,
            "source": str(video.relative_to(root)),
            "split": split,
            "source_width": source_width,
            "source_height": source_height,
            "output_width": OUTPUT_WIDTH,
            "output_height": OUTPUT_HEIGHT,
            "frame_count": len(frames),
            "annotated_frame_count": annotated_count,
            "first_frame": frames[0].name if frames else None,
            "last_frame": frames[-1].name if frames else None,
        })

    output.mkdir(parents=True, exist_ok=True)
    (output / "classes.json").write_text(json.dumps(classes, indent=2) + "\n")
    (output / "sequences.json").write_text(json.dumps(records, indent=2) + "\n")
    with (output / "video_splits.txt").open("w") as stream:
        for split in ("val", "test"):
            stream.write(f"[{split}]\n")
            for item in records:
                if item["split"] == split:
                    stream.write(
                        f'{item["video"]}\t{item["source"]}\t'
                        f'{item["frame_count"]} frames\t'
                        f'{item["annotated_frame_count"]} annotated frames\n'
                    )
            stream.write("\n")
    print(f"Prepared {len(records)} continuous val/test videos at {output}")


if __name__ == "__main__":
    main()
