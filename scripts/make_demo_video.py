#!/usr/bin/env python
"""生成示例视频, 便于在没有数据集与权重时跑通链路。

画面中的"鱼"与 `MockDetector` / `SyntheticSource` 使用同一条运动规律,
因此 `--type video` 与 `--type synthetic` 得到的结果应当一致。

用法:
    python scripts/make_demo_video.py --out data/demo/demo.mp4 --seconds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pandeng.io.video_source import SyntheticSource  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="生成水下目标示例视频")
    parser.add_argument("--out", default="data/demo/demo.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--extra-targets", type=int, default=0, help="额外干扰目标数")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    total = int(args.seconds * args.fps)
    source = SyntheticSource(
        width=args.width,
        height=args.height,
        fps=args.fps,
        max_frames=total,
        extra_targets=args.extra_targets,
        seed=args.seed,
    )

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        print(f"无法创建视频: {out_path}", file=sys.stderr)
        return 1

    count = 0
    while True:
        item = source.read()
        if item is None:
            break
        frame, _ = item
        writer.write(frame)
        count += 1

    writer.release()
    source.close()
    print(f"已生成 {out_path} ({count} 帧, {args.fps:.0f}fps, {args.width}x{args.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
