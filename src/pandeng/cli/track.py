"""命令行入口。

用法示例:

```bash
# 1) 生成示例视频并跑通链路(无需数据集与权重)
python -m pandeng.cli.track --config configs/default.yaml \
    --override detector.backend=mock corrector.backend=mock \
               source.type=synthetic source.max_frames=300

# 2) 实验组 B: A + 每十帧 DINOv3 校正(叠加本地资源配置)
python -m pandeng.cli.track \
    --overlay configs/experiment_b.yaml \
    --overlay configs/dinov3_local.yaml \
    --source data/demo/demo.mp4
```
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from typing import Sequence

from ..config import load_config
from ..pipeline.tracking_loop import TrackingPipeline
from ..utils.logging_utils import setup_logging

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pandeng-track",
        description="快慢双环跟踪闭环(检测 + ByteTrack + DINOv3 低频校正 + 图像伺服)",
    )
    parser.add_argument("--config", default=None, help="主配置路径, 默认 configs/default.yaml")
    parser.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="叠加配置；可重复指定，按顺序合并(实验组后可叠加 dinov3_local.yaml)",
    )
    parser.add_argument(
        "--override",
        action="extend",
        nargs="+",
        default=[],
        # 注意: 必须用 action="extend"。若用默认的 store + nargs="*",
        # 多次给出 --override 时只有最后一组生效, 前一组会被静默丢弃。
        help="按 key=value 覆盖配置, 可多次给出, 如 controller.rate_hz=10",
    )
    parser.add_argument("--source", default=None, help="覆盖 source.path(sequence_dir 时即 source.root)")
    parser.add_argument(
        "--type",
        default=None,
        help="覆盖 source.type: video|camera|image_dir|sequence_dir|synthetic",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="sequence_dir 专用: 只跑指定划分, 如 val / test / val,test",
    )
    parser.add_argument(
        "--max-sequences", type=int, default=None, help="sequence_dir 专用: 最多跑几个序列"
    )
    parser.add_argument("--max-frames", type=int, default=None, help="整个运行最多处理帧数")
    parser.add_argument(
        "--max-frames-per-sequence",
        type=int,
        default=None,
        help="sequence_dir 专用: 每个序列最多处理帧数(调试用)",
    )
    parser.add_argument("--run-name", default=None, help="输出子目录名")
    parser.add_argument("--no-record", action="store_true", help="不落盘")
    parser.add_argument("--no-video", action="store_true", help="不写叠加视频")
    parser.add_argument("--log-level", default=None, help="日志级别")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides = list(args.override)
    if args.source:
        overrides.append(f"source.path={args.source}")
    if args.type:
        overrides.append(f"source.type={args.type}")
    if args.split:
        overrides.append(f"source.splits={args.split}")
    if args.max_sequences is not None:
        overrides.append(f"source.max_sequences={int(args.max_sequences)}")
    if args.max_frames_per_sequence is not None:
        overrides.append(f"source.max_frames={int(args.max_frames_per_sequence)}")
    if args.no_video:
        overrides.append("recorder.save_video=false")

    cfg = load_config(args.config, overlay=args.overlay, overrides=overrides)
    setup_logging(args.log_level or str(cfg.get_path("logging.level", "INFO")))

    pipeline = TrackingPipeline(cfg, run_name=args.run_name, enable_recording=not args.no_record)

    def _handle_signal(signum, _frame):  # pragma: no cover - 现场中断用
        print(f"\n收到信号 {signum}, 正在优雅退出...", file=sys.stderr)
        pipeline.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):  # pragma: no cover - 非主线程
            pass

    max_frames = args.max_frames
    source_type = str(cfg.get_path("source.type", "video")).lower()
    # sequence_dir 下 source.max_frames 是"每个序列"的上限, 由帧源自己截断;
    # 不能把它当成整个运行的上限, 否则跑完第一个序列就被掐断。
    if max_frames is None and source_type != "sequence_dir":
        configured = int(cfg.get_path("source.max_frames", 0))
        max_frames = configured if configured > 0 else None

    try:
        result = pipeline.run(max_frames=max_frames)
    finally:
        pipeline.close()

    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    _print_sequences(result.summary)
    criteria = result.success
    print("\n成功判据:")
    for key, ok in criteria.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {key}")
    return 0


def _print_sequences(summary: dict) -> None:
    """逐序列结果速览。"""
    sequences = summary.get("sequences")
    if not sequences:
        return
    print(f"\n序列结果 ({len(sequences)} 个):")
    header = f"  {'sequence':<28}{'frames':>8}{'vis%':>8}{'ctr%':>8}{'switch':>8}{'loss':>7}"
    detection = sequences[0].get("detection") is not None
    if detection:
        header += f"{'det_P':>8}{'det_R':>8}{'det_F1':>8}"
    print(header)
    for item in sequences:
        line = (
            f"  {str(item.get('name'))[:27]:<28}"
            f"{item.get('frames', 0):>8}"
            f"{float(item.get('visibility_ratio') or 0) * 100:>8.1f}"
            f"{float(item.get('central_ratio_when_visible') or 0) * 100:>8.1f}"
            f"{item.get('target_switches', 0):>8}"
            f"{item.get('losses', 0):>7}"
        )
        det = item.get("detection")
        if detection:
            line += "".join(
                f"{(det or {}).get(k) if (det or {}).get(k) is not None else float('nan'):>8.3f}"
                for k in ("precision", "recall", "f1")
            )
        print(line)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
