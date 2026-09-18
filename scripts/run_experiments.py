#!/usr/bin/env python
"""批量运行实验组 A~E 并汇总指标。

对应方案第 7 节:
"各组使用固定数据划分、相同试验条件及独立评价; 记录错误校正、模板污染、
 过期丢弃和慢链路故障, 验证闭环慢链路后能回退基线。"

用法:
    # 无权重时先验证流程(用 mock 后端)
    python scripts/run_experiments.py --groups A B C D \
        --override detector.backend=scenario corrector.backend=mock \
                   source.type=simulated source.max_frames=600

    # 正式对比；B-E 会自动叠加 configs/dinov3_local.yaml
    python scripts/run_experiments.py --groups A B C D E \
        --source data/voyage2026_03/clip01.mp4 --max-frames 1800

输出: `runs/experiments/<时间戳>/` 下每个实验组一个子目录, 并汇总 comparison.csv
与 comparison.json。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

GROUPS = {
    "A": ("configs/experiment_a.yaml", "YOLO+ByteTrack+图像伺服"),
    "B": ("configs/experiment_b.yaml", "A+每十帧 DINOv3 校正"),
    "C": ("configs/experiment_c.yaml", "B+事件触发"),
    "D": ("configs/experiment_d.yaml", "C+恒速预测"),
    "E": ("configs/experiment_e.yaml", "C+学习的时序预测"),
}

SUMMARY_COLUMNS = [
    "group",
    "description",
    "frames",
    "fast_hz",
    "bridge_latency_p95_ms",
    "visibility_ratio",
    "central_ratio_when_visible",
    "target_switches",
    "losses",
    "recoveries",
    "mean_recovery_time_s",
    "corrections_applied",
    "corrections_rejected",
    "correction_drift",
    "template_freezes",
    "templates_frozen",
    "slow_link_faults",
    "slow_request_hz",
    "slow_completion_hz",
    "slow_completion_ratio",
    "slow_latency_p95_s",
    "slow_result_age_p95_s",
    "success_all",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量运行实验组 A~E")
    p.add_argument("--groups", nargs="+", default=["A", "B", "C", "D", "E"], choices=list(GROUPS))
    p.add_argument("--config", default=None, help="主配置路径")
    p.add_argument("--source", default=None, help="覆盖 source.path")
    p.add_argument("--type", default=None, help="覆盖 source.type")
    p.add_argument("--split", default=None, help="sequence_dir 专用: val | test | val,test")
    p.add_argument("--max-sequences", type=int, default=None, help="sequence_dir 专用")
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="帧数上限; sequence_dir 下自动解释为逐序列上限",
    )
    p.add_argument("--repeat", type=int, default=1, help="每组重复次数(用于统计分散度)")
    p.add_argument("--realtime", action="store_true", help="按真实帧率回放(指标才有实时意义)")
    p.add_argument("--override", nargs="*", default=[], help="额外 key=value 覆盖")
    p.add_argument("--out", default=None, help="输出根目录")
    p.add_argument("--no-record", action="store_true", help="不写叠加视频(批量时更快)")
    return p.parse_args()


def run_group(args, group: str, repeat: int, out_root: Path) -> dict | None:
    overlay, description = GROUPS[group]
    run_name = f"{group}_{repeat}"

    cmd = [
        sys.executable,
        "-m",
        "pandeng.cli.track",
        "--overlay",
        str(REPO_ROOT / overlay),
        "--run-name",
        run_name,
        "--override",
        f"recorder.dir={out_root}",
    ]
    # B-E 使用统一的本地 DINOv3 资源配置；A 保持纯快速环，不加载慢环资源。
    if group in {"B", "C", "D", "E"}:
        cmd += ["--overlay", str(REPO_ROOT / "configs" / "dinov3_local.yaml")]
    if args.config:
        cmd += ["--config", args.config]
    if args.source:
        cmd += ["--source", args.source]
    if args.type:
        cmd += ["--type", args.type]
    if args.split:
        cmd += ["--split", args.split]
    if args.max_sequences:
        cmd += ["--max-sequences", str(args.max_sequences)]
    if args.max_frames:
        # sequence_dir 下 --max-frames 是整个运行的上限, 会把第一个序列跑完就掐断,
        # 所以自动改写为逐序列上限。
        if (args.type or "").lower() == "sequence_dir":
            cmd += ["--max-frames-per-sequence", str(args.max_frames)]
        else:
            cmd += ["--max-frames", str(args.max_frames)]
    if args.realtime:
        cmd += ["--override", "source.realtime_pacing=true"]
    if args.no_record:
        # 只关闭叠加视频(批量实验的耗时主要在这里); 逐帧 CSV 与 summary.json
        # 必须保留, 否则无法统计错误校正/模板污染/过期丢弃。
        cmd += ["--no-video"]
    if args.override:
        cmd += ["--override", *args.override]

    print(f"\n{'=' * 78}\n[实验组 {group}] {description}\n{'=' * 78}")
    print(" ".join(cmd[1:]))
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    elapsed = time.monotonic() - started

    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:], file=sys.stderr)
        print(f"[实验组 {group}] 失败, 退出码 {proc.returncode}", file=sys.stderr)
        return None

    summary_path = out_root / run_name / "summary.json"
    if not summary_path.is_file():
        print(f"未找到 {summary_path}", file=sys.stderr)
        return None

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    criteria = payload["success_criteria"]
    slow = summary.get("slow_loop") or {}

    row = {
        "group": group,
        "description": description,
        "repeat": repeat,
        "wall_time_s": round(elapsed, 2),
        "frames": summary.get("frames"),
        "fast_hz": summary.get("fast_hz"),
        "bridge_latency_p95_ms": summary.get("bridge_latency_p95_ms"),
        "visibility_ratio": summary.get("visibility_ratio"),
        "central_ratio_when_visible": summary.get("central_ratio_when_visible"),
        "target_switches": summary.get("target_switches"),
        "losses": summary.get("losses"),
        "recoveries": summary.get("recoveries"),
        "mean_recovery_time_s": summary.get("mean_recovery_time_s"),
        "corrections_applied": summary.get("corrections_applied"),
        "corrections_rejected": summary.get("corrections_rejected"),
        "correction_drift": summary.get("correction_drift"),
        "template_freezes": summary.get("template_freezes"),
        "slow_link_faults": summary.get("slow_link_faults"),
        "slow_request_hz": slow.get("request_hz"),
        "slow_completion_hz": slow.get("completion_hz"),
        "slow_completion_ratio": slow.get("completion_ratio"),
        "slow_latency_p95_s": slow.get("latency_p95_s"),
        "slow_result_age_p95_s": slow.get("result_age_p95_s"),
        "success_all": all(criteria.values()),
        "criteria": json.dumps(criteria, ensure_ascii=False),
    }
    print(
        f"[实验组 {group}] 完成: 居中率={row['central_ratio_when_visible']}, "
        f"可见率={row['visibility_ratio']}, 慢链路请求={row['slow_request_hz']}Hz, "
        f"成功率={'PASS' if row['success_all'] else 'FAIL'}"
    )
    return row


def main() -> int:
    args = parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) if args.out else REPO_ROOT / "runs" / "experiments" / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for group in args.groups:
        for repeat in range(args.repeat):
            row = run_group(args, group, repeat, out_root)
            if row is not None:
                rows.append(row)

    if not rows:
        print("没有任何实验组成功完成", file=sys.stderr)
        return 1

    # --- 汇总 ---
    csv_path = out_root / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS + ["repeat", "wall_time_s", "criteria"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})

    aggregated = {}
    for group in args.groups:
        entries = [r for r in rows if r["group"] == group]
        if not entries:
            continue
        aggregated[group] = {
            "description": GROUPS[group][1],
            "runs": len(entries),
            "central_ratio_mean": _mean(entries, "central_ratio_when_visible"),
            "central_ratio_std": _std(entries, "central_ratio_when_visible"),
            "visibility_ratio_mean": _mean(entries, "visibility_ratio"),
            "target_switches_mean": _mean(entries, "target_switches"),
            "losses_mean": _mean(entries, "losses"),
            "recovery_time_mean": _mean(entries, "mean_recovery_time_s", skip_none=True),
            "slow_request_hz_mean": _mean(entries, "slow_request_hz"),
            "slow_latency_p95_s_mean": _mean(entries, "slow_latency_p95_s"),
        }

    # D/E 相对 C 的跟随增益
    gains = {}
    if "C" in aggregated:
        for group in ("D", "E"):
            if group in aggregated:
                gains[f"{group}_vs_C_central_gain"] = _delta(
                    aggregated[group]["central_ratio_mean"],
                    aggregated["C"]["central_ratio_mean"],
                )

    report = {
        "generated_at": stamp,
        "groups": aggregated,
        "gains": gains,
        "rows": rows,
        "notes": [
            "指标均以整机实测验收, 仿真结果只用于链路与控制逻辑验证。",
            "请求频率不等于实际完成频率, 二者分别报告。",
            "成功要求: 无人工接管、目标可见时间≥80%、中央 50% 区域时间比例≥80%。",
        ],
    }
    (out_root / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 78}\n汇总\n{'=' * 78}")
    header = f"{'组':<4}{'居中率':>10}{'可见率':>10}{'误切':>8}{'丢失':>8}{'重捕(s)':>10}{'慢请求Hz':>11}"
    print(header)
    for group, stat in aggregated.items():
        print(
            f"{group:<4}"
            f"{_fmt(stat['central_ratio_mean']):>10}"
            f"{_fmt(stat['visibility_ratio_mean']):>10}"
            f"{_fmt(stat['target_switches_mean']):>8}"
            f"{_fmt(stat['losses_mean']):>8}"
            f"{_fmt(stat['recovery_time_mean']):>10}"
            f"{_fmt(stat['slow_request_hz_mean']):>11}"
        )
    if gains:
        print("\n跟随增益(相对 C 组居中率):")
        for key, value in gains.items():
            print(f"  {key}: {value:+.4f}" if value is not None else f"  {key}: n/a")
    print(f"\n明细: {csv_path}\n报告: {out_root / 'comparison.json'}")
    return 0


def _mean(entries, key, skip_none: bool = False):
    values = [r[key] for r in entries if r.get(key) is not None]
    if skip_none and not values:
        return None
    if not values:
        return 0.0
    return round(float(statistics.fmean(values)), 4)


def _std(entries, key):
    values = [r[key] for r in entries if r.get(key) is not None]
    if len(values) < 2:
        return 0.0
    return round(float(statistics.pstdev(values)), 4)


def _delta(a, b):
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 4)


def _fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
