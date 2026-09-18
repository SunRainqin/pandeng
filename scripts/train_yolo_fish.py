#!/usr/bin/env python
"""YOLO11n 鱼类检测微调。

对应方案第 5 节:
- 加载 `yolo11n.pt`, **先适配鱼类检测头**, 再**低学习率微调后部网络**;
- 通用 COCO 类别不含鱼, 必须使用公开鱼类检测标签适配;
- 输入尺寸以 640 为起点, 使用验证集早停;
- 直接使用已准备好的 YOLO 格式数据集及其固定 train/val/test 划分。

两阶段策略(与方案一致):
  阶段 1(适配检测头): 冻结主干与前部, 只训练检测头, 用较大学习率快速收敛;
  阶段 2(低学习率微调): 解冻后部网络, 用小学习率精调。

用法:
    python scripts/train_yolo_fish.py \
        --data ./data/brackish-dataset/dataset_yolo/data.yaml \
        --epochs-head 20 --epochs-finetune 60 \
        --imgsz 640 --batch 16 --device 4\
        --name fish_yolo11n
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO11n 鱼类检测微调")
    p.add_argument("--data", required=True, help="数据集 data.yaml")
    p.add_argument("--base-weights", default="yolo11n.pt", help="初始权重")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="CUDA 设备号或 cpu")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs-head", type=int, default=20, help="阶段 1: 适配鱼类检测头的轮数")
    p.add_argument("--epochs-finetune", type=int, default=60, help="阶段 2: 低学习率微调的轮数")
    p.add_argument("--lr-head", type=float, default=1e-2)
    p.add_argument("--lr-finetune", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=15, help="验证集早停轮数")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="fish_yolo11n")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def stage1_adapt_head(model, args) -> None:
    """阶段 1: 冻结主干, 只训练检测头。

    通用 COCO 权重的检测头输出 80 类, 鱼类数据集类别数不同, ultralytics 会在
    首次训练时自动重建检测头。此阶段只让新头收敛, 避免大梯度破坏预训练特征。
    """
    head_names = ("model.23", "model.22")  # YOLO11 的检测头位于最后几层
    frozen = 0
    for name, param in model.model.named_parameters():
        if any(name.startswith(prefix) for prefix in head_names):
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
            frozen += 1
    print(f"[阶段1] 已冻结 {frozen} 个参数张量, 仅训练检测头")
    model.train(
        data=args.data,
        epochs=args.epochs_head,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=args.lr_head,
        lrf=0.1,
        warmup_epochs=1.0,
        patience=args.patience,
        project=args.project,
        name=f"{args.name}_stage1_head",
        seed=args.seed,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )


def stage2_finetune(model, args) -> Path:
    """阶段 2: 解冻后部网络, 低学习率微调。"""
    for param in model.model.parameters():
        param.requires_grad_(True)
    print("[阶段2] 已解冻全部参数, 使用低学习率微调后部网络")
    model.train(
        data=args.data,
        epochs=args.epochs_finetune,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=args.lr_finetune,
        lrf=0.01,
        warmup_epochs=0.5,
        patience=args.patience,
        project=args.project,
        name=f"{args.name}_stage2_finetune",
        seed=args.seed,
        exist_ok=True,
        verbose=True,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    return best


def main() -> int:
    args = parse_args()
    project_dir = Path(args.project).expanduser()
    if not project_dir.is_absolute():
        project_dir = REPO_ROOT / project_dir
    args.project = str(project_dir.resolve())

    try:
        from ultralytics import YOLO
    except ImportError:
        print("未安装 ultralytics, 请执行: pip install ultralytics", file=sys.stderr)
        return 1

    data_yaml = Path(args.data)
    if not data_yaml.is_file():
        print(
            f"数据集配置不存在: {data_yaml}\n"
            "请传入已准备好的 YOLO 数据集 data.yaml。",
            file=sys.stderr,
        )
        return 1

    print(f"加载初始权重: {args.base_weights}")
    model = YOLO(args.base_weights)

    stage1_adapt_head(model, args)

    # 阶段 2 从阶段 1 的最优权重继续
    stage1_best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    stage2_start = stage1_best if stage1_best.is_file() else Path(args.base_weights)
    print(f"阶段 2 起点权重: {stage2_start}")
    model = YOLO(str(stage2_start))
    best = stage2_finetune(model, args)

    # 交付物: 固定权重
    deliver = REPO_ROOT / "weights" / f"{args.name}.pt"
    deliver.parent.mkdir(parents=True, exist_ok=True)
    if best.is_file():
        shutil.copy2(best, deliver)
        print(f"\n已导出交付权重: {deliver}")

    report = {
        "args": vars(args),
        "stage1_best": str(stage1_best),
        "stage2_best": str(best),
        "deliverable": str(deliver),
    }
    out = project_dir / f"{args.name}_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练记录: {out}")
    # print(
    #     "\n注意(方案第 5 节): 模型完成微调后固定权重部署; "
    #     "现场只更新目标模板和状态, 不修改网络参数。"
    # )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
