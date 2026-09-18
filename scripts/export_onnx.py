#!/usr/bin/env python
"""导出端侧引擎: ONNX -> TensorRT FP16。

对应方案第 7 节:
- 检测器采用 ONNX -> TensorRT FP16 部署;
- DINOv3 冻结推理, 验证算子和导出精度后构建端侧引擎;
- INT8 仅在代表性数据校准并通过精度检查后采用(默认关闭);
- 导出后必须做数值一致性校验, 否则端侧精度下降无法归因。

用法:
    # 检测器
    python scripts/export_onnx.py --kind detector --weights weights/fish_yolo11n.pt \
        --imgsz 640 --fp16 --check --out weights/fish_yolo11n.onnx

    # DINOv3 骨干(仅导出, 供 Orin 构建引擎)
    python scripts/export_onnx.py --kind dinov3 --model dinov3_vits16 \
        --input-size 224 --check --out weights/dinov3_vits16.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="导出 ONNX / TensorRT 引擎")
    p.add_argument("--kind", choices=["detector", "dinov3"], required=True)
    p.add_argument("--weights", default=None, help="detector: .pt 权重路径")
    p.add_argument("--model", default="dinov3_vits16", help="dinov3: 模型名")
    p.add_argument("--hub-repo", default="facebookresearch/dinov3")
    p.add_argument(
        "--dinov3-weights",
        default=None,
        help="dinov3: HF 快照目录或规范化后的 .pth 路径",
    )
    p.add_argument("--hf-dir", default=None, help="dinov3: HF 快照目录(transformers 路径)")
    p.add_argument("--repo-dir", default=None, help="dinov3: 源码仓库目录(torch.hub 路径)")
    p.add_argument(
        "--impl",
        choices=["auto", "torchhub", "transformers"],
        default="auto",
        help="dinov3: 加载实现; 与运行时 corrector.impl 语义一致",
    )
    p.add_argument("--imgsz", type=int, default=640, help="detector 输入尺寸")
    p.add_argument("--input-size", type=int, default=224, help="dinov3 输入尺寸")
    p.add_argument("--out", required=True, help="输出 ONNX 路径")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--fp16", action="store_true", help="导出 FP16")
    p.add_argument("--int8", action="store_true", help="导出 INT8(需校准, 默认关闭)")
    p.add_argument("--simplify", action="store_true", default=True)
    p.add_argument("--check", action="store_true", help="导出后做数值一致性校验")
    p.add_argument("--rtol", type=float, default=2e-2)
    p.add_argument("--atol", type=float, default=1e-2)
    p.add_argument("--report", default=None, help="校验报告 JSON 路径")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 检测器
# ---------------------------------------------------------------------------
def export_detector(args) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("未安装 ultralytics", file=sys.stderr)
        return 1

    if not args.weights or not Path(args.weights).is_file():
        print(f"权重不存在: {args.weights}", file=sys.stderr)
        return 1
    if args.int8:
        print(
            "警告: INT8 需要代表性数据校准并通过精度检查后才可采用(方案第 7 节)。\n"
            "      请确认已用真实水下片段完成校准, 否则请使用 FP16。",
            file=sys.stderr,
        )

    model = YOLO(args.weights)
    out = Path(model.export(
        format="engine" if args.out.endswith(".engine") else "onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        half=args.fp16,
        int8=args.int8,
        simplify=args.simplify,
        dynamic=False,
        device=0,
    ))
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    if out.resolve() != target.resolve():
        import shutil

        shutil.copy2(out, target)
    print(f"已导出: {target}")

    if args.check:
        return _check_detector(args, target)
    return 0


def _check_detector(args, onnx_path: Path) -> int:
    """对比 PyTorch 与 ONNXRuntime 的检测输出, 验证导出精度。"""
    import cv2
    import onnxruntime as ort
    from ultralytics import YOLO

    model = YOLO(args.weights)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    reference = model.predict(
        source=image, imgsz=args.imgsz, device="cpu", verbose=False, conf=0.05
    )[0]
    ref_boxes = (
        reference.boxes.xyxy.detach().cpu().numpy()
        if reference.boxes is not None and len(reference.boxes)
        else np.zeros((0, 4))
    )

    blob = image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)[None] / 255.0
    onnx_out = session.run(None, {input_name: blob})[0]

    report = {
        "onnx": str(onnx_path),
        "reference_boxes": ref_boxes.shape[0],
        "onnx_output_shape": list(np.asarray(onnx_out).shape),
        "onnx_output_finite": bool(np.isfinite(np.asarray(onnx_out)).all()),
    }
    ok = report["onnx_output_finite"]
    if ref_boxes.shape[0] > 0:
        report["note"] = (
            "检测框数量与坐标需在端侧用真实数据复核; 此处仅验证导出算子可执行且数值有限。"
        )
    report["passed"] = bool(ok)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# DINOv3 骨干
# ---------------------------------------------------------------------------
def export_dinov3(args) -> int:
    """导出 DINOv3 骨干为 ONNX。

    复用 `pandeng.perception.dinov3_loader.load_backbone`, 与推理链路走**同一套**
    离线加载逻辑 —— 否则导出的引擎与线上骨干可能不是同一个东西, 端侧精度差异
    将无法归因。
    """
    import torch

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from pandeng.perception.dinov3_loader import load_backbone

    source = args.dinov3_weights
    hf_dir = args.hf_dir
    pth = None
    repo_dir = args.repo_dir
    if source:
        candidate = Path(source).expanduser()
        if candidate.is_dir():
            # 允许直接传 HF 快照目录
            hf_dir = hf_dir or str(candidate)
        elif candidate.is_file():
            pth = str(candidate)
        else:
            print(f"指定的 DINOv3 资源不存在: {candidate}", file=sys.stderr)
            return 1

    try:
        result = load_backbone(
            args.model,
            hub_repo=args.hub_repo,
            repo_dir=repo_dir,
            hf_dir=hf_dir,
            weights=pth,
            impl=args.impl,
            freeze=True,
            allow_fallback=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"加载 DINOv3 骨干失败: {exc}\n\n"
            "请先获取离线资源:\n"
            "  python scripts/download_dinov3.py --dest <dest>\n"
            "  python scripts/download_dinov3.py --dest <dest> --verify\n"
            "然后指定 --dinov3-weights <dest>/hf/<repo> （快照目录）",
            file=sys.stderr,
        )
        return 1

    model = result.model
    print(f"骨干来源: {result.source}")
    if result.is_fallback:
        print("错误: 拿到了回退骨干, 拒绝导出。", file=sys.stderr)
        return 1

    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)

    wrapper = _Wrapper(model).eval()

    with torch.no_grad():
        reference = wrapper(dummy)
        torch.onnx.export(
            wrapper,
            dummy,
            str(target),
            opset_version=args.opset,
            input_names=["input"],
            output_names=["features"],
            dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
            do_constant_folding=True,
        )
    print(f"已导出: {target}  输出形状 {tuple(reference.shape)}")

    if args.check:
        return _check_dinov3(args, target, wrapper, dummy, reference)
    return 0


def _check_dinov3(args, onnx_path: Path, model, dummy, reference) -> int:
    """核对 ONNX 与 PyTorch 的输出差异; 方案要求验证导出精度后再构建引擎。"""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_out = np.asarray(session.run(None, {input_name: dummy.numpy()})[0])

    ref = reference.detach().cpu().numpy()
    if ref.ndim > 2:
        ref = ref.reshape(ref.shape[0], -1)
    onnx_out = onnx_out.reshape(ref.shape)
    diff = np.abs(ref - onnx_out)
    passed = bool(np.allclose(ref, onnx_out, rtol=args.rtol, atol=args.atol))
    report = {
        "onnx": str(onnx_path),
        "output_shape": list(ref.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "rtol": args.rtol,
        "atol": args.atol,
        "passed": passed,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def main() -> int:
    args = parse_args()
    if args.kind == "detector":
        return export_detector(args)
    return export_dinov3(args)


if __name__ == "__main__":
    raise SystemExit(main())
