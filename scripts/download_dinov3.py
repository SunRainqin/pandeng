#!/usr/bin/env python
"""从 ModelScope 获取 DINOv3 的离线 Transformers 快照并校验。

当前项目只使用 ModelScope 上的 DINOv3 ViT-S/16 快照。下载支持断点续传
语义：已完整存在且大小一致的文件会跳过，下载中断后直接重跑即可继续。

用法:
    python scripts/download_dinov3.py
    python scripts/download_dinov3.py --verify
    python scripts/download_dinov3.py --dry-run
    python scripts/download_dinov3.py --pack /path/dinov3.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence

DEFAULT_DEST = Path("/data2/zyq/weights/DINOv3")
MS_ENDPOINT = "https://modelscope.cn"
MS_REVISION = "master"
MODEL_CATALOG = {
    "dinov3_vits16": {
        "repo": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "params_m": 21,
        "embed_dim": 384,
    },
}
SNAPSHOT_SUFFIXES = {".json", ".safetensors", ".pth", ".bin", ".md"}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def list_files(repo: str, revision: str, root: str = "") -> List[dict]:
    url = f"{MS_ENDPOINT}/api/v1/models/{repo}/repo/files"
    query = f"?Revision={revision}&Root={root}"
    request = urllib.request.Request(
        url + query,
        headers={"User-Agent": "pandeng-dinov3-fetch/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    files: List[dict] = []
    for item in (payload.get("Data") or {}).get("Files") or []:
        path = str(item.get("Path", "")).lstrip("/")
        if not path:
            continue
        if str(item.get("Type")) == "tree":
            files.extend(list_files(repo, revision, path))
        else:
            files.append({"path": path, "size": int(item.get("Size") or 0)})
    return files


def download_file(url: str, target: Path, expected_size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "pandeng-dinov3-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if temporary.stat().st_size != expected_size:
        temporary.unlink(missing_ok=True)
        raise IOError(
            f"文件大小不匹配: {target.name}, "
            f"expected={expected_size}, actual={temporary.stat().st_size}"
        )
    temporary.replace(target)


def snapshot_complete(snapshot: Path) -> bool:
    if not (snapshot / "config.json").is_file():
        return False
    return any(
        path.suffix.lower() in {".safetensors", ".bin", ".pth"}
        and path.stat().st_size > (1 << 20)
        for path in snapshot.iterdir()
        if path.is_file()
    )


def fetch_snapshot(
    model: str,
    dest: Path,
    *,
    revision: str,
    dry_run: bool,
    force: bool,
) -> Path | None:
    repo = MODEL_CATALOG[model]["repo"]
    snapshot = dest / "hf" / repo.rsplit("/", 1)[-1]
    if snapshot_complete(snapshot) and not force:
        log(f"快照已存在且完整: {snapshot}")
        return snapshot

    files = [
        item
        for item in list_files(repo, revision)
        if Path(item["path"]).suffix.lower() in SNAPSHOT_SUFFIXES
        and Path(item["path"]).name != ".gitattributes"
    ]
    if not files:
        raise RuntimeError(f"ModelScope 仓库没有可用快照文件: {repo}")

    total = sum(item["size"] for item in files)
    log(f"ModelScope {repo}: {len(files)} 个文件, 共 {human_size(total)}")
    if dry_run:
        for item in files:
            log(f"[dry-run] {item['path']} ({human_size(item['size'])})")
        return None

    for item in files:
        local = snapshot / item["path"]
        if local.is_file() and local.stat().st_size == item["size"] and not force:
            log(f"跳过(已完整): {item['path']}")
            continue
        url = f"{MS_ENDPOINT}/models/{repo}/resolve/{revision}/{item['path']}"
        log(f"下载 {item['path']} ({human_size(item['size'])})")
        download_file(url, local, item["size"])

    if not snapshot_complete(snapshot):
        raise RuntimeError(f"下载完成但快照不完整: {snapshot}")
    log(f"快照已就绪: {snapshot}")
    return snapshot


def verify(dest: Path, models: Sequence[str]) -> int:
    """离线校验已下载的 Transformers 快照。"""
    report: Dict[str, object] = {
        "dest": str(dest),
        "models": {},
        "recommended_impl": "transformers",
        "problems": [],
    }
    problems: List[str] = []
    for model in models:
        repo = MODEL_CATALOG[model]["repo"]
        snapshot = dest / "hf" / repo.rsplit("/", 1)[-1]
        entry: Dict[str, object] = {"path": str(snapshot)}
        report["models"][model] = entry  # type: ignore[index]
        if not snapshot.is_dir():
            problems.append(f"{snapshot} 不存在")
            continue

        config = snapshot / "config.json"
        weights = [
            path
            for path in snapshot.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".safetensors", ".bin", ".pth"}
            and path.stat().st_size > (1 << 20)
        ]
        entry["files"] = sorted(path.name for path in snapshot.iterdir() if path.is_file())
        entry["weights"] = [path.name for path in weights]
        if not config.is_file():
            problems.append(f"{snapshot} 缺少 config.json")
            continue
        if not weights:
            problems.append(f"{snapshot} 缺少模型权重文件")
            continue

        try:
            config_data = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{config} 无法解析: {exc}")
            continue
        entry["hidden_size"] = config_data.get("hidden_size")
        entry["expected_embed_dim"] = MODEL_CATALOG[model]["embed_dim"]
        if config_data.get("hidden_size") not in (None, MODEL_CATALOG[model]["embed_dim"]):
            problems.append(
                f"{config} hidden_size={config_data.get('hidden_size')} "
                f"与预期 {MODEL_CATALOG[model]['embed_dim']} 不一致"
            )
        log(f"快照: {snapshot}, 权重: {[path.name for path in weights]}")

    report["problems"] = problems
    report["ok"] = not problems
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "verify.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if problems:
        log(f"校验未通过, 共 {len(problems)} 项问题")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    log("校验通过。推荐 corrector.impl = transformers")
    return 0


def pack(dest: Path, archive: Path) -> int:
    archive.parent.mkdir(parents=True, exist_ok=True)
    log(f"打包 {dest} -> {archive}")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(dest, arcname=dest.name)
    log(f"完成: {archive} ({human_size(archive.stat().st_size)})")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 ModelScope 获取 DINOv3 Transformers 离线快照"
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dinov3_vits16"],
        choices=sorted(MODEL_CATALOG),
    )
    parser.add_argument("--ms-revision", default=MS_REVISION)
    parser.add_argument("--ms-endpoint", default=None)
    parser.add_argument("--verify", action="store_true", help="只校验已有快照")
    parser.add_argument("--dry-run", action="store_true", help="只显示下载计划")
    parser.add_argument("--pack", type=Path, default=None, help="打包资源目录")
    parser.add_argument("--force", action="store_true", help="重新下载已有文件")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global MS_ENDPOINT
    if args.ms_endpoint:
        MS_ENDPOINT = args.ms_endpoint.rstrip("/")
    dest = args.dest.expanduser().resolve()

    if args.verify:
        return verify(dest, args.models)
    if args.pack:
        return pack(dest, args.pack.expanduser().resolve())

    dest.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for model in args.models:
        snapshot = fetch_snapshot(
            model,
            dest,
            revision=args.ms_revision,
            dry_run=args.dry_run,
            force=args.force,
        )
        if snapshot is not None:
            snapshots.append(snapshot)

    if args.dry_run:
        return 0
    if not snapshots:
        raise RuntimeError("没有生成可用快照")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dest": str(dest),
        "source": "modelscope",
        "modelscope_endpoint": MS_ENDPOINT,
        "revision": args.ms_revision,
        "models": {
            model: {
                "snapshot": str(dest / "hf" / MODEL_CATALOG[model]["repo"].rsplit("/", 1)[-1]),
                "sha256": {
                    path.name: sha256_of(path)
                    for path in snapshot.iterdir()
                    if path.is_file()
                },
            }
            for model, snapshot in zip(args.models, snapshots)
        },
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return verify(dest, args.models)


if __name__ == "__main__":
    raise SystemExit(main())
