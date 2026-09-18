"""DINOv3 骨干的加载策略。

支持本地快照和已有本地源码目录，正式配置使用 Transformers 快照:

1. 本地源码仓库 `repo_dir`(按文件路径导入 hubconf.py);
2. HF 快照 `hf_dir` + transformers;
3. 已有 torch.hub 缓存;
4. 回退骨干 torchvision ViT-B/16 —— 必须显式允许, 并会在 `summary.json` 里
   标记 `embedder_is_fallback=true`, 避免用错误骨干跑出看起来正常的结论。

权重加载容错: 形状不匹配的张量跳过并记录, 而不是整体抛错 —— 上游 checkpoint
的键名前缀与包装层级经常变化, 整体失败会让排查无从下手。
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "BackboneLoadResult",
    "Dinov3ViTAdapter",
    "find_hubconf",
    "list_hubconf_factories",
    "load_backbone",
    "load_state_dict_tolerant",
    "read_preprocessor_norm",
    "unwrap_state_dict",
]

LOGGER = logging.getLogger(__name__)

_PREFIXES = ("module.", "model.", "backbone.", "_orig_mod.")


class BackboneLoadResult:
    """一次骨干加载的结果与来源。"""

    __slots__ = ("model", "source", "is_fallback", "detail", "image_mean", "image_std")

    def __init__(
        self,
        model: Any,
        source: str,
        is_fallback: bool,
        detail: str = "",
        image_mean: Optional[Tuple[float, float, float]] = None,
        image_std: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.model = model
        self.source = source
        self.is_fallback = is_fallback
        self.detail = detail
        self.image_mean = image_mean
        self.image_std = image_std

    def __repr__(self) -> str:  # pragma: no cover
        return f"BackboneLoadResult(source={self.source}, fallback={self.is_fallback})"


class Dinov3ViTAdapter:
    """把 `transformers.DINOv3ViTModel` 适配成"输入像素、输出全局描述子"。

    `DINOv3ViTModel` 返回 `BaseModelOutputWithPooling`, 其中:
    - `pooler_output`  : CLS token, 形状 (B, hidden) —— DINOv3 的标准全局描述子
    - `last_hidden_state`: (B, 1+N, hidden)

    这里取 `pooler_output`(缺失时退化为 CLS 位置), 使上层 `BaseEmbedder` 的
    "前向返回张量" 约定在所有后端上保持一致。
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __call__(self, pixel_values):
        output = self.inner(pixel_values=pixel_values)
        pooled = getattr(output, "pooler_output", None)
        if pooled is not None:
            return pooled
        return output.last_hidden_state[:, 0]

    def parameters(self):
        return self.inner.parameters()

    def to(self, *args, **kwargs):
        self.inner.to(*args, **kwargs)
        return self

    def eval(self):
        self.inner.eval()
        return self

    def train(self, mode: bool = True):
        self.inner.train(mode)
        return self

    @property
    def device(self):
        return next(self.inner.parameters()).device


def read_preprocessor_norm(
    hf_dir: str | Path,
) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    """从 `preprocessor_config.json` 读取归一化参数。

    DINOv3 用 ImageNet 统计量, 但我们不假设这一点: 若上游换了预处理而代码仍用
    硬编码均值方差, 特征会**静默**退化 —— 这类错误在关联指标上很难归因。
    """
    import json

    config_path = Path(hf_dir).expanduser() / "preprocessor_config.json"
    if not config_path.is_file():
        return None, None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None, None

    def _triple(key: str):
        value = payload.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return tuple(float(v) for v in value)
        return None

    # 部分版本把它写在 image_mean/image_std, 部分放在 image_processor 里
    mean = _triple("image_mean") or _triple("mean")
    std = _triple("image_std") or _triple("std")
    if mean is None or std is None:
        nested = payload.get("image_processor")
        if isinstance(nested, dict):
            mean = mean or _triple_from(nested, "image_mean")
            std = std or _triple_from(nested, "image_std")
    return mean, std


def _triple_from(node: Dict[str, Any], key: str):
    value = node.get(key)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(float(v) for v in value)
    return None


def _load_from_transformers(hf_dir: Path, model_name: str):
    """用 transformers 的 `DINOv3ViTModel` 加载本地 HF 快照(离线)。"""
    try:
        from transformers import DINOv3ViTModel
    except ImportError as exc:
        raise ImportError(
            "未安装 transformers。HF 快照是 transformers 格式, 需要:\n"
            "  pip install -r requirements-dinov3.txt"
        ) from exc

    if not hf_dir.is_dir():
        raise FileNotFoundError(f"HF 快照目录不存在: {hf_dir}")
    if not (hf_dir / "config.json").is_file():
        raise FileNotFoundError(f"{hf_dir} 下缺少 config.json, 快照不完整")

    # local_files_only=True: 无网机器上必须完全离线, 不允许回退到联网解析
    model = DINOv3ViTModel.from_pretrained(str(hf_dir), local_files_only=True)
    return Dinov3ViTAdapter(model), f"transformers:{hf_dir.name}"


# ---------------------------------------------------------------------------
# 本地仓库
# ---------------------------------------------------------------------------
def find_hubconf(repo_dir: str | Path) -> Optional[Path]:
    """在候选位置查找 `hubconf.py`。

    上游布局并不唯一, 因此同时接受 `repo/hubconf.py`、`repo/dinov3/hubconf.py`
    以及任意一层子目录下的 `hubconf.py`。
    """
    root = Path(repo_dir).expanduser()
    if not root.is_dir():
        return None
    for candidate in (root / "hubconf.py", root / "dinov3" / "hubconf.py"):
        if candidate.is_file():
            return candidate
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "hubconf.py").is_file():
            return child / "hubconf.py"
    return None


def list_hubconf_factories(hubconf: Path) -> List[str]:
    """静态枚举 hubconf 中形如 `dinov3_*` 的工厂函数(不执行仓库代码)。"""
    import ast

    try:
        tree = ast.parse(hubconf.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 语法不兼容时静默降级
        return []
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("dinov3")
    ]


@contextmanager
def _sys_path_prepended(path: Path):
    entries = [str(path), str(path.parent)]
    for entry in reversed(entries):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    try:
        yield
    finally:
        for entry in entries:
            if entry in sys.path:
                sys.path.remove(entry)


def _import_hubconf(hubconf: Path):
    """按文件路径导入 hubconf, 避免与其它同名模块冲突。"""
    module_name = f"_pandeng_dinov3_hubconf_{abs(hash(str(hubconf))) % 10**8}"
    spec = importlib.util.spec_from_file_location(module_name, str(hubconf))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {hubconf} 建立导入规格")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with _sys_path_prepended(hubconf.parent):
        spec.loader.exec_module(module)
    return module


def _call_factory(factory, model_name: str):
    """以最保守的方式调用模型工厂。

    不同版本的 hubconf 参数名不同(`pretrained=` / `weights=` / 无参), 这里
    逐个尝试并优先选择"只建结构、不下权重"的调用, 因为权重由我们自己加载 ——
    这样在无网环境下不会因为工厂内部尝试下载而失败。
    """
    attempts: List[Tuple[str, Dict[str, Any]]] = [
        ("pretrained=False", {"pretrained": False}),
        ("weights=None", {"weights": None}),
        ("pretrained_weights=None", {"pretrained_weights": None}),
        ("no-args", {}),
    ]
    errors: List[str] = []
    for label, kwargs in attempts:
        try:
            model = factory(**kwargs)
        except TypeError as exc:
            errors.append(f"{label}: TypeError({exc})")
            continue
        except Exception as exc:  # noqa: BLE001 - 工厂内部可能尝试联网
            errors.append(f"{label}: {type(exc).__name__}({str(exc)[:120]})")
            continue
        if model is not None:
            LOGGER.debug("DINOv3 工厂 %s 以 %s 调用成功", model_name, label)
            return model
    raise RuntimeError(
        f"无法构建 {model_name}; 已尝试的调用方式均失败: " + "; ".join(errors)
    )


def _load_from_local_repo(repo_dir: Path, model_name: str):
    hubconf = find_hubconf(repo_dir)
    if hubconf is None:
        raise FileNotFoundError(f"{repo_dir} 下未找到 hubconf.py")

    available = list_hubconf_factories(hubconf)
    module = _import_hubconf(hubconf)

    factory = getattr(module, model_name, None)
    if factory is None:
        raise AttributeError(
            f"hubconf 中不存在 {model_name}; 可用: {available or '(未能枚举)'}"
        )
    model = _call_factory(factory, model_name)
    return model, f"local-repo:{repo_dir}"


# ---------------------------------------------------------------------------
# 权重
# ---------------------------------------------------------------------------
def _strip_prefixes(state: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in _PREFIXES:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def unwrap_state_dict(payload: Any) -> Dict[str, Any]:
    """从 checkpoint 中取出真正的 state_dict。

    依次尝试 `model` / `state_dict` / `model_state_dict` / `teacher` / 整体。
    """
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint 类型不支持: {type(payload).__name__}")
    for key in ("model", "state_dict", "model_state_dict", "backbone", "teacher"):
        value = payload.get(key)
        if isinstance(value, dict) and value:
            # 再往下钻一层: DINOv3 的 EMA teacher 里可能还有嵌套
            for inner in ("model", "state_dict", "backbone"):
                if isinstance(value.get(inner), dict) and value[inner]:
                    return _strip_prefixes(value[inner])
            return _strip_prefixes(value)
    return _strip_prefixes(payload)


def load_state_dict_tolerant(model, state: Dict[str, Any]) -> Tuple[int, List[str], List[str], List[str]]:
    """容错加载 state_dict, 返回 (加载张量数, 缺失键, 多余键, 形状不匹配键)。

    与 `load_state_dict(strict=False)` 的区别: 形状不匹配的键会被**跳过并记录**,
    而不是直接抛 RuntimeError。上游 checkpoint 的键名与包装层级经常变化,
    容错加载能让"部分加载"这种可诊断状态暴露出来, 而不是整体失败。
    """
    import torch

    own = model.state_dict()
    to_load: Dict[str, Any] = {}
    missing: List[str] = []
    mismatched: List[str] = []

    for key, target in own.items():
        source = state.get(key)
        if source is None:
            missing.append(key)
            continue
        if not hasattr(source, "shape"):
            mismatched.append(f"{key}(非张量)")
            continue
        if tuple(source.shape) != tuple(target.shape):
            mismatched.append(
                f"{key}({tuple(source.shape)} != {tuple(target.shape)})"
            )
            continue
        to_load[key] = source

    unexpected = [k for k in state if k not in own]
    if to_load:
        model.load_state_dict(to_load, strict=False)

    if not to_load:
        raise RuntimeError(
            "没有任何张量被加载。可能原因: 权重不是 DINOv3 骨干, 或键名前缀与当前"
            f"实现不匹配。文件中的键示例: {list(state)[:5]}"
        )
    return len(to_load), missing, unexpected, mismatched


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def load_backbone(
    model_name: str,
    *,
    hub_repo: str = "facebookresearch/dinov3",
    repo_dir: Optional[str | Path] = None,
    hf_dir: Optional[str | Path] = None,
    weights: Optional[str | Path] = None,
    impl: str = "auto",
    freeze: bool = True,
    allow_fallback: bool = True,
    fallback_arch: str = "vit_b_16",
) -> BackboneLoadResult:
    """按优先级加载 DINOv3 骨干。

    Args:
        repo_dir: 本地源码仓库目录(torch.hub 路径, 离线首选)。
        hf_dir: 本地 Transformers 快照目录，详见 `scripts/download_dinov3.py`。
        weights: torch.hub 路径使用的 `.pth` state_dict。
        impl: 实现选择。`auto` = 本地源码 → Transformers 快照 → torch.hub 缓存;
            `torchhub` / `transformers` 则只用对应实现。
        allow_fallback: 是否允许回退到 torchvision ViT。为 False 时加载失败直接
            抛错 —— 正式实验建议设为 False, 避免结论建立在错误骨干上。
    """
    impl = str(impl).lower()
    if impl not in {"auto", "torchhub", "transformers"}:
        raise ValueError(f"未知的 corrector.impl: {impl}")

    # torch 必须在函数作用域顶部导入: 若只在某个分支里 import, 它就会变成
    # 局部变量, 使其它分支引用 torch 时报 UnboundLocalError。
    import torch

    model = None
    source = ""
    detail_parts: List[str] = []
    image_mean: Optional[Tuple[float, float, float]] = None
    image_std: Optional[Tuple[float, float, float]] = None
    use_torchhub = impl in {"auto", "torchhub"}
    use_transformers = impl in {"auto", "transformers"}

    # 1) 本地源码仓库(torch.hub 路径)
    if model is None and use_torchhub and repo_dir:
        try:
            model, source = _load_from_local_repo(Path(repo_dir), model_name)
            LOGGER.info("已从本地仓库加载 DINOv3 结构: %s", repo_dir)
        except Exception as exc:  # noqa: BLE001
            detail_parts.append(f"local-repo 失败: {type(exc).__name__}: {exc}")
            LOGGER.warning("从本地仓库加载 DINOv3 失败: %s", exc)

    # 2) transformers + 本地 HF 快照
    if model is None and use_transformers and hf_dir:
        try:
            hf_path = Path(hf_dir).expanduser()
            model, source = _load_from_transformers(hf_path, model_name)
            image_mean, image_std = read_preprocessor_norm(hf_path)
            LOGGER.info("已从 HF 快照加载 DINOv3(transformers): %s", hf_path)
        except Exception as exc:  # noqa: BLE001
            detail_parts.append(f"hf-snapshot 失败: {type(exc).__name__}: {exc}")
            LOGGER.warning("从 HF 快照加载 DINOv3 失败: %s", exc)
            model = None

    # 3) torch.hub(先本地缓存, 后在线)
    if model is None and use_torchhub:
        for label in ("hub-cache", "hub-online"):
            try:
                model = torch.hub.load(hub_repo, model_name, trust_repo=True, verbose=False)
                source = label
                LOGGER.info("已通过 torch.hub 加载 DINOv3 结构 (%s)", label)
                break
            except Exception as exc:  # noqa: BLE001
                detail_parts.append(f"{label} 失败: {type(exc).__name__}: {str(exc)[:160]}")
                model = None

    # 4) 回退骨干
    if model is None:
        if not allow_fallback:
            raise RuntimeError(
                "DINOv3 骨干不可用, 且 corrector.allow_fallback=false。\n"
                "  - 无网环境请先运行 scripts/download_dinov3.py 获取资源, 并设置:\n"
                "      corrector.repo_dir=<dest>/dinov3   (torch.hub 路径), 或\n"
                "      corrector.hf_dir=<dest>/hf/<repo>  (transformers 路径)\n"
                "  - 失败详情: " + " | ".join(detail_parts)
            )
        import torchvision.models as tvm

        LOGGER.warning(
            "DINOv3 骨干不可用, 已回退到 torchvision %s。"
            "该回退仅用于链路自检, 关联改善结论不可用于验收。",
            fallback_arch,
        )
        if fallback_arch != "vit_b_16":
            raise ValueError(f"不支持的回退骨干: {fallback_arch}")
        model = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = torch.nn.Identity()
        return BackboneLoadResult(
            model, f"fallback:{fallback_arch}", True, " | ".join(detail_parts) or "DINOv3 不可用"
        )

    # --- 额外 .pth 权重 ---
    # 只有 torch.hub 路径需要外部 .pth(工厂默认返回随机初始化)。
    # transformers 路径的权重已随快照加载, 此时再去抱怨 corrector.weights 缺失
    # 是误导性的。
    from_torchhub = source.startswith("hub-") or source.startswith("local-repo")
    loaded_path: Optional[Path] = None
    if weights and from_torchhub:
        candidate = Path(weights).expanduser()
        if candidate.is_file():
            loaded_path = candidate
        else:
            LOGGER.warning("未找到权重文件 %s, 将继续使用工厂自带的权重。", candidate)
    elif weights and not from_torchhub:
        LOGGER.debug(
            "当前为 %s 路径, 权重已随快照加载; 忽略 corrector.weights=%s", source, weights
        )

    if loaded_path is not None:
        try:
            payload = torch.load(str(loaded_path), map_location="cpu")
            state = unwrap_state_dict(payload)
            n_loaded, missing, unexpected, mismatched = load_state_dict_tolerant(model, state)
            LOGGER.info(
                "已加载权重 %s: %d 个张量, 缺失 %d, 多余 %d, 形状不匹配 %d",
                loaded_path, n_loaded, len(missing), len(unexpected), len(mismatched),
            )
            if missing:
                LOGGER.debug("缺失键示例: %s", missing[:5])
            if mismatched:
                LOGGER.warning("形状不匹配键示例: %s", mismatched[:5])
            source = f"{source}+weights:{loaded_path.name}"
        except Exception as exc:  # noqa: BLE001
            detail_parts.append(f"weights 加载失败: {type(exc).__name__}: {str(exc)[:160]}")
            LOGGER.warning("加载 %s 失败: %s", loaded_path, exc)
    elif from_torchhub:
        LOGGER.warning(
            "torch.hub 路径未加载任何外部权重文件, 当前骨干为工厂返回的权重"
            "(通常为随机初始化)。请在 corrector.weights 指定由 "
            "scripts/download_dinov3.py 产出的 .pth。"
        )
        source = f"{source}+no-weights"

    if freeze:
        params = model.parameters()
        for param in params:
            param.requires_grad_(False)
        model.eval()

    return BackboneLoadResult(
        model, source, False, " | ".join(detail_parts), image_mean, image_std
    )
