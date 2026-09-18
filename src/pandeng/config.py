"""配置加载与访问。

设计目标:
- YAML 为唯一事实来源, 便于交付(方案第 8 节要求交付"模板与校正配置、
  异步调度参数、跟踪控制参数")。
- 支持按 key 路径覆盖, 便于实验组 A~E 通过命令行差异化配置。
- 属性式访问 + 明确的默认值缺失报错, 避免静默使用错误参数。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

__all__ = ["Config", "load_config", "deep_merge"]


class Config(Mapping):
    """只读的嵌套配置对象, 支持 `cfg.controller.gains.yaw.kp` 式访问。"""

    def __init__(self, data: Mapping[str, Any], path: str = "<memory>") -> None:
        self._data: dict[str, Any] = {}
        self._path = path
        for key, value in data.items():
            self._data[str(key)] = _wrap(value, f"{path}.{key}")

    # -- 属性访问 -----------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:  # pragma: no cover - 参数拼写错误时给出清晰提示
            raise AttributeError(
                f"配置项缺失: {self._path}.{name}。"
                f"可用键: {sorted(self._data)}"
            ) from exc

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterable[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """按 `a.b.c` 路径取值, 缺失时返回 default。"""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Config) or part not in node._data:
                return default
            node = node._data[part]
        return node

    def has_path(self, dotted: str) -> bool:
        sentinel = object()
        return self.get_path(dotted, sentinel) is not sentinel

    def to_dict(self) -> dict[str, Any]:
        return {k: _unwrap(v) for k, v in self._data.items()}

    def dump(self, path: str | os.PathLike[str]) -> None:
        """把生效配置回写到文件, 作为实验记录的一部分。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, allow_unicode=True, sort_keys=False)

    def __repr__(self) -> str:  # pragma: no cover - 调试友好
        return f"Config({self._path}, keys={sorted(self._data)})"


def _wrap(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        return Config(value, path)
    if isinstance(value, list):
        return [_wrap(v, f"{path}[{i}]") for i, v in enumerate(value)]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value.to_dict()
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并字典。override 中的 None 表示"保留 base 值"。"""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if value is None:
            continue
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_overrides(data: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """应用 `a.b.c=value` 形式的命令行覆盖。"""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"覆盖项需为 key=value 形式, 收到: {item!r}")
        dotted, raw = item.split("=", 1)
        parts = [p for p in dotted.strip().split(".") if p]
        if not parts:
            raise ValueError(f"非法的覆盖键: {dotted!r}")
        try:
            parsed = yaml.safe_load(raw)
        except yaml.YAMLError:
            parsed = raw
        node = data
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = parsed
    return data


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    overlay: str | os.PathLike[str] | Iterable[str | os.PathLike[str]] | None = None,
    overrides: Iterable[str] = (),
) -> Config:
    """加载配置。

    Args:
        path: 主 YAML 路径, 默认使用仓库内 `configs/default.yaml`。
        overlay: 可选的叠加 YAML，或按顺序叠加的 YAML 列表
            (如实验配置和 `configs/dinov3_local.yaml`)。
        overrides: 形如 `controller.gains.yaw.kp=1.2` 的覆盖项。
    """
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    main_path = Path(path).expanduser().resolve()
    if not main_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {main_path}")

    with main_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置根节点必须是映射: {main_path}")

    if overlay is not None:
        overlays = [overlay] if isinstance(overlay, (str, os.PathLike)) else overlay
        for item in overlays:
            overlay_path = Path(item).expanduser().resolve()
            if not overlay_path.is_file():
                raise FileNotFoundError(f"叠加配置不存在: {overlay_path}")
            with overlay_path.open("r", encoding="utf-8") as fh:
                over = yaml.safe_load(fh) or {}
            if not isinstance(over, dict):
                raise ValueError(f"叠加配置根节点必须是映射: {overlay_path}")
            data = deep_merge(data, over)

    data = _apply_overrides(data, overrides)
    return Config(data, path=str(main_path))
