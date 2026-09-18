"""单元测试: DINOv3 资源发现与加载。

覆盖离线部署最容易出错的几处:
- 上游仓库布局/键名前缀变化时的资源发现;
- checkpoint 包装层级(`model` / `state_dict` / `teacher` / 裸 state_dict);
- 形状不匹配的键被跳过并记录, 而不是整体抛错;
- `allow_fallback=False` 时资源缺失必须报错, 而不是悄悄换骨干。

不需要网络, 也不依赖真实 DINOv3 权重。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pandeng.perception.dinov3_loader import (
    find_hubconf,
    list_hubconf_factories,
    load_backbone,
    load_state_dict_tolerant,
    read_preprocessor_norm,
    unwrap_state_dict,
)

REPO_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 资源发现
# ---------------------------------------------------------------------------
def test_find_hubconf_at_repo_root(tmp_path: Path):
    (tmp_path / "hubconf.py").write_text("def dinov3_vits16(): pass\n", encoding="utf-8")
    assert find_hubconf(tmp_path) == tmp_path / "hubconf.py"


def test_find_hubconf_in_nested_package(tmp_path: Path):
    """上游也可能把 hubconf 放在包目录里。"""
    (tmp_path / "dinov3").mkdir()
    (tmp_path / "dinov3" / "hubconf.py").write_text("", encoding="utf-8")
    assert find_hubconf(tmp_path) == tmp_path / "dinov3" / "hubconf.py"


def test_find_hubconf_in_versioned_subdir(tmp_path: Path):
    """tarball 解压后常见的 dinov3-main/ 一层。"""
    (tmp_path / "dinov3-main").mkdir()
    (tmp_path / "dinov3-main" / "hubconf.py").write_text("", encoding="utf-8")
    assert find_hubconf(tmp_path) == tmp_path / "dinov3-main" / "hubconf.py"


def test_find_hubconf_missing(tmp_path: Path):
    assert find_hubconf(tmp_path) is None
    assert find_hubconf(tmp_path / "does-not-exist") is None


def test_list_hubconf_factories_static(tmp_path: Path):
    hubconf = tmp_path / "hubconf.py"
    hubconf.write_text(
        "import os\n"
        "def dinov3_vits16(**kwargs): pass\n"
        "def dinov3_vitb16(**kwargs): pass\n"
        "def helper(): pass\n",
        encoding="utf-8",
    )
    assert list_hubconf_factories(hubconf) == ["dinov3_vits16", "dinov3_vitb16"]


# ---------------------------------------------------------------------------
# checkpoint 解包
# ---------------------------------------------------------------------------
def test_unwrap_state_dict_wrapped_formats():
    inner = {"cls_token": torch.zeros(1, 8), "pos_embed": torch.zeros(1, 4, 8)}
    assert set(unwrap_state_dict({"model": inner})) == set(inner)
    assert set(unwrap_state_dict({"state_dict": inner})) == set(inner)
    assert set(unwrap_state_dict({"model_state_dict": inner})) == set(inner)
    # DINOv3 的 EMA teacher 常有两层包装
    assert set(unwrap_state_dict({"teacher": {"model": inner}})) == set(inner)


def test_unwrap_state_dict_bare():
    inner = {"cls_token": torch.zeros(1, 8)}
    assert set(unwrap_state_dict(inner)) == set(inner)


def test_unwrap_state_dict_strips_prefixes():
    state = {"module.model.cls_token": torch.zeros(1, 8)}
    assert "cls_token" in unwrap_state_dict(state)


def test_unwrap_state_dict_rejects_non_dict():
    with pytest.raises(TypeError):
        unwrap_state_dict([1, 2, 3])


# ---------------------------------------------------------------------------
# 容错加载
# ---------------------------------------------------------------------------
class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 8))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 4, 8))
        self.blocks = torch.nn.Linear(8, 8)


def test_load_state_dict_tolerant_all_matching():
    model = _Tiny()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    n_loaded, missing, unexpected, mismatched = load_state_dict_tolerant(model, state)
    assert n_loaded == len(state)
    assert not missing and not unexpected and not mismatched


def test_load_state_dict_tolerant_skips_shape_mismatch():
    """形状不匹配必须被跳过并记录, 不能整体抛错。"""
    model = _Tiny()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["pos_embed"] = torch.zeros(1, 999, 8)      # 形状错误
    state["unexpected_key"] = torch.zeros(3)

    n_loaded, missing, unexpected, mismatched = load_state_dict_tolerant(model, state)
    assert n_loaded == len(model.state_dict()) - 1
    assert any("pos_embed" in m for m in mismatched)
    assert "unexpected_key" in unexpected
    assert not missing


def test_load_state_dict_tolerant_records_missing():
    model = _Tiny()
    state = {"cls_token": torch.zeros(1, 8)}
    n_loaded, missing, _, _ = load_state_dict_tolerant(model, state)
    assert n_loaded == 1
    assert "pos_embed" in missing


def test_load_state_dict_tolerant_raises_when_nothing_loads():
    """一个张量都装不上时必须报错, 而不是静默返回一个随机骨干。"""
    model = _Tiny()
    with pytest.raises(RuntimeError, match="没有任何张量被加载"):
        load_state_dict_tolerant(model, {"totally_different": torch.zeros(1)})


# ---------------------------------------------------------------------------
# 预处理配置
# ---------------------------------------------------------------------------
def test_read_preprocessor_norm(tmp_path: Path):
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"image_mean": [0.1, 0.2, 0.3], "image_std": [0.4, 0.5, 0.6]}),
        encoding="utf-8",
    )
    mean, std = read_preprocessor_norm(tmp_path)
    assert mean == (0.1, 0.2, 0.3)
    assert std == (0.4, 0.5, 0.6)


def test_read_preprocessor_norm_missing_returns_none(tmp_path: Path):
    assert read_preprocessor_norm(tmp_path) == (None, None)


def test_read_preprocessor_norm_tolerates_bad_json(tmp_path: Path):
    (tmp_path / "preprocessor_config.json").write_text("{not json", encoding="utf-8")
    assert read_preprocessor_norm(tmp_path) == (None, None)


# ---------------------------------------------------------------------------
# load_backbone 的失败语义
# ---------------------------------------------------------------------------
def test_no_fallback_raises_when_resources_missing(tmp_path: Path):
    """正式实验必须走 allow_fallback=false: 资源缺失要直接报错。"""
    with pytest.raises(RuntimeError) as excinfo:
        load_backbone(
            "dinov3_vits16",
            repo_dir=tmp_path / "missing_repo",
            hf_dir=tmp_path / "missing_hf",
            weights=None,
            impl="auto",
            allow_fallback=False,
        )
    message = str(excinfo.value)
    assert "allow_fallback=false" in message
    assert "download_dinov3.py" in message


def test_fallback_is_flagged(tmp_path: Path):
    """允许回退时必须显式标记, 便于在 summary.json 里暴露。"""
    result = load_backbone(
        "dinov3_vits16",
        repo_dir=tmp_path / "missing_repo",
        hf_dir=tmp_path / "missing_hf",
        weights=None,
        impl="auto",
        allow_fallback=True,
    )
    assert result.is_fallback
    assert result.source.startswith("fallback:")


# ---------------------------------------------------------------------------
# transformers 路径(需要 transformers; 用同构小模型验证代码通路)
# ---------------------------------------------------------------------------
def _make_snapshot(tmp_path: Path, hidden_size: int = 384) -> Path:
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "DINOv3ViTModel"):
        pytest.skip("当前 transformers 版本未提供 DINOv3ViTModel")

    snapshot = tmp_path / "hf" / "dinov3-vits16-pretrain-lvd1689m"
    snapshot.mkdir(parents=True)
    config = transformers.DINOv3ViTConfig(
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=6,
        patch_size=16,
        image_size=224,
        num_channels=3,
    )
    transformers.DINOv3ViTModel(config).save_pretrained(str(snapshot))
    (snapshot / "preprocessor_config.json").write_text(
        json.dumps({"image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225]}),
        encoding="utf-8",
    )
    return snapshot


def test_load_backbone_from_hf_snapshot(tmp_path: Path):
    snapshot = _make_snapshot(tmp_path)
    result = load_backbone(
        "dinov3_vits16",
        hf_dir=snapshot,
        impl="transformers",
        allow_fallback=False,
        freeze=True,
    )
    assert not result.is_fallback
    assert result.source.startswith("transformers:")
    assert result.image_mean == (0.485, 0.456, 0.406)

    out = result.model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 384)
    # 骨干必须被冻结
    assert all(not p.requires_grad for p in result.model.parameters())


def test_hf_snapshot_without_config_fails(tmp_path: Path):
    """快照不完整(缺 config.json)必须走失败分支, 不能静默返回空模型。"""
    snapshot = tmp_path / "hf" / "broken"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"x" * 10)
    with pytest.raises(RuntimeError):
        load_backbone(
            "dinov3_vits16",
            hf_dir=snapshot,
            impl="transformers",
            allow_fallback=False,
        )
