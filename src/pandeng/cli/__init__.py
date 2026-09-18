"""命令行入口集合。"""

from __future__ import annotations

__all__ = ["main"]


def main(argv=None) -> int:  # pragma: no cover - 便捷转发
    from .track import main as track_main

    return track_main(argv)
