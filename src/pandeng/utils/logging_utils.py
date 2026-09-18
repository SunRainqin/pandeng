"""日志配置。"""

from __future__ import annotations

import logging
import sys
from typing import Optional

__all__ = ["setup_logging"]


_FORMAT = "%(asctime)s [%(levelname).1s] %(name)-28s %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: str | int = "INFO", *, log_file: Optional[str] = None) -> None:
    """配置根日志器。

    Args:
        level: 日志级别名称或数值。
        log_file: 可选的日志文件路径, 便于水试后回溯。
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=_FORMAT,
        datefmt=_DATEFMT,
        handlers=handlers,
        force=True,
    )
    # 第三方库降噪
    for noisy in ("ultralytics", "matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))
