from __future__ import annotations

import logging as _logging
import os
import sys
from types import SimpleNamespace

DATA_CONFIG = "dataset_info.json"
IGNORE_INDEX = -100
IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")

_HANDLER: _logging.Handler | None = None


def get_logger(name: str | None = None) -> _logging.Logger:
    global _HANDLER
    if _HANDLER is None:
        _HANDLER = _logging.StreamHandler(sys.stdout)
        _HANDLER.setFormatter(
            _logging.Formatter("[%(levelname)s|%(asctime)s] %(name)s:%(lineno)s >> %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        root = _logging.getLogger("llamafactory")
        root.addHandler(_HANDLER)
        root.setLevel(os.getenv("LLAMAFACTORY_VERBOSITY", "INFO").upper())
        root.propagate = False
    return _logging.getLogger(name or "llamafactory")


def _rank0() -> bool:
    return int(os.getenv("LOCAL_RANK", "0")) == 0


def _info_rank0(self: _logging.Logger, *args, **kwargs) -> None:
    if _rank0():
        self.info(*args, **kwargs)


def _warning_rank0(self: _logging.Logger, *args, **kwargs) -> None:
    if _rank0():
        self.warning(*args, **kwargs)


_logging.Logger.info_rank0 = _info_rank0  # type: ignore[attr-defined]
_logging.Logger.warning_rank0 = _warning_rank0  # type: ignore[attr-defined]
logging = SimpleNamespace(get_logger=get_logger)
