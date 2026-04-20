from __future__ import annotations

import logging as _logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

AUDIO_PLACEHOLDER = os.getenv("AUDIO_PLACEHOLDER", "<audio>")
DATA_CONFIG = "dataset_info.json"
IGNORE_INDEX = -100
IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")
VIDEO_PLACEHOLDER = os.getenv("VIDEO_PLACEHOLDER", "<video>")

MROPE_MODELS = {
    "glm4v",
    "glm_ocr",
    "Keye",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen2_5_omni_thinker",
    "qwen3_omni_moe_thinker",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
}

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


def has_tokenized_data(path: str | os.PathLike[str] | None) -> bool:
    return path is not None and Path(path).is_dir() and any(Path(path).iterdir())
