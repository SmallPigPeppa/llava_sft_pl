# Copyright 2025 Optuna, HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/utils/logging.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import threading
from typing import Optional

_thread_lock = threading.RLock()
_default_handler: Optional["logging.Handler"] = None
_default_log_level: "logging._Level" = logging.INFO


def _get_default_logging_level() -> "logging._Level":
    r"""Return the default logging level."""
    env_level_str = os.getenv("LLAMAFACTORY_VERBOSITY", None)
    if env_level_str:
        if env_level_str.upper() in logging._nameToLevel:
            return logging._nameToLevel[env_level_str.upper()]
        else:
            raise ValueError(f"Unknown logging level: {env_level_str}.")

    return _default_log_level


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> "logging.Logger":
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    r"""Configure root logger using a stdout stream handler with an explicit format."""
    global _default_handler

    with _thread_lock:
        if _default_handler:  # already configured
            return

        formatter = logging.Formatter(
            fmt="[%(levelname)s|%(asctime)s] %(name)s:%(lineno)s >> %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.setFormatter(formatter)
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False


def get_logger(name: str | None = None) -> "logging.Logger":
    r"""Return a logger with the specified name. It it not supposed to be accessed externally."""
    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)


def info_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.info(*args, **kwargs)


def warning_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)

logging.Logger.info_rank0 = info_rank0
logging.Logger.warning_rank0 = warning_rank0
