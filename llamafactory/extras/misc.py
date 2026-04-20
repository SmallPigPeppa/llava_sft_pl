from __future__ import annotations

import os


def has_tokenized_data(path: str | os.PathLike[str] | None) -> bool:
    return path is not None and os.path.isdir(path) and len(os.listdir(path)) > 0


def is_env_enabled(env_var: str, default: str = "0") -> bool:
    return os.getenv(env_var, default).lower() in {"true", "y", "1"}


def use_modelscope() -> bool:
    return is_env_enabled("USE_MODELSCOPE_HUB")


def use_openmind() -> bool:
    return is_env_enabled("USE_OPENMIND_HUB")
