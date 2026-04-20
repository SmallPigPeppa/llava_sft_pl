from __future__ import annotations

import importlib.metadata
import importlib.util
from functools import lru_cache

from packaging import version


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def is_pillow_available() -> bool:
    return _is_package_available("PIL")


def is_pyav_available() -> bool:
    return _is_package_available("av")


@lru_cache
def is_transformers_version_greater_than(content: str) -> bool:
    try:
        installed = version.parse(importlib.metadata.version("transformers"))
    except Exception:
        installed = version.parse("0.0.0")
    return installed >= version.parse(content)
