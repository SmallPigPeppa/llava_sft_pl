from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ModelArguments:
    """Minimal model arguments required by the bundled LLaMA-Factory DataModule."""

    model_name_or_path: str | None = None
    cache_dir: str | None = None
    hf_hub_token: str | None = None
    trust_remote_code: bool = False

    # Filled by train.py after model/precision selection.
    compute_dtype: Any | None = None
    block_diag_attn: bool = False
    model_max_length: int | None = None

    def __post_init__(self) -> None:
        if self.model_name_or_path is None:
            raise ValueError("Please provide `model_name_or_path`.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
