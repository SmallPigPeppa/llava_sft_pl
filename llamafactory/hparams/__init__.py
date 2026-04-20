from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DataArguments:
    """Minimal train-only data arguments used by this project."""

    template: str | None = None
    dataset: str | list[str] | None = None
    dataset_dir: str = "data"
    media_dir: str | None = None
    cutoff_len: int = 2048
    max_samples: int | None = None
    preprocessing_batch_size: int = 1000
    preprocessing_num_workers: int | None = None
    overwrite_cache: bool = False
    ignore_pad_token_for_loss: bool = True
    train_on_prompt: bool = False
    mask_history: bool = False
    packing: bool | None = None
    neat_packing: bool = False
    tokenized_path: str | None = None
    data_shared_file_system: bool = False
    mix_strategy: str = "concat"
    tool_format: str | None = None
    default_system: str | None = None
    enable_thinking: bool | None = True

    def __post_init__(self) -> None:
        if isinstance(self.dataset, str):
            self.dataset = [item.strip() for item in self.dataset.split(",") if item.strip()]
        if self.media_dir is None:
            self.media_dir = self.dataset_dir
        if self.mask_history and self.train_on_prompt:
            raise ValueError("`mask_history` is incompatible with `train_on_prompt`.")
        if self.neat_packing:
            self.packing = True
        if self.packing:
            self.cutoff_len -= 1  # matches LLaMA-Factory packing convention

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelArguments:
    """Minimal model fields read by the data pipeline."""

    model_name_or_path: str | None = None
    cache_dir: str | None = None
    hf_hub_token: str | None = None
    trust_remote_code: bool = False
    compute_dtype: Any | None = None
    block_diag_attn: bool = False
    model_max_length: int | None = None

    def __post_init__(self) -> None:
        if self.model_name_or_path is None:
            raise ValueError("Please provide `model_name_or_path`.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
