from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DataArguments:
    """Train-only data arguments for online-tokenized image SFT."""

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
    mix_strategy: str = "concat"
    default_system: str | None = None
    enable_thinking: bool | None = True

    def __post_init__(self) -> None:
        if isinstance(self.dataset, str):
            self.dataset = [x.strip() for x in self.dataset.split(",") if x.strip()]
        if self.media_dir is None:
            self.media_dir = self.dataset_dir

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelArguments:
    """Minimal model fields needed by the data pipeline."""

    model_name_or_path: str | None = None
    cache_dir: str | None = None
    hf_hub_token: str | None = None
    trust_remote_code: bool = True
    compute_dtype: Any | None = None
    model_max_length: int | None = None

    def __post_init__(self) -> None:
        if self.model_name_or_path is None:
            raise ValueError("Please provide model.model_name_or_path.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
