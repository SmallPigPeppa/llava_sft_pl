from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


def _split_csv(value: str | list[str] | None) -> list[str] | None:
    if value is None or isinstance(value, list):
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class DataArguments:
    """Small SFT-only data config used by the bundled parquet DataModule."""

    template: str | None = "llava"
    dataset: str | list[str] | None = None
    dataset_dir: str = "data"
    media_dir: str | None = None

    cutoff_len: int = 2048
    train_on_prompt: bool = False
    mask_history: bool = False
    packing: bool | None = False
    neat_packing: bool = False

    preprocessing_mode: Literal["online", "offline"] = "online"
    preprocessing_batch_size: int = 1000
    preprocessing_num_workers: int | None = None
    overwrite_cache: bool = False
    tokenized_path: str | None = None
    data_shared_file_system: bool = False

    max_samples: int | None = None
    ignore_pad_token_for_loss: bool = True
    val_size: float = 0.0
    eval_dataset: str | list[str] | None = None
    streaming: bool = False

    # Kept for config compatibility; tool/eval/RL paths are intentionally not implemented.
    tool_format: str | None = None
    default_system: str | None = None
    enable_thinking: bool | None = True

    def __post_init__(self) -> None:
        self.dataset = _split_csv(self.dataset)
        self.eval_dataset = _split_csv(self.eval_dataset)
        if self.media_dir is None:
            self.media_dir = self.dataset_dir
        if self.eval_dataset:
            raise ValueError("Evaluation datasets were removed from this SFT-only project.")
        if self.val_size and self.val_size > 1e-6:
            raise ValueError("Validation split was removed; set data.val_size=0.0.")
        if self.streaming:
            raise ValueError("Streaming was removed; use local parquet datasets.")
        if self.mask_history and self.train_on_prompt:
            raise ValueError("mask_history and train_on_prompt are mutually exclusive.")
        if self.neat_packing:
            self.packing = True
        if self.packing:
            raise ValueError("Packing was removed to keep the SFT data path small and predictable.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelArguments:
    """Small model config surface consumed by the parquet DataModule."""

    model_name_or_path: str | None = None
    cache_dir: str | None = None
    hf_hub_token: str | None = None
    trust_remote_code: bool = False

    compute_dtype: Any | None = None
    block_diag_attn: bool = False
    model_max_length: int | None = None

    def __post_init__(self) -> None:
        if not self.model_name_or_path:
            raise ValueError("Please provide model.model_name_or_path.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["DataArguments", "ModelArguments"]
