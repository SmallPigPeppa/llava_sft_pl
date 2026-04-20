from __future__ import annotations

from enum import StrEnum, unique
from typing import TYPE_CHECKING, Optional, TypedDict, Union

from datasets import DatasetDict, concatenate_datasets

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from ..hparams import DataArguments


SLOTS = list[Union[str, set[str], dict[str, str]]]


@unique
class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


class DatasetModule(TypedDict, total=False):
    train_dataset: Optional[Union["Dataset", "IterableDataset"]]
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]


def merge_dataset(
    all_datasets: list[Union["Dataset", "IterableDataset"]], data_args: "DataArguments", seed: int
) -> Union["Dataset", "IterableDataset"]:
    """Merge local parquet datasets. The simplified project keeps concat only."""
    if len(all_datasets) == 0:
        raise ValueError("No train dataset was loaded.")
    if len(all_datasets) == 1:
        return all_datasets[0]
    if data_args.mix_strategy != "concat":
        raise ValueError("This simplified DataModule only supports data.mix_strategy='concat'.")
    if data_args.streaming:
        raise ValueError("Streaming is disabled; use local parquet datasets.")
    return concatenate_datasets(all_datasets)


def get_dataset_module(dataset: Union["Dataset", DatasetDict]) -> DatasetModule:
    module: DatasetModule = {}
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            module["train_dataset"] = dataset["train"]
        if "validation" in dataset:
            module["eval_dataset"] = dataset["validation"]
    else:
        module["train_dataset"] = dataset
    return module
