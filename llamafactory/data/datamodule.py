from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

try:
    from lightning import LightningDataModule
except Exception:  # pragma: no cover
    from lightning.pytorch import LightningDataModule  # type: ignore

from ..extras import DATA_CONFIG, IGNORE_INDEX, IMAGE_PLACEHOLDER, get_logger
from .collator import SFTDataCollatorWith4DAttentionMask
from .template import Role

logger = get_logger(__name__)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


@dataclass
class DatasetAttr:
    load_from: Literal["file"]
    dataset_name: str
    formatting: Literal["sharegpt"] = "sharegpt"
    split: str = "train"
    num_samples: int | None = None
    system: str | None = None
    images: str | None = None
    messages: str | None = "conversations"
    role_tag: str = "from"
    content_tag: str = "value"
    user_tag: str = "human"
    assistant_tag: str = "gpt"
    system_tag: str = "system"

    def __repr__(self) -> str:
        return self.dataset_name

    def join(self, info: dict[str, Any]) -> None:
        self.formatting = info.get("formatting", "sharegpt")
        if self.formatting != "sharegpt":
            raise ValueError("Only ShareGPT/LLaVA-style SFT data is kept.")
        if info.get("ranking"):
            raise ValueError("Only standard SFT data is kept.")
        self.split = info.get("split", "train")
        self.num_samples = info.get("num_samples")
        columns = info.get("columns", {})
        tags = info.get("tags", {})
        for key in ["messages", "system", "images"]:
            if key in columns:
                setattr(self, key, columns[key])
        for key in ["role_tag", "content_tag", "user_tag", "assistant_tag", "system_tag"]:
            if key in tags:
                setattr(self, key, tags[key])


def get_dataset_list(dataset_names: list[str] | None, dataset_dir: str | dict[str, Any]) -> list[DatasetAttr]:
    if not dataset_names:
        return []
    if isinstance(dataset_dir, dict):
        dataset_info = dataset_dir
    else:
        info_path = os.path.join(dataset_dir, DATA_CONFIG)
        with open(info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)

    attrs: list[DatasetAttr] = []
    for name in dataset_names:
        if name not in dataset_info:
            raise ValueError(f"Undefined dataset {name!r} in {DATA_CONFIG}.")
        item = dataset_info[name]
        attr = DatasetAttr("file", dataset_name=item["file_name"])
        attr.join(item)
        attrs.append(attr)
    return attrs


def _merge_datasets(datasets: list[Any], data_args: Any) -> Any:
    if not datasets:
        raise ValueError("No train dataset was loaded.")
    if len(datasets) == 1:
        return datasets[0]
    if _get_attr(data_args, "mix_strategy", "concat") != "concat":
        raise ValueError("Only data.mix_strategy='concat' is kept.")
    return concatenate_datasets(datasets)


@dataclass
class ShareGPTConverter:
    attr: DatasetAttr
    data_args: Any

    def _resolve_media(self, media: Any) -> Any:
        if isinstance(media, str):
            path = media if os.path.isabs(media) else os.path.join(self.data_args.media_dir, media)
            return path if os.path.isfile(path) else media
        if isinstance(media, dict) and media.get("path") is not None:
            path = str(media["path"])
            full_path = path if os.path.isabs(path) else os.path.join(self.data_args.media_dir, path)
            return {**media, "path": full_path} if os.path.isfile(full_path) else media
        return media

    def _image_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [self._resolve_media(item) for item in items]

    @staticmethod
    def _load_messages(raw: Any) -> list[dict[str, Any]]:
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, list):
            raise TypeError(f"ShareGPT messages must be a list, got {type(raw)!r}.")
        return raw

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        attr = self.attr
        messages = self._load_messages(example[attr.messages])
        if messages and messages[0].get(attr.role_tag) == attr.system_tag:
            system = messages[0].get(attr.content_tag, "")
            messages = messages[1:]
        else:
            system = example.get(attr.system, "") if attr.system else ""

        aligned: list[dict[str, str]] = []
        broken = False
        for idx, message in enumerate(messages):
            role = message.get(attr.role_tag)
            expected = attr.user_tag if idx % 2 == 0 else attr.assistant_tag
            if role != expected:
                logger.warning_rank0("Dropped sample with invalid role order: %s", messages)
                broken = True
                break
            aligned.append(
                {
                    "role": Role.USER.value if role == attr.user_tag else Role.ASSISTANT.value,
                    "content": str(message.get(attr.content_tag, "")),
                }
            )
        if len(aligned) % 2 != 0:
            logger.warning_rank0("Dropped sample with incomplete user/assistant pair: %s", messages)
            broken = True

        prompt, response = ([], []) if broken else (aligned[:-1], aligned[-1:])
        images = self._image_list(example.get(attr.images)) if attr.images else []
        if images:
            placeholder_count = sum(m["content"].count(IMAGE_PLACEHOLDER) for m in prompt + response)
            missing = max(0, len(images) - placeholder_count)
            if missing:
                for message in prompt:
                    if message["role"] == Role.USER.value:
                        message["content"] = (IMAGE_PLACEHOLDER + "\n") * missing + message["content"]
                        break

        return {"_prompt": prompt, "_response": response, "_system": system, "_images": images}


def _align_dataset(dataset: Any, attr: DatasetAttr, data_args: Any, training_args: Any) -> Any:
    columns = list(getattr(dataset, "column_names", None) or next(iter(dataset)).keys())
    return dataset.map(
        ShareGPTConverter(attr, data_args),
        batched=False,
        remove_columns=columns,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=(not data_args.overwrite_cache) or (_get_attr(training_args, "local_process_index", 0) != 0),
        desc="Converting ShareGPT format",
    )


def truncate_lengths(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))
    target_len = min(max_target_len, target_len)
    return min(max(cutoff_len - target_len, 0), source_len), target_len


@dataclass
class SupervisedDatasetProcessor:
    template: Any
    tokenizer: Any
    processor: Any | None
    data_args: Any

    def _encode_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: str | None,
        images: list[Any],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids([], [], images, self.tokenizer, self.processor)
        pairs = self.template.encode_multiturn(self.tokenizer, messages, system)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            pairs = pairs[::-1]

        for turn_idx, (source_ids, target_ids) in enumerate(pairs):
            if total_length >= self.data_args.cutoff_len:
                break
            source_len, target_len = truncate_lengths(len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length)
            source_ids, target_ids = source_ids[:source_len], target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos and turn_idx != 0:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len
            target_label = [IGNORE_INDEX] * target_len if self.data_args.mask_history and turn_idx != 0 else target_ids

            if self.data_args.mask_history:
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)
        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                continue
            images = examples["_images"][i] or []
            input_ids, labels = self._encode_example(
                examples["_prompt"][i],
                examples["_response"][i],
                examples["_system"][i],
                images,
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(images)
        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        labels = [x for x in example["labels"] if x != IGNORE_INDEX]
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(self.tokenizer.decode(labels, skip_special_tokens=False)))


@dataclass
class _OnlineTokenizedIterableDataset(IterableDataset):
    dataset: Any
    dataset_processor: SupervisedDatasetProcessor
    preprocessing_batch_size: int
    seed: int
    shuffle: bool = True

    def __post_init__(self) -> None:
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _dist_worker() -> tuple[int, int, int, int]:
        world_size, rank = 1, 0
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                world_size, rank = dist.get_world_size(), dist.get_rank()
        except Exception:
            pass
        worker = get_worker_info()
        return world_size, rank, (worker.num_workers if worker else 1), (worker.id if worker else 0)

    @staticmethod
    def _batchify(examples: list[dict[str, Any]]) -> dict[str, list[Any]]:
        return {key: [example.get(key) for example in examples] for key in examples[0]}

    def _yield_processed(self, examples: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        model_inputs = self.dataset_processor.preprocess_dataset(self._batchify(examples))
        if not model_inputs:
            return
        first_key = next(iter(model_inputs))
        for i in range(len(model_inputs[first_key])):
            yield {key: value[i] for key, value in model_inputs.items()}

    def __iter__(self) -> Iterable[dict[str, Any]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        world_size, rank, num_workers, worker_id = self._dist_worker()
        shard_count = max(1, world_size * num_workers)
        shard_id = rank * num_workers + worker_id
        buffer: list[dict[str, Any]] = []
        for position, sample_index in enumerate(indices):
            if position % shard_count != shard_id:
                continue
            buffer.append(self.dataset[int(sample_index)])
            if len(buffer) >= max(1, self.preprocessing_batch_size):
                yield from self._yield_processed(buffer)
                buffer.clear()
        if buffer:
            yield from self._yield_processed(buffer)


class ParquetSFTDataModule(LightningDataModule):
    """SFT-only DataModule: parquet -> ShareGPT align -> online tokenize -> image collate."""

    def __init__(
        self,
        template: Any,
        model_args: Any,
        data_args: Any,
        training_args: Any,
        tokenizer: Any,
        processor: Any | None = None,
        model: Any | None = None,
        train_batch_size: int | None = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        if data_args.dataset is None:
            raise ValueError("Please set data.dataset to one or more parquet dataset names or paths.")
        self.template = template
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.train_batch_size = train_batch_size
        self.shuffle = shuffle
        self.train_dataset = None
        self.data_collator = None

    def _main_first(self, desc: str):
        fn = _get_attr(self.training_args, "main_process_first", None)
        return fn(desc=desc, local=True) if callable(fn) else nullcontext()

    def _collect_parquet_files(self, attr: DatasetAttr) -> list[str]:
        local_path = os.path.join(self.data_args.dataset_dir, attr.dataset_name)
        files: list[str] = []
        if os.path.isdir(local_path):
            for root, _, names in os.walk(local_path):
                files.extend(os.path.join(root, name) for name in sorted(names) if name.endswith(".parquet"))
        elif os.path.isfile(local_path) and local_path.endswith(".parquet"):
            files.append(local_path)
        else:
            raise ValueError(f"Parquet file or directory {local_path!r} not found.")
        if not files:
            raise ValueError(f"No parquet files found under {local_path!r}.")
        return files

    def _dataset_attrs(self) -> list[DatasetAttr]:
        try:
            return get_dataset_list(self.data_args.dataset, self.data_args.dataset_dir)
        except (FileNotFoundError, ValueError) as err:
            attrs: list[DatasetAttr] = []
            for name in self.data_args.dataset:
                local_path = os.path.join(self.data_args.dataset_dir, name)
                if os.path.isdir(local_path) or local_path.endswith(".parquet"):
                    attrs.append(DatasetAttr("file", dataset_name=name))
                else:
                    raise err
            return attrs

    def _load_one_dataset(self, attr: DatasetAttr) -> Any:
        logger.info_rank0("Loading parquet dataset %s", attr)
        dataset = load_dataset(
            path="parquet",
            data_files=self._collect_parquet_files(attr),
            split=attr.split,
            cache_dir=_get_attr(self.model_args, "cache_dir", None),
            token=_get_attr(self.model_args, "hf_hub_token", None),
            num_proc=_get_attr(self.data_args, "preprocessing_num_workers", None),
        )
        if attr.num_samples is not None:
            indices = np.random.permutation(len(dataset))[: attr.num_samples]
            if len(indices) < attr.num_samples:
                indices = np.concatenate([indices, np.random.choice(len(dataset), attr.num_samples - len(indices))])
            dataset = dataset.select(indices)
        if self.data_args.max_samples is not None:
            dataset = dataset.select(range(min(self.data_args.max_samples, len(dataset))))
        return _align_dataset(dataset, attr, self.data_args, self.training_args)

    def _load_train_dataset(self) -> Any:
        return _merge_datasets([self._load_one_dataset(attr) for attr in self._dataset_attrs()], self.data_args)

    def _collator(self) -> SFTDataCollatorWith4DAttentionMask:
        import torch
        return SFTDataCollatorWith4DAttentionMask(
            template=self.template,
            model=self.model,
            tokenizer=self.tokenizer,
            processor=self.processor,
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id,
            compute_dtype=_get_attr(self.model_args, "compute_dtype", None) or torch.float32,
        )

    def prepare_data(self) -> None:
        return None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return
        self.data_collator = self._collator()
        with self._main_first("load parquet dataset"):
            raw_dataset = self._load_train_dataset()
        proc = SupervisedDatasetProcessor(self.template, self.tokenizer, self.processor, self.data_args)
        self.train_dataset = _OnlineTokenizedIterableDataset(
            raw_dataset,
            proc,
            self.data_args.preprocessing_batch_size,
            int(_get_attr(self.training_args, "seed", 42)),
            self.shuffle,
        )
        if _get_attr(self.training_args, "should_log", True):
            try:
                proc.print_data_example(next(iter(self.train_dataset)))
            except StopIteration as exc:
                raise RuntimeError("Cannot find valid SFT samples; check the parquet data format.") from exc

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None or self.data_collator is None:
            self.setup("fit")
        num_workers = int(_get_attr(self.training_args, "dataloader_num_workers", 0) or 0)
        kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size or int(_get_attr(self.training_args, "per_device_train_batch_size", 1)),
            collate_fn=self.data_collator,
            num_workers=num_workers,
            pin_memory=bool(_get_attr(self.training_args, "dataloader_pin_memory", False)),
            drop_last=bool(_get_attr(self.training_args, "dataloader_drop_last", False)),
        )
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(_get_attr(self.training_args, "dataloader_persistent_workers", False))
            if _get_attr(self.training_args, "dataloader_prefetch_factor", None) is not None:
                kwargs["prefetch_factor"] = self.training_args.dataloader_prefetch_factor
        return DataLoader(**kwargs)
