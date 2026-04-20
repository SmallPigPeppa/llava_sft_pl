from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

try:  # match either Lightning package name
    from lightning.pytorch import LightningDataModule
except Exception:  # pragma: no cover
    from pytorch_lightning import LightningDataModule

from .template import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER, Template

DATA_CONFIG = "dataset_info.json"


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _rank0() -> bool:
    try:
        import torch.distributed as dist

        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True


def _log(message: str) -> None:
    if _rank0():
        print(message)


def _has_tokenized_data(path: str | os.PathLike[str] | None) -> bool:
    return bool(path) and os.path.isdir(path) and len(os.listdir(path)) > 0


def _load_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _as_list(value: Any) -> list[Any] | None:
    value = _load_jsonish(value)
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return value if isinstance(value, list) else [value]


def _resolve_media(media: Any, media_dir: str | None) -> Any:
    if isinstance(media, str):
        candidate = os.path.join(media_dir or "", media) if media_dir and not os.path.isabs(media) else media
        return candidate if os.path.isfile(candidate) else media
    if isinstance(media, dict) and media.get("path") is not None:
        path = str(media["path"])
        candidate = os.path.join(media_dir or "", path) if media_dir and not os.path.isabs(path) else path
        if os.path.isfile(candidate):
            media = dict(media)
            media["path"] = candidate
    return media


def _resolve_media_list(value: Any, media_dir: str | None) -> list[Any] | None:
    items = _as_list(value)
    if not items:
        return None
    return [_resolve_media(item, media_dir) for item in items]


def _infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / max(1, source_len + target_len)))
    new_target_len = min(max_target_len, target_len)
    new_source_len = min(max(cutoff_len - new_target_len, 0), source_len)
    return new_source_len, new_target_len


@dataclass
class DatasetAttr:
    dataset_name: str
    split: str = "train"
    num_samples: int | None = None

    messages: str = "conversations"
    system: str | None = None
    tools: str | None = None
    images: str | None = None
    videos: str | None = None
    audios: str | None = None

    role_tag: str = "from"
    content_tag: str = "value"
    user_tag: str = "human"
    assistant_tag: str = "gpt"
    observation_tag: str = "observation"
    function_tag: str = "function_call"
    system_tag: str = "system"

    def __repr__(self) -> str:
        return self.dataset_name

    @classmethod
    def from_info(cls, name: str, info: dict[str, Any]) -> "DatasetAttr":
        if info.get("formatting", "sharegpt") != "sharegpt":
            raise ValueError("Only ShareGPT/LLaVA SFT data is kept in this slim project.")
        if info.get("ranking", False):
            raise ValueError("Ranking/DPO data was removed; use ShareGPT SFT data.")
        if "file_name" not in info:
            raise ValueError(f"Dataset {name!r} must define `file_name` in {DATA_CONFIG}.")

        attr = cls(
            dataset_name=info["file_name"],
            split=info.get("split", "train"),
            num_samples=info.get("num_samples"),
        )
        for key in ["messages", "system", "tools", "images", "videos", "audios"]:
            if key in info.get("columns", {}):
                setattr(attr, key, info["columns"][key])
        for key in [
            "role_tag",
            "content_tag",
            "user_tag",
            "assistant_tag",
            "observation_tag",
            "function_tag",
            "system_tag",
        ]:
            if key in info.get("tags", {}):
                setattr(attr, key, info["tags"][key])
        return attr


def _dataset_attrs(dataset_names: list[str], dataset_dir: str | dict[str, Any]) -> list[DatasetAttr]:
    if isinstance(dataset_dir, dict):
        dataset_info = dataset_dir
    else:
        config_path = os.path.join(dataset_dir, DATA_CONFIG)
        if not os.path.isfile(config_path):
            # Allow direct parquet path(s) without dataset_info.json.
            return [DatasetAttr(dataset_name=name) for name in dataset_names]
        with open(config_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)

    attrs = []
    for name in dataset_names:
        if name not in dataset_info:
            # Convenience fallback for direct local paths mixed with dataset_info configs.
            path = os.path.join(str(dataset_dir), name)
            if os.path.exists(path) or os.path.exists(name):
                attrs.append(DatasetAttr(dataset_name=name))
                continue
            raise ValueError(f"Undefined dataset {name!r} in {DATA_CONFIG}.")
        attrs.append(DatasetAttr.from_info(name, dataset_info[name]))
    return attrs


def _collect_parquet_files(dataset_dir: str, dataset_name: str) -> list[str]:
    local_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isdir(local_path):
        files = []
        for root, _, filenames in os.walk(local_path):
            for filename in sorted(filenames):
                path = os.path.join(root, filename)
                if filename.endswith(".parquet"):
                    files.append(path)
                elif filename.startswith(".") or filename.startswith("_"):
                    continue
                else:
                    raise ValueError(f"Only parquet files are supported, found {path!r}.")
    elif os.path.isfile(local_path):
        if not local_path.endswith(".parquet"):
            raise ValueError(f"Only parquet files are supported, found {local_path!r}.")
        files = [local_path]
    else:
        raise ValueError(f"Parquet file or directory not found: {local_path}")

    if not files:
        raise ValueError(f"No parquet files found under {local_path}.")
    return files


def _convert_sharegpt(example: dict[str, Any], attr: DatasetAttr, media_dir: str | None) -> dict[str, Any]:
    raw_messages = _load_jsonish(example[attr.messages])
    if not isinstance(raw_messages, list):
        raise TypeError(f"ShareGPT messages must be a list, got {type(raw_messages)!r}.")

    role_map = {
        attr.user_tag: "user",
        attr.assistant_tag: "assistant",
        attr.observation_tag: "observation",
        attr.function_tag: "function",
        attr.system_tag: "system",
    }
    if raw_messages and raw_messages[0].get(attr.role_tag) == attr.system_tag:
        system = raw_messages[0].get(attr.content_tag, "")
        raw_messages = raw_messages[1:]
    else:
        system = example.get(attr.system, "") if attr.system else ""

    aligned: list[dict[str, str]] = []
    valid = True
    for i, message in enumerate(raw_messages):
        raw_role = message.get(attr.role_tag)
        role = role_map.get(raw_role)
        expected = {"user", "observation"} if i % 2 == 0 else {"assistant", "function"}
        if role not in expected:
            valid = False
            break
        aligned.append({"role": "user" if role == "observation" else "assistant" if role == "function" else role, "content": str(message.get(attr.content_tag, ""))})

    if not valid or len(aligned) < 2 or len(aligned) % 2 != 0:
        prompt, response = [], []
    else:
        prompt, response = aligned[:-1], aligned[-1:]

    images = _resolve_media_list(example.get(attr.images), media_dir) if attr.images else None
    videos = _resolve_media_list(example.get(attr.videos), media_dir) if attr.videos else None
    audios = _resolve_media_list(example.get(attr.audios), media_dir) if attr.audios else None

    if images:
        placeholder_count = sum(m["content"].count(IMAGE_PLACEHOLDER) for m in prompt + response)
        missing = max(0, len(images) - placeholder_count)
        if missing:
            for message in prompt:
                if message["role"] == "user":
                    message["content"] = (IMAGE_PLACEHOLDER + "\n") * missing + message["content"]
                    break
    if videos and not any(VIDEO_PLACEHOLDER in m["content"] for m in prompt + response):
        raise ValueError("Video columns are present, but video preprocessing was removed from this slim build.")
    if audios and not any(AUDIO_PLACEHOLDER in m["content"] for m in prompt + response):
        raise ValueError("Audio columns are present, but audio preprocessing was removed from this slim build.")

    tools = example.get(attr.tools, "") if attr.tools else ""
    if isinstance(tools, (dict, list)):
        tools = json.dumps(tools, ensure_ascii=False)

    return {"_prompt": prompt, "_response": response, "_system": system, "_tools": tools, "_images": images, "_videos": videos, "_audios": audios}


def _pad_to_multiple(length: int, multiple: int | None) -> int:
    if not multiple:
        return length
    return ((length + multiple - 1) // multiple) * multiple


@dataclass
class SFTProcessor:
    template: Template
    tokenizer: Any
    processor: Any
    data_args: Any

    def preprocess_batch(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs: dict[str, list[Any]] = {"input_ids": [], "attention_mask": [], "labels": [], "images": [], "videos": [], "audios": []}
        for i in range(len(examples["_prompt"])):
            prompt = examples["_prompt"][i]
            response = examples["_response"][i]
            if len(prompt) % 2 != 1 or len(response) != 1:
                continue

            images = examples["_images"][i] or []
            videos = examples["_videos"][i] or []
            audios = examples["_audios"][i] or []
            messages = self.template.process_messages(prompt + response, images, videos, audios, self.processor)
            pairs = self.template.encode_multiturn(self.tokenizer, messages, examples["_system"][i], examples["_tools"][i])

            input_ids: list[int] = []
            labels: list[int] = []
            total = 1 if self.template.efficient_eos else 0
            pairs_to_encode = list(reversed(pairs)) if self.data_args.mask_history else pairs

            for turn_idx, (source_ids, target_ids) in enumerate(pairs_to_encode):
                if total >= self.data_args.cutoff_len:
                    break
                source_len, target_len = _infer_seqlen(len(source_ids), len(target_ids), self.data_args.cutoff_len - total)
                source_ids, target_ids = source_ids[:source_len], target_ids[:target_len]
                source_labels = source_ids if self.data_args.train_on_prompt else [IGNORE_INDEX] * source_len
                target_labels = [IGNORE_INDEX] * target_len if self.data_args.mask_history and turn_idx != 0 else target_ids
                if self.data_args.mask_history:
                    input_ids = source_ids + target_ids + input_ids
                    labels = source_labels + target_labels + labels
                else:
                    input_ids.extend(source_ids + target_ids)
                    labels.extend(source_labels + target_labels)
                total += source_len + target_len

            if self.template.efficient_eos and self.tokenizer.eos_token_id is not None:
                input_ids.append(self.tokenizer.eos_token_id)
                labels.append(self.tokenizer.eos_token_id)

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(images or None)
            model_inputs["videos"].append(None)
            model_inputs["audios"].append(None)
        return model_inputs

    def print_example(self, example: dict[str, Any]) -> None:
        valid_labels = [x for x in example["labels"] if x != IGNORE_INDEX]
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(self.tokenizer.decode(valid_labels, skip_special_tokens=False)))


@dataclass
class SFTCollator:
    tokenizer: Any
    processor: Any
    template: Template
    label_pad_token_id: int = IGNORE_INDEX
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = _pad_to_multiple(max(len(f["input_ids"]) for f in features), self.pad_to_multiple_of)
        input_ids, attention_mask, labels = [], [], []
        flat_images: list[Any] = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [self.label_pad_token_id] * pad_len)
            flat_images.extend(feature.get("images") or [])
            if feature.get("videos"):
                raise ValueError("Video preprocessing was removed from this slim build.")
            if feature.get("audios"):
                raise ValueError("Audio preprocessing was removed from this slim build.")

        batch: dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if flat_images:
            batch.update(self.template.image_inputs(flat_images, self.processor))
        return batch


@dataclass
class _OnlineTokenizedDataset(IterableDataset):
    dataset: Any
    processor: SFTProcessor
    preprocessing_batch_size: int
    seed: int
    shuffle: bool = True

    def __post_init__(self) -> None:
        self.epoch = 0
        if self.preprocessing_batch_size <= 0:
            raise ValueError("preprocessing_batch_size must be > 0.")

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _dist() -> tuple[int, int]:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return dist.get_world_size(), dist.get_rank()
        except Exception:
            pass
        return 1, 0

    @staticmethod
    def _batchify(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
        keys = list(rows[0].keys()) if rows else []
        return {key: [row.get(key) for row in rows] for key in keys}

    @staticmethod
    def _iter_rows(columns: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
        if not columns:
            return
        n = len(next(iter(columns.values())))
        for i in range(n):
            yield {key: value[i] for key, value in columns.items()}

    def _yield_processed(self, rows: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        yield from self._iter_rows(self.processor.preprocess_batch(self._batchify(rows)))

    def __iter__(self) -> Iterable[dict[str, Any]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)

        world, rank = self._dist()
        worker = get_worker_info()
        num_workers, worker_id = (worker.num_workers, worker.id) if worker else (1, 0)
        shards = max(1, world * num_workers)
        shard_id = rank * num_workers + worker_id

        buf: list[dict[str, Any]] = []
        for pos, idx in enumerate(indices):
            if pos % shards != shard_id:
                continue
            buf.append(self.dataset[int(idx)])
            if len(buf) >= self.preprocessing_batch_size:
                yield from self._yield_processed(buf)
                buf = []
        if buf:
            yield from self._yield_processed(buf)


class ParquetSFTDataModule(LightningDataModule):
    """Small parquet-only, ShareGPT/LLaVA-only Lightning DataModule."""

    def __init__(
        self,
        template: Template,
        model_args: Any,
        data_args: Any,
        training_args: Any,
        stage: str,
        tokenizer: Any,
        processor: Any = None,
        model: Any = None,
        preprocessing_mode: str = "online",
        train_batch_size: int | None = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        if stage != "sft":
            raise ValueError("Only stage='sft' is supported.")
        if preprocessing_mode not in {"online", "offline"}:
            raise ValueError("preprocessing_mode must be 'online' or 'offline'.")
        if preprocessing_mode == "offline" and not data_args.tokenized_path:
            raise ValueError("Offline preprocessing requires data.tokenized_path.")
        if not data_args.dataset:
            raise ValueError("Please set data.dataset.")

        self.template = template
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.stage = stage
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.preprocessing_mode = preprocessing_mode
        self.train_batch_size = train_batch_size
        self.shuffle = shuffle
        self.train_dataset = None
        self.data_collator = None

    def _main_process_first(self, desc: str):
        fn = _get_attr(self.training_args, "main_process_first", None)
        if callable(fn):
            return fn(desc=desc, local=(not bool(self.data_args.data_shared_file_system)))
        return nullcontext()

    def _load_one(self, attr: DatasetAttr):
        _log(f"Loading parquet dataset {attr}...")
        files = _collect_parquet_files(self.data_args.dataset_dir, attr.dataset_name)
        dataset = load_dataset(
            path="parquet",
            data_files=files,
            split=attr.split,
            cache_dir=_get_attr(self.model_args, "cache_dir", None),
            token=_get_attr(self.model_args, "hf_hub_token", None),
            num_proc=_get_attr(self.data_args, "preprocessing_num_workers", None),
        )
        if attr.num_samples is not None:
            indexes = np.random.permutation(len(dataset))[: int(attr.num_samples)]
            dataset = dataset.select(indexes)
        if self.data_args.max_samples is not None:
            dataset = dataset.select(range(min(int(self.data_args.max_samples), len(dataset))))

        columns = list(next(iter(dataset)).keys())
        kwargs = dict(
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=(not self.data_args.overwrite_cache) or (_get_attr(self.training_args, "local_process_index", 0) != 0),
            desc="Converting ShareGPT parquet",
        )
        media_dir = self.data_args.media_dir
        return dataset.map(lambda ex: _convert_sharegpt(ex, attr, media_dir), batched=False, remove_columns=columns, **kwargs)

    def _load_raw_dataset(self):
        datasets = [self._load_one(attr) for attr in _dataset_attrs(self.data_args.dataset, self.data_args.dataset_dir)]
        if not datasets:
            raise ValueError("No dataset was loaded.")
        if len(datasets) == 1:
            return datasets[0]
        merged = concatenate_datasets(datasets)
        return merged.shuffle(seed=int(_get_attr(self.training_args, "seed", 42)))

    def _build_processor(self) -> SFTProcessor:
        return SFTProcessor(self.template, self.tokenizer, self.processor, self.data_args)

    def _build_collator(self) -> SFTCollator:
        pad_id = IGNORE_INDEX if bool(self.data_args.ignore_pad_token_for_loss) else self.tokenizer.pad_token_id
        return SFTCollator(tokenizer=self.tokenizer, processor=self.processor, template=self.template, label_pad_token_id=pad_id)

    def _build_and_save_offline(self) -> None:
        with self._main_process_first("build tokenized parquet dataset"):
            if _has_tokenized_data(self.data_args.tokenized_path):
                _log(f"Tokenized dataset already exists at {self.data_args.tokenized_path}.")
                return
            raw = self._load_raw_dataset()
            proc = self._build_processor()
            columns = list(next(iter(raw)).keys())
            tokenized = raw.map(
                proc.preprocess_batch,
                batched=True,
                batch_size=self.data_args.preprocessing_batch_size,
                remove_columns=columns,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=(not self.data_args.overwrite_cache) or (_get_attr(self.training_args, "local_process_index", 0) != 0),
                desc="Tokenizing SFT dataset",
            )
            DatasetDict({"train": tokenized}).save_to_disk(self.data_args.tokenized_path)
            _log(f"Tokenized dataset saved at {self.data_args.tokenized_path}.")

    def prepare_data(self) -> None:
        if self.preprocessing_mode == "offline":
            self._build_and_save_offline()

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return
        self.data_collator = self._build_collator()
        if self.preprocessing_mode == "offline":
            if not _has_tokenized_data(self.data_args.tokenized_path):
                self._build_and_save_offline()
            data = load_from_disk(self.data_args.tokenized_path)
            self.train_dataset = data["train"] if isinstance(data, DatasetDict) else data
            _log(f"Loaded tokenized dataset from {self.data_args.tokenized_path}.")
        else:
            with self._main_process_first("load parquet dataset"):
                raw = self._load_raw_dataset()
            proc = self._build_processor()
            self.train_dataset = _OnlineTokenizedDataset(
                dataset=raw,
                processor=proc,
                preprocessing_batch_size=int(self.data_args.preprocessing_batch_size),
                seed=int(_get_attr(self.training_args, "seed", 42)),
                shuffle=self.shuffle,
            )
            if bool(_get_attr(self.training_args, "should_log", True)):
                iterator = iter(self.train_dataset)
                try:
                    proc.print_example(next(iterator))
                except StopIteration as exc:
                    raise RuntimeError("Cannot find valid samples; check the ShareGPT/LLaVA parquet format.") from exc

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
            prefetch = _get_attr(self.training_args, "dataloader_prefetch_factor", None)
            if prefetch is not None:
                kwargs["prefetch_factor"] = prefetch
        if not isinstance(self.train_dataset, IterableDataset):
            kwargs["shuffle"] = self.shuffle
        return DataLoader(**kwargs)

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None
