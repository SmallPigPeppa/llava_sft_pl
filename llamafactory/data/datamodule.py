from __future__ import annotations

import bisect
import json
import os
import random
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Optional

import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

try:
    from lightning import LightningDataModule
except Exception:  # pragma: no cover
    from lightning.pytorch import LightningDataModule  # type: ignore

from ..extras import DATA_CONFIG, IGNORE_INDEX, IMAGE_PLACEHOLDER, get_logger, has_tokenized_data
from .collator import SFTDataCollatorWith4DAttentionMask
from .template import Role

logger = get_logger(__name__)
MAX_SU_SEQ_IDX = 2**32


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


@dataclass
class DatasetAttr:
    load_from: Literal["file"]
    dataset_name: str
    formatting: Literal["sharegpt"] = "sharegpt"
    ranking: bool = False
    split: str = "train"
    num_samples: int | None = None
    system: str | None = None
    tools: str | None = None
    images: str | None = None
    videos: str | None = None
    audios: str | None = None
    messages: str | None = "conversations"
    role_tag: str | None = "from"
    content_tag: str | None = "value"
    user_tag: str | None = "human"
    assistant_tag: str | None = "gpt"
    observation_tag: str | None = "observation"
    function_tag: str | None = "function_call"
    system_tag: str | None = "system"

    def __repr__(self) -> str:
        return self.dataset_name

    def join(self, info: dict[str, Any]) -> None:
        self.formatting = info.get("formatting", "sharegpt")
        if self.formatting != "sharegpt":
            raise ValueError("Only ShareGPT/LLaVA-style SFT datasets are kept.")
        self.ranking = bool(info.get("ranking", False))
        if self.ranking:
            raise ValueError("Ranking/DPO-style datasets were removed; use standard ShareGPT SFT data.")
        self.split = info.get("split", "train")
        self.num_samples = info.get("num_samples")
        columns = info.get("columns", {})
        tags = info.get("tags", {})
        for key in ["messages", "system", "tools", "images", "videos", "audios"]:
            if key in columns:
                setattr(self, key, columns[key])
        for key in ["role_tag", "content_tag", "user_tag", "assistant_tag", "observation_tag", "function_tag", "system_tag"]:
            if key in tags:
                setattr(self, key, tags[key])


def get_dataset_list(dataset_names: list[str] | None, dataset_dir: str | dict[str, Any]) -> list[DatasetAttr]:
    if not dataset_names:
        return []
    if isinstance(dataset_dir, dict):
        dataset_info = dataset_dir
    else:
        path = os.path.join(dataset_dir, DATA_CONFIG)
        try:
            with open(path, "r", encoding="utf-8") as f:
                dataset_info = json.load(f)
        except Exception as exc:
            raise ValueError(f"Cannot open {path}: {exc}") from exc

    attrs: list[DatasetAttr] = []
    for name in dataset_names:
        if name not in dataset_info:
            raise ValueError(f"Undefined dataset {name} in {DATA_CONFIG}.")
        item = dataset_info[name]
        if "file_name" not in item:
            raise ValueError(f"Dataset {name} must define `file_name` in {DATA_CONFIG}.")
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
        raise ValueError("This simplified DataModule only supports data.mix_strategy='concat'.")
    return concatenate_datasets(datasets)


def _train_split(dataset: Any) -> Any:
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError("Tokenized dataset does not contain a train split.")
        return dataset["train"]
    return dataset


@dataclass
class ShareGPTConverter:
    attr: DatasetAttr
    data_args: Any

    def _resolve_media(self, media: Any) -> Any:
        if isinstance(media, str):
            candidate = os.path.join(self.data_args.media_dir, media) if not os.path.isabs(media) else media
            return candidate if os.path.isfile(candidate) else media
        if isinstance(media, dict) and media.get("path") is not None:
            path = str(media["path"])
            candidate = os.path.join(self.data_args.media_dir, path) if not os.path.isabs(path) else path
            return {**media, "path": candidate} if os.path.isfile(candidate) else media
        if isinstance(media, list):
            return [self._resolve_media(x) for x in media]
        return media

    def _media_list(self, value: Any) -> list[Any] | None:
        if value is None:
            return None
        items = value if isinstance(value, list) else [value]
        return [self._resolve_media(item) for item in items] or None

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
        role_map = {
            attr.user_tag: Role.USER.value,
            attr.assistant_tag: Role.ASSISTANT.value,
            attr.observation_tag: Role.OBSERVATION.value,
            attr.function_tag: Role.FUNCTION.value,
            attr.system_tag: Role.SYSTEM.value,
        }
        if attr.system_tag and messages and messages[0][attr.role_tag] == attr.system_tag:
            system = messages[0][attr.content_tag]
            messages = messages[1:]
        else:
            system = example[attr.system] if attr.system else ""

        aligned, broken = [], False
        for turn_idx, message in enumerate(messages):
            role = message[attr.role_tag]
            expected = (attr.user_tag, attr.observation_tag) if turn_idx % 2 == 0 else (attr.assistant_tag, attr.function_tag)
            if role not in expected:
                logger.warning_rank0("Invalid role tag in %s", messages)
                broken = True
                break
            aligned.append({"role": role_map[role], "content": message[attr.content_tag]})
        if len(aligned) % 2 != 0:
            logger.warning_rank0("Invalid message count in %s", messages)
            broken = True

        prompt, response = ([], []) if broken else (aligned[:-1], aligned[-1:])
        images = self._media_list(example[attr.images]) if attr.images else None
        if images:
            placeholders = sum(m["content"].count(IMAGE_PLACEHOLDER) for m in prompt + response)
            missing = max(0, len(images) - placeholders)
            if missing:
                for message in prompt:
                    if message["role"] == Role.USER.value:
                        message["content"] = (IMAGE_PLACEHOLDER + "\n") * missing + message["content"]
                        break

        tools = example[attr.tools] if attr.tools else ""
        if isinstance(tools, (dict, list)):
            tools = json.dumps(tools, ensure_ascii=False)
        return {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": tools,
            "_images": images,
            "_videos": self._media_list(example[attr.videos]) if attr.videos else None,
            "_audios": self._media_list(example[attr.audios]) if attr.audios else None,
        }


def _align_dataset(dataset: Any, attr: DatasetAttr, data_args: Any, training_args: Any) -> Any:
    column_names = list(next(iter(dataset)).keys())
    kwargs = dict(
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=(not data_args.overwrite_cache) or (_get_attr(training_args, "local_process_index", 0) != 0),
        desc="Converting ShareGPT format",
    )
    return dataset.map(ShareGPTConverter(attr, data_args), batched=False, remove_columns=column_names, **kwargs)


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))
    new_target_len = min(max_target_len, target_len)
    return min(max(cutoff_len - new_target_len, 0), source_len), new_target_len


def greedy_knapsack(lengths: list[int], capacity: int) -> list[list[int]]:
    lengths = sorted(lengths)
    packs: list[list[int]] = []
    while lengths:
        pack, remaining = [], capacity
        while True:
            index = bisect.bisect(lengths, remaining) - 1
            if index < 0:
                break
            remaining -= lengths[index]
            pack.append(lengths.pop(index))
        packs.append(pack)
    return packs


@dataclass
class SupervisedDatasetProcessor:
    template: Any
    tokenizer: Any
    processor: Any | None
    data_args: Any

    def _encode_example(self, prompt: list[dict[str, str]], response: list[dict[str, str]], system: str | None, tools: str | None, images: list[Any], videos: list[Any], audios: list[Any]) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids([], [], images, videos, audios, self.tokenizer, self.processor)
        pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            pairs = pairs[::-1]

        for turn_idx, (source_ids, target_ids) in enumerate(pairs):
            if total_length >= self.data_args.cutoff_len:
                break
            source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length)
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
                logger.warning_rank0("Dropped invalid example: %s", examples["_prompt"][i] + examples["_response"][i])
                continue
            input_ids, labels = self._encode_example(
                examples["_prompt"][i],
                examples["_response"][i],
                examples["_system"][i],
                examples["_tools"][i],
                examples["_images"][i] or [],
                examples["_videos"][i] or [],
                examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])
        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        labels = [x for x in example["labels"] if x != IGNORE_INDEX]
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(self.tokenizer.decode(labels, skip_special_tokens=False)))


@dataclass
class PackingParams:
    sequence_boundaries: list[int]
    image_subseq_ids: list[int]
    video_subseq_ids: list[int]
    audio_subseq_ids: list[int]
    right_padding_length: int


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        encoded, lengths = [], []
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                continue
            input_ids, labels = self._encode_example(
                examples["_prompt"][i], examples["_response"][i], examples["_system"][i], examples["_tools"][i],
                examples["_images"][i] or [], examples["_videos"][i] or [], examples["_audios"][i] or []
            )
            if len(input_ids) <= self.data_args.cutoff_len:
                encoded.append((input_ids, labels, examples["_images"][i] or [], examples["_videos"][i] or [], examples["_audios"][i] or []))
                lengths.append(len(input_ids))
            else:
                logger.warning_rank0("Dropped lengthy example with length %s > %s.", len(input_ids), self.data_args.cutoff_len)

        by_len: dict[int, list[int]] = defaultdict(list)
        for idx, length in enumerate(lengths):
            by_len[length].append(idx)

        model_inputs = defaultdict(list)
        for pack in greedy_knapsack(lengths, self.data_args.cutoff_len):
            packed_ids, packed_labels, packed_images, packed_videos, packed_audios = [], [], [], [], []
            packed_mask, packed_pos = [], []
            boundaries = [0]
            image_subseq_ids: list[int] = []
            video_subseq_ids: list[int] = []
            audio_subseq_ids: list[int] = []
            for subseq_idx, length in enumerate(pack):
                idx = by_len[length].pop()
                ids, labels, images, videos, audios = encoded[idx]
                packed_ids += ids
                packed_labels += labels
                packed_pos += list(range(len(ids)))
                packed_mask += ([subseq_idx + 1] if self.data_args.neat_packing else [1]) * len(ids)
                packed_images += images
                packed_videos += videos
                packed_audios += audios
                boundaries.append(boundaries[-1] + len(ids))
                image_subseq_ids += [subseq_idx] * len(images)
                video_subseq_ids += [subseq_idx] * len(videos)
                audio_subseq_ids += [subseq_idx] * len(audios)

            pad_len = self.data_args.cutoff_len - len(packed_ids) + 1
            if pad_len > 0:
                packed_ids += [self.tokenizer.pad_token_id] * pad_len
                packed_labels += [IGNORE_INDEX] * pad_len
                packed_pos += [0] * pad_len
                packed_mask += ([0] if self.data_args.neat_packing else [1]) * pad_len
                boundaries.append(boundaries[-1] + pad_len)
            if len(packed_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to cutoff_len + 1.")

            model_inputs["input_ids"].append(packed_ids)
            model_inputs["attention_mask"].append(packed_mask)
            model_inputs["position_ids"].append(packed_pos)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)
            if self.data_args.neat_packing:
                model_inputs["packing_params"].append(asdict(PackingParams(boundaries, image_subseq_ids or [MAX_SU_SEQ_IDX], video_subseq_ids or [MAX_SU_SEQ_IDX], audio_subseq_ids or [MAX_SU_SEQ_IDX], pad_len)))
        return model_inputs


def _dataset_processor(data_args: Any, template: Any, tokenizer: Any, processor: Any | None) -> SupervisedDatasetProcessor:
    cls = PackedSupervisedDatasetProcessor if data_args.packing else SupervisedDatasetProcessor
    return cls(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _tokenize_dataset(dataset: Any, data_args: Any, training_args: Any, template: Any, tokenizer: Any, processor: Any | None) -> Any:
    proc = _dataset_processor(data_args, template, tokenizer, processor)
    column_names = list(next(iter(dataset)).keys())
    dataset = dataset.map(
        proc.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=(not data_args.overwrite_cache) or (_get_attr(training_args, "local_process_index", 0) != 0),
        desc="Tokenizing SFT dataset",
    )
    if _get_attr(training_args, "should_log", True):
        try:
            print("training example:")
            proc.print_data_example(next(iter(dataset)))
        except StopIteration as exc:
            raise RuntimeError("Cannot find valid samples; check the ShareGPT/LLaVA data format.") from exc
    return dataset


@dataclass
class _OnlineTokenizedIterableDataset(IterableDataset):
    dataset: Any
    dataset_processor: SupervisedDatasetProcessor
    preprocessing_batch_size: int
    seed: int
    shuffle: bool = True

    def __post_init__(self) -> None:
        self.epoch = 0
        if self.preprocessing_batch_size <= 0:
            raise ValueError("`preprocessing_batch_size` should be greater than 0.")

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
        num_shards = max(1, world_size * num_workers)
        shard_id = rank * num_workers + worker_id
        buffer: list[dict[str, Any]] = []
        for position, sample_index in enumerate(indices):
            if position % num_shards != shard_id:
                continue
            buffer.append(self.dataset[int(sample_index)])
            if len(buffer) >= self.preprocessing_batch_size:
                yield from self._yield_processed(buffer)
                buffer = []
        if buffer:
            yield from self._yield_processed(buffer)


class ParquetSFTDataModule(LightningDataModule):
    """Train-only, parquet-only SFT DataModule with LLaMA-Factory multimodal collate."""

    def __init__(self, template: Any, model_args: Any, data_args: Any, training_args: Any, stage: Literal["sft"], tokenizer: Any, processor: Any | None = None, model: Any | None = None, preprocessing_mode: Literal["offline", "online"] = "offline", train_batch_size: int | None = None, shuffle: bool = True) -> None:
        super().__init__()
        if stage != "sft":
            raise ValueError("Only stage='sft' is kept.")
        if preprocessing_mode not in {"offline", "online"}:
            raise ValueError("preprocessing_mode must be 'offline' or 'online'.")
        if data_args.dataset is None:
            raise ValueError("Please set data.dataset to one or more parquet dataset names or paths.")
        if preprocessing_mode == "offline" and data_args.tokenized_path is None:
            raise ValueError("Offline preprocessing requires data.tokenized_path.")
        self.template = template
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.preprocessing_mode = preprocessing_mode
        self.train_batch_size = train_batch_size
        self.shuffle = shuffle
        self.train_dataset = None
        self.data_collator = None

    def _main_first(self, desc: str):
        fn = _get_attr(self.training_args, "main_process_first", None)
        if callable(fn):
            return fn(desc=desc, local=(not bool(_get_attr(self.data_args, "data_shared_file_system", False))))
        return nullcontext()

    def _collect_parquet_files(self, attr: DatasetAttr) -> list[str]:
        if attr.load_from != "file":
            raise ValueError("Only local parquet files are supported.")
        local_path = os.path.join(self.data_args.dataset_dir, attr.dataset_name)
        files: list[str] = []
        if os.path.isdir(local_path):
            for root, _, names in os.walk(local_path):
                for name in sorted(names):
                    if name.endswith(".parquet"):
                        files.append(os.path.join(root, name))
                    elif not (name.startswith(".") or name.startswith("_")):
                        raise ValueError(f"Only parquet files are allowed, found {os.path.join(root, name)}.")
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
        except ValueError as err:
            attrs: list[DatasetAttr] = []
            for name in self.data_args.dataset:
                local_path = os.path.join(self.data_args.dataset_dir, name)
                if os.path.isdir(local_path) or local_path.endswith(".parquet"):
                    attrs.append(DatasetAttr("file", dataset_name=name))
                else:
                    raise err
            return attrs

    def _load_one_dataset(self, attr: DatasetAttr) -> Any:
        logger.info_rank0("Loading parquet dataset %s...", attr)
        dataset = load_dataset(
            path="parquet",
            data_files=self._collect_parquet_files(attr),
            split=attr.split,
            cache_dir=_get_attr(self.model_args, "cache_dir", None),
            token=_get_attr(self.model_args, "hf_hub_token", None),
            num_proc=_get_attr(self.data_args, "preprocessing_num_workers", None),
        )
        if attr.num_samples is not None:
            indexes = np.random.permutation(len(dataset))[: attr.num_samples]
            if len(indexes) < attr.num_samples:
                indexes = np.concatenate([indexes, np.random.choice(len(dataset), attr.num_samples - len(indexes))])
            dataset = dataset.select(indexes)
        if self.data_args.max_samples is not None:
            dataset = dataset.select(range(min(self.data_args.max_samples, len(dataset))))
        return _align_dataset(dataset, attr, self.data_args, self.training_args)

    def _load_train_dataset(self) -> Any:
        return _merge_datasets([self._load_one_dataset(attr) for attr in self._dataset_attrs()], self.data_args)

    def _collator(self) -> SFTDataCollatorWith4DAttentionMask:
        import torch
        config = _get_attr(self.model, "config", None)
        return SFTDataCollatorWith4DAttentionMask(
            template=self.template,
            model=self.model,
            tokenizer=self.tokenizer,
            processor=self.processor,
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id,
            block_diag_attn=bool(_get_attr(self.model_args, "block_diag_attn", self.data_args.neat_packing)),
            neat_packing=bool(self.data_args.neat_packing),
            attn_implementation=_get_attr(config, "_attn_implementation", None),
            compute_dtype=_get_attr(self.model_args, "compute_dtype", None) or torch.float32,
        )

    def _build_and_save_offline_dataset(self) -> None:
        with self._main_first("build tokenized parquet dataset"):
            if has_tokenized_data(self.data_args.tokenized_path):
                logger.info_rank0("Tokenized dataset already exists at %s.", self.data_args.tokenized_path)
                return
            tokenized = _tokenize_dataset(self._load_train_dataset(), self.data_args, self.training_args, self.template, self.tokenizer, self.processor)
            if _get_attr(self.training_args, "should_save", True):
                DatasetDict({"train": tokenized}).save_to_disk(self.data_args.tokenized_path)
                logger.info_rank0("Tokenized dataset is saved at %s.", self.data_args.tokenized_path)

    def prepare_data(self) -> None:
        if self.preprocessing_mode == "offline":
            self._build_and_save_offline_dataset()

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return
        self.data_collator = self._collator()
        if self.preprocessing_mode == "offline":
            if not has_tokenized_data(self.data_args.tokenized_path):
                self._build_and_save_offline_dataset()
            self.train_dataset = _train_split(load_from_disk(self.data_args.tokenized_path))
            logger.info_rank0("Loaded tokenized dataset from %s.", self.data_args.tokenized_path)
        else:
            with self._main_first("load parquet dataset"):
                raw_dataset = self._load_train_dataset()
            proc = _dataset_processor(self.data_args, self.template, self.tokenizer, self.processor)
            self.train_dataset = _OnlineTokenizedIterableDataset(raw_dataset, proc, self.data_args.preprocessing_batch_size, int(_get_attr(self.training_args, "seed", 42)), self.shuffle)
            if _get_attr(self.training_args, "should_log", True):
                try:
                    proc.print_data_example(next(iter(self.train_dataset)))
                except StopIteration as exc:
                    raise RuntimeError("Cannot find valid samples; check the parquet data format.") from exc

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
        if not isinstance(self.train_dataset, IterableDataset):
            kwargs["shuffle"] = self.shuffle
        return DataLoader(**kwargs)
