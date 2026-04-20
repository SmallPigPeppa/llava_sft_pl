# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightning DataModule for the simplified parquet-only PT/SFT data path.

This module intentionally reuses LlamaFactory's dataset converters, templates,
dataset processors and collators.  The only parts that are simplified are:

* load local parquet files only;
* expose train dataloader only;
* support PT and SFT only;
* support two tokenization modes:
  - ``offline``: tokenize the whole train set and save it to ``tokenized_path``;
  - ``online``: tokenize in DataLoader workers every epoch without saving.
"""

import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import DataCollatorForLanguageModeling

from ..extras import logging
from ..extras.constants import IGNORE_INDEX
from ..extras.misc import has_tokenized_data
from .collator import SFTDataCollatorWith4DAttentionMask
from .converter import align_dataset
from .data_utils import get_dataset_module, merge_dataset
from .loader import _get_dataset_processor, _get_preprocessed_dataset
from .parser import DatasetAttr, get_dataset_list

try:  # keep lightning as an optional dependency of the data package
    from lightning.pytorch import LightningDataModule
except Exception:  # pragma: no cover - depends on user's lightning package
    try:
        from pytorch_lightning import LightningDataModule  # type: ignore
    except Exception:  # pragma: no cover

        class LightningDataModule:  # type: ignore[no-redef]
            pass


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


@dataclass
class _OnlineTokenizedIterableDataset(IterableDataset):
    r"""Tokenize raw aligned samples on the fly and yield processed features.

    The tokenizer is applied to chunks of ``preprocessing_batch_size`` examples,
    which keeps PT packing and SFT packing close to the original ``Dataset.map``
    behavior. The final DataLoader still batches the yielded tokenized features
    with the original LlamaFactory collators.
    """

    dataset: "Dataset"
    dataset_processor: "DatasetProcessor"
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
        # For packing, the real number of yielded samples can differ from the raw
        # dataset length. Returning the raw length gives Lightning a stable epoch
        # estimate while the iterator remains the source of truth.
        return len(self.dataset)

    @staticmethod
    def _get_distributed_info() -> tuple[int, int]:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return dist.get_world_size(), dist.get_rank()
        except Exception:
            pass

        return 1, 0

    @staticmethod
    def _batchify(examples: list[dict[str, Any]]) -> dict[str, list[Any]]:
        if len(examples) == 0:
            return {}

        keys = list(examples[0].keys())
        return {key: [example.get(key) for example in examples] for key in keys}

    @staticmethod
    def _iter_model_inputs(model_inputs: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
        if len(model_inputs) == 0:
            return

        first_key = next(iter(model_inputs))
        num_examples = len(model_inputs[first_key])
        for i in range(num_examples):
            yield {key: value[i] for key, value in model_inputs.items()}

    def _preprocess_and_yield(self, examples: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        model_inputs = self.dataset_processor.preprocess_dataset(self._batchify(examples))
        yield from self._iter_model_inputs(model_inputs)

    def __iter__(self) -> Iterable[dict[str, Any]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)

        world_size, rank = self._get_distributed_info()
        worker = get_worker_info()
        if worker is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = worker.num_workers, worker.id

        num_shards = max(1, world_size * num_workers)
        shard_id = rank * num_workers + worker_id

        buffer: list[dict[str, Any]] = []
        for position, sample_index in enumerate(indices):
            if position % num_shards != shard_id:
                continue

            buffer.append(self.dataset[int(sample_index)])
            if len(buffer) >= self.preprocessing_batch_size:
                yield from self._preprocess_and_yield(buffer)
                buffer = []

        if len(buffer) != 0:
            yield from self._preprocess_and_yield(buffer)


class ParquetSFTPTDataModule(LightningDataModule):
    r"""Parquet-only Lightning DataModule that preserves LlamaFactory preprocessing.

    Args:
        template: LlamaFactory template returned by ``get_template_and_fix_tokenizer``.
        model_args: LlamaFactory model arguments. Only cache/token/block-attention
            related fields are read here.
        data_args: LlamaFactory data arguments. ``dataset`` and ``dataset_dir``
            follow the original ``dataset_info.json`` convention, but every
            resolved file must be parquet.
        training_args: LlamaFactory/HF training arguments. Batch-size and dataloader
            fields are reused when present.
        stage: ``"pt"`` or ``"sft"``.
        tokenizer: HF tokenizer.
        processor: Optional multimodal processor.
        model: Optional model passed to the SFT collator for M-RoPE and model-specific
            multimodal handling.
        preprocessing_mode: ``"offline"`` or ``"online"``.
        train_batch_size: Override ``training_args.per_device_train_batch_size``.
        shuffle: Shuffle train data. For offline mode this is DataLoader shuffle;
            for online mode this is index shuffle before chunked tokenization.
    """

    def __init__(
        self,
        template: "Template",
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        stage: Literal["pt", "sft"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"] = None,
        model: Optional[Any] = None,
        preprocessing_mode: Literal["offline", "online"] = "offline",
        train_batch_size: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
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

        self.train_dataset: Optional[Any] = None
        self.data_collator = None

        self._validate_and_normalize_args()

    def _validate_and_normalize_args(self) -> None:
        if self.stage not in ["pt", "sft"]:
            raise ValueError("This Lightning DataModule only supports `stage='pt'` and `stage='sft'`.")

        if self.preprocessing_mode not in ["offline", "online"]:
            raise ValueError("`preprocessing_mode` must be either 'offline' or 'online'.")

        if _get_attr(self.data_args, "eval_dataset", None) is not None:
            raise ValueError("Evaluation datasets are intentionally disabled in this simplified DataModule.")

        if float(_get_attr(self.data_args, "val_size", 0.0) or 0.0) > 1e-6:
            raise ValueError("`val_size` is intentionally disabled in this simplified DataModule.")

        if _get_attr(self.data_args, "streaming", False):
            raise ValueError("Streaming is disabled here: use local parquet map-style datasets.")

        if _get_attr(self.data_args, "dataset", None) is None:
            raise ValueError("Please set `data_args.dataset` to one or more parquet dataset names or paths.")

        # Match LlamaFactory's parser side effect: PT defaults to packing when the
        # user did not explicitly specify a packing value.
        if _get_attr(self.data_args, "packing", None) is None:
            self.data_args.packing = self.stage == "pt"

        if self.preprocessing_mode == "offline" and _get_attr(self.data_args, "tokenized_path", None) is None:
            raise ValueError(
                "Offline preprocessing requires `data_args.tokenized_path` so the tokenized data is saved."
            )

    def _get_main_process_context(self, desc: str):
        main_process_first = _get_attr(self.training_args, "main_process_first", None)
        if callable(main_process_first):
            return main_process_first(
                desc=desc,
                local=(not bool(_get_attr(self.data_args, "data_shared_file_system", False))),
            )

        return nullcontext()

    def _should_log(self) -> bool:
        return bool(_get_attr(self.training_args, "should_log", True))

    def _should_save(self) -> bool:
        return bool(_get_attr(self.training_args, "should_save", True))

    def _collect_parquet_files(self, dataset_attr: DatasetAttr) -> list[str]:
        if dataset_attr.load_from != "file":
            raise ValueError(
                f"Dataset `{dataset_attr}` resolves to `{dataset_attr.load_from}`. "
                "This simplified DataModule only accepts local parquet files."
            )

        local_path = os.path.join(self.data_args.dataset_dir, dataset_attr.dataset_name)
        data_files: list[str] = []
        if os.path.isdir(local_path):
            for root, _, filenames in os.walk(local_path):
                for filename in sorted(filenames):
                    if filename.endswith(".parquet"):
                        data_files.append(os.path.join(root, filename))
                    elif not filename.startswith("."):
                        raise ValueError(
                            f"Only parquet files are allowed, but found `{os.path.join(root, filename)}`."
                        )
        elif os.path.isfile(local_path):
            if not local_path.endswith(".parquet"):
                raise ValueError(f"Only parquet files are allowed, but found `{local_path}`.")
            data_files.append(local_path)
        else:
            raise ValueError(f"Parquet file or directory `{local_path}` not found.")

        if len(data_files) == 0:
            raise ValueError(f"No parquet files found under `{local_path}`.")

        return data_files

    def _get_dataset_attrs(self) -> list[DatasetAttr]:
        try:
            return get_dataset_list(self.data_args.dataset, self.data_args.dataset_dir)
        except ValueError as err:
            # Convenience path for configs that pass parquet file/dir paths directly
            # instead of declaring them in dataset_info.json.
            dataset_attrs: list[DatasetAttr] = []
            for dataset_name in self.data_args.dataset:
                local_path = os.path.join(self.data_args.dataset_dir, dataset_name)
                if os.path.exists(local_path) and (
                    os.path.isdir(local_path) or os.path.splitext(local_path)[-1] == ".parquet"
                ):
                    dataset_attrs.append(DatasetAttr("file", dataset_name=dataset_name))
                else:
                    raise err

            return dataset_attrs

    def _load_single_parquet_dataset(self, dataset_attr: DatasetAttr) -> "Dataset":
        logger.info_rank0(f"Loading parquet dataset {dataset_attr}...")
        data_files = self._collect_parquet_files(dataset_attr)
        dataset = load_dataset(
            path="parquet",
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=_get_attr(self.model_args, "cache_dir", None),
            token=_get_attr(self.model_args, "hf_hub_token", None),
            num_proc=_get_attr(self.data_args, "preprocessing_num_workers", None),
        )

        if dataset_attr.num_samples is not None:
            target_num = dataset_attr.num_samples
            indexes = np.random.permutation(len(dataset))[:target_num]
            target_num -= len(indexes)
            if target_num > 0:
                expand_indexes = np.random.choice(len(dataset), target_num)
                indexes = np.concatenate((indexes, expand_indexes), axis=0)

            if len(indexes) != dataset_attr.num_samples:
                raise ValueError("Sample num mismatched.")

            dataset = dataset.select(indexes)
            logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

        max_samples = _get_attr(self.data_args, "max_samples", None)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        return align_dataset(dataset, dataset_attr, self.data_args, self.training_args)

    def _load_merged_parquet_dataset(self) -> "Dataset":
        datasets = []
        for dataset_attr in self._get_dataset_attrs():
            if dataset_attr.ranking:
                raise ValueError("Ranking datasets are not applicable because only PT and SFT are supported.")
            datasets.append(self._load_single_parquet_dataset(dataset_attr))

        if len(datasets) == 0:
            raise ValueError("No train dataset was loaded.")

        seed = int(_get_attr(self.training_args, "seed", 42))
        return merge_dataset(datasets, self.data_args, seed=seed)

    def _build_data_collator(self):
        if self.stage == "pt":
            return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        compute_dtype = _get_attr(self.model_args, "compute_dtype", None)
        if compute_dtype is None:
            import torch

            compute_dtype = torch.float32

        model_config = _get_attr(self.model, "config", None)
        return SFTDataCollatorWith4DAttentionMask(
            template=self.template,
            model=self.model,
            tokenizer=self.tokenizer,
            processor=self.processor,
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX
            if bool(_get_attr(self.data_args, "ignore_pad_token_for_loss", True))
            else self.tokenizer.pad_token_id,
            block_diag_attn=bool(
                _get_attr(self.model_args, "block_diag_attn", _get_attr(self.data_args, "neat_packing", False))
            ),
            neat_packing=bool(_get_attr(self.data_args, "neat_packing", False)),
            attn_implementation=_get_attr(model_config, "_attn_implementation", None),
            compute_dtype=compute_dtype,
        )

    def _build_and_save_offline_dataset(self) -> None:
        tokenized_path = self.data_args.tokenized_path
        with self._get_main_process_context("build tokenized parquet dataset"):
            if has_tokenized_data(tokenized_path):
                logger.info_rank0(f"Tokenized dataset already exists at {tokenized_path}.")
                return

            raw_dataset = self._load_merged_parquet_dataset()
            train_dataset = _get_preprocessed_dataset(
                raw_dataset,
                self.data_args,
                self.training_args,
                self.stage,
                self.template,
                self.tokenizer,
                self.processor,
                is_eval=False,
            )
            dataset_dict = DatasetDict({"train": train_dataset})
            if self._should_save():
                dataset_dict.save_to_disk(tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {tokenized_path}.")

    def prepare_data(self) -> None:
        if self.preprocessing_mode == "offline":
            self._build_and_save_offline_dataset()

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return

        self.data_collator = self._build_data_collator()
        if self.preprocessing_mode == "offline":
            if not has_tokenized_data(self.data_args.tokenized_path):
                self._build_and_save_offline_dataset()

            tokenized_data = load_from_disk(self.data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            self.train_dataset = dataset_module.get("train_dataset")
            if self.train_dataset is None:
                raise ValueError(f"No train split was found in tokenized dataset `{self.data_args.tokenized_path}`.")

            logger.info_rank0(f"Loaded tokenized dataset from {self.data_args.tokenized_path}.")
        else:
            with self._get_main_process_context("load parquet dataset"):
                raw_dataset = self._load_merged_parquet_dataset()

            dataset_processor = _get_dataset_processor(
                self.data_args,
                self.stage,
                self.template,
                self.tokenizer,
                self.processor,
                do_generate=False,
            )
            self.train_dataset = _OnlineTokenizedIterableDataset(
                dataset=raw_dataset,
                dataset_processor=dataset_processor,
                preprocessing_batch_size=int(_get_attr(self.data_args, "preprocessing_batch_size", 1000)),
                seed=int(_get_attr(self.training_args, "seed", 42)),
                shuffle=self.shuffle,
            )

            if self._should_log():
                iterator = iter(self.train_dataset)
                try:
                    dataset_processor.print_data_example(next(iterator))
                except StopIteration:
                    raise RuntimeError("Cannot find valid samples, check the parquet data format.")

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None or self.data_collator is None:
            self.setup("fit")

        batch_size = self.train_batch_size or int(_get_attr(self.training_args, "per_device_train_batch_size", 1))
        num_workers = int(_get_attr(self.training_args, "dataloader_num_workers", 0) or 0)
        dataloader_kwargs = dict(
            dataset=self.train_dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            num_workers=num_workers,
            pin_memory=bool(_get_attr(self.training_args, "dataloader_pin_memory", False)),
            drop_last=bool(_get_attr(self.training_args, "dataloader_drop_last", False)),
        )

        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = bool(
                _get_attr(self.training_args, "dataloader_persistent_workers", False)
            )
            prefetch_factor = _get_attr(self.training_args, "dataloader_prefetch_factor", None)
            if prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = prefetch_factor

        if not isinstance(self.train_dataset, IterableDataset):
            dataloader_kwargs["shuffle"] = self.shuffle

        return DataLoader(**dataloader_kwargs)

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None


# A shorter alias for downstream training code.
LlamaFactoryLightningDataModule = ParquetSFTPTDataModule
