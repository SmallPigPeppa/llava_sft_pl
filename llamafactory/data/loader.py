from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments
    from ..hparams import DataArguments
    from .processor import DatasetProcessor
    from .template import Template

from .processor import PackedSupervisedDatasetProcessor, SupervisedDatasetProcessor


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["sft"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> "DatasetProcessor":
    if stage != "sft" or do_generate:
        raise ValueError("The simplified data path only supports SFT training.")
    cls = PackedSupervisedDatasetProcessor if data_args.packing else SupervisedDatasetProcessor
    return cls(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Union["Dataset", "IterableDataset"] | None,
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["sft"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Union["Dataset", "IterableDataset"] | None:
    if dataset is None:
        return None
    if stage != "sft" or is_eval:
        raise ValueError("The simplified data path only builds the SFT train split.")

    dataset_processor = _get_dataset_processor(data_args, stage, template, tokenizer, processor, do_generate=False)
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Tokenizing SFT dataset",
        )

    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        try:
            print("training example:")
            dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration as exc:
            raise RuntimeError("Cannot find valid samples; check the ShareGPT/LLaVA data format.") from exc

    return dataset
