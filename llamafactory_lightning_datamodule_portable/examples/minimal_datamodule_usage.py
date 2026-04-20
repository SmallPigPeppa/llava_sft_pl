"""Minimal usage of the portable parquet-only LlamaFactory Lightning DataModule.

This example intentionally does not depend on ``llamafactory.model``. In your
own project, load the tokenizer/processor/model however you already do, then
pass them to the DataModule.
"""

from transformers import AutoTokenizer, Seq2SeqTrainingArguments

from llamafactory.data import LlamaFactoryLightningDataModule, get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments, ModelArguments


model_args = ModelArguments(model_name_or_path="/path/to/model")

data_args = DataArguments(
    dataset="my_sft_parquet",          # dataset_info.json key, or a parquet file/dir path
    dataset_dir="data",               # contains dataset_info.json and/or parquet files
    template="default",
    cutoff_len=2048,
    preprocessing_batch_size=1000,
    tokenized_path="data/tokenized/my_sft_parquet",  # required for offline mode
    val_size=0.0,
)

training_args = Seq2SeqTrainingArguments(
    output_dir="saves/lightning_sft",
    per_device_train_batch_size=1,
    dataloader_num_workers=0,
    do_train=True,
    do_eval=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
template = get_template_and_fix_tokenizer(tokenizer, data_args)

# For text-only PT/SFT, ``processor`` and ``model`` can usually be None.
# For multimodal SFT, pass your HF processor and model so the original
# LlamaFactory SFT collator can handle image/video/audio metadata correctly.
dm = LlamaFactoryLightningDataModule(
    template=template,
    model_args=model_args,
    data_args=data_args,
    training_args=training_args,
    stage="sft",                      # only "sft" or "pt"
    tokenizer=tokenizer,
    processor=None,
    model=None,
    preprocessing_mode="offline",     # "offline" saves tokenized data; "online" tokenizes in workers
)

dm.prepare_data()
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))
print(batch.keys())
