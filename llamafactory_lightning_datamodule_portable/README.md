# LlamaFactory parquet-only Lightning DataModule portable slice

This folder contains a portable subset of the modified LlamaFactory code for
moving `src/llamafactory/data/lightning_datamodule.py` into another project.

The package keeps the original `llamafactory` source layout. The easiest
migration path is to copy this folder's `src/llamafactory` directory into your
project, or add this folder's `src` directory to `PYTHONPATH`.

```bash
pip install -r requirements-datamodule.txt
export PYTHONPATH=/path/to/llamafactory_lightning_datamodule_portable/src:$PYTHONPATH
```

Then import:

```python
from llamafactory.data import LlamaFactoryLightningDataModule
```

## What is included

The DataModule supports only:

- train dataloader only;
- local parquet files only;
- `stage="pt"` and `stage="sft"` only;
- `preprocessing_mode="offline"`: tokenize the whole training set and save to
  `data_args.tokenized_path`;
- `preprocessing_mode="online"`: tokenize in DataLoader workers without saving.

It reuses the original LlamaFactory dataset converters, template logic,
pretrain/supervised processors, multimodal plugin, and collators.

## Parquet image bytes

The copied `mm_plugin.py` already supports image values as `bytes`, PIL images,
paths, or HuggingFace-style dictionaries such as:

```python
{"bytes": image_bytes, "path": None}
```

So a parquet column like `images: list[bytes]` or `images: list[{"bytes": ..., "path": ...}]`
can be used through the usual `dataset_info.json` column mapping.

## Important migration notes

This slice does not include `llamafactory.model` or the HF Trainer training
loop. In the target Lightning project, load the tokenizer, processor and model
with your own code, then pass them into `LlamaFactoryLightningDataModule`.

If your target project already has a `llamafactory` package installed, put this
portable `src` earlier on `PYTHONPATH`, or rename the package and update the
relative imports consistently.

The file `requirements-datamodule.txt` is intentionally conservative. If your
project already provides PyTorch, Transformers, Datasets, PEFT, Lightning, etc.,
you can rely on the existing environment as long as versions remain compatible
with the copied LlamaFactory code.
