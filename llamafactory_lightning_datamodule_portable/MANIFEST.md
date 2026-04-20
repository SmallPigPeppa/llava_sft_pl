# Portable file manifest

This directory is a minimal source-tree slice for the parquet-only Lightning
DataModule. It preserves the original `llamafactory.*` import paths so the
relative imports in `lightning_datamodule.py` keep working after migration.

## Main file

- `src/llamafactory/data/lightning_datamodule.py`

## Data processing dependencies copied from LlamaFactory

- `src/llamafactory/data/collator.py`
- `src/llamafactory/data/converter.py`
- `src/llamafactory/data/data_utils.py`
- `src/llamafactory/data/formatter.py`
- `src/llamafactory/data/loader.py`
- `src/llamafactory/data/mm_plugin.py`
- `src/llamafactory/data/parser.py`
- `src/llamafactory/data/template.py`
- `src/llamafactory/data/tool_utils.py`
- `src/llamafactory/data/processor/*.py`

## Support files

- `src/llamafactory/extras/*.py`
- `src/llamafactory/hparams/*.py`
- `src/llamafactory/__init__.py`
- `LICENSE`

## Examples and helper files

- `examples/minimal_datamodule_usage.py`
- `examples/dataset_info_parquet_example.json`
- `requirements-datamodule.txt`
