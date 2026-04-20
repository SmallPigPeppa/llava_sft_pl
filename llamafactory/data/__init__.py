from .template import TEMPLATES, Template, get_template_and_fix_tokenizer


def __getattr__(name: str):
    if name == "ParquetSFTDataModule":
        from .lightning_datamodule import ParquetSFTDataModule

        return ParquetSFTDataModule
    raise AttributeError(name)


__all__ = ["ParquetSFTDataModule", "TEMPLATES", "Template", "get_template_and_fix_tokenizer"]
