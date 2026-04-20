#!/usr/bin/env python3
"""Train-only Lightning LoRA SFT for image VLMs.

Kept scope: qwen3vl, llava, internvl, kimivl; parquet ShareGPT SFT;
image-only batches; online tokenization.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, Seq2SeqTrainingArguments, get_scheduler, set_seed

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import WandbLogger
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore
    from pytorch_lightning.loggers import WandbLogger  # type: ignore

from llamafactory.data import ParquetSFTDataModule, get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments, ModelArguments

EXCLUDE_LORA_KEYWORDS = (
    "vision_tower",
    "vision_model",
    "visual",
    "multi_modal_projector",
    "mm_projector",
    "projector",
    "lm_head",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("overrides", nargs="*", help="Dot-list overrides, e.g. data.max_samples=64")
    return parser.parse_args()


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for item in overrides:
        key, raw_value = item.split("=", 1)
        cursor = cfg
        for part in key.split(".")[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[key.split(".")[-1]] = parse_value(raw_value)
    return cfg


def keep_dataclass_kwargs(cls: type, source: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls) if getattr(field, "init", True)}
    return {key: value for key, value in source.items() if key in allowed}


def resolve_torch_dtype(name: Any) -> torch.dtype | str:
    if name is None or name == "auto" or isinstance(name, torch.dtype):
        return name or "auto"
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[str(name).lower()]


def apply_runtime_flags(train_cfg: dict[str, Any]) -> None:
    tf32 = bool(train_cfg.get("tf32", True))
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = tf32


def set_use_cache(model: torch.nn.Module, value: bool = False) -> None:
    seen: set[int] = set()

    def visit(obj: Any) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        if hasattr(obj, "use_cache"):
            obj.use_cache = value
        for attr in ("config", "generation_config", "text_config", "language_config"):
            visit(getattr(obj, attr, None))

    visit(model)
    for module in model.modules():
        visit(getattr(module, "config", None))


def load_vision_language_model(model_cfg: dict[str, Any]):
    model_name = model_cfg.get("model_name_or_path") or model_cfg.get("name_or_path")
    if not model_name:
        raise KeyError("model.model_name_or_path is required.")

    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("AutoProcessor must expose processor.tokenizer for SFT.")
    tokenizer.padding_side = str(model_cfg.get("padding_side", "right"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": resolve_torch_dtype(model_cfg.get("torch_dtype", "bfloat16")),
    }
    for key in ("device_map", "attn_implementation"):
        if model_cfg.get(key) is not None:
            model_kwargs[key] = model_cfg[key]
    if model_cfg.get("load_in_4bit"):
        model_kwargs["load_in_4bit"] = True
    if model_cfg.get("load_in_8bit"):
        model_kwargs["load_in_8bit"] = True

    import transformers

    errors: list[str] = []
    for auto_cls in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        factory = getattr(transformers, auto_cls, None)
        if factory is None:
            continue
        try:
            model = factory.from_pretrained(model_name, **model_kwargs)
            break
        except Exception as exc:  # try the next compatible Auto class
            errors.append(f"{auto_cls}: {exc}")
    else:
        raise RuntimeError("Could not load VLM. Tried Auto classes:\n" + "\n".join(errors))

    if bool(model_cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
    if not bool(model_cfg.get("use_cache", False)):
        set_use_cache(model, False)
    return model, processor, tokenizer


def freeze_by_keywords(model: torch.nn.Module, keywords: tuple[str, ...]) -> None:
    for name, param in model.named_parameters():
        if any(key in name for key in keywords):
            param.requires_grad = False


def find_linear_lora_targets(model: torch.nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and not any(k in name for k in EXCLUDE_LORA_KEYWORDS)]


def apply_lora(model: torch.nn.Module, lora_cfg: dict[str, Any]) -> torch.nn.Module:
    if not bool(lora_cfg.get("enable", True)):
        return model
    if lora_cfg.get("freeze_vision_tower", True):
        freeze_by_keywords(model, ("vision_tower", "vision_model", "visual"))
    if lora_cfg.get("freeze_multi_modal_projector", True):
        freeze_by_keywords(model, ("multi_modal_projector", "mm_projector", "projector"))
    if lora_cfg.get("prepare_kbit", False):
        model = prepare_model_for_kbit_training(model)

    target_modules = lora_cfg.get("target_modules", "all")
    if target_modules in {"all", "all-linear", "all_linear"}:
        target_modules = find_linear_lora_targets(model)
    elif isinstance(target_modules, str):
        target_modules = [x.strip() for x in target_modules.split(",") if x.strip()]

    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("alpha", 16)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias=str(lora_cfg.get("bias", "none")),
            task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
            target_modules=target_modules,
        ),
    )
    model.print_trainable_parameters()
    return model


def patch_processor_for_llava(processor: Any, model: torch.nn.Module, template_name: str) -> None:
    if processor is None or not template_name.startswith("llava"):
        return
    model_config = getattr(model, "config", None)
    vision_config = getattr(model_config, "vision_config", None)
    processor.patch_size = getattr(processor, "patch_size", None) or getattr(vision_config, "patch_size", 14)
    processor.num_additional_image_tokens = getattr(processor, "num_additional_image_tokens", None) or 1
    processor.vision_feature_select_strategy = getattr(
        processor,
        "vision_feature_select_strategy",
        getattr(model_config, "vision_feature_select_strategy", "default"),
    )


def training_args_kwargs(train_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    candidates = {
        "output_dir": str(train_cfg.get("output_dir", "saves/lightning_sft")),
        "overwrite_output_dir": bool(train_cfg.get("overwrite_output_dir", False)),
        "do_train": True,
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "dataloader_pin_memory": bool(train_cfg.get("dataloader_pin_memory", torch.cuda.is_available())),
        "dataloader_drop_last": bool(train_cfg.get("dataloader_drop_last", False)),
        "remove_unused_columns": False,
        "seed": seed,
        "logging_steps": int(train_cfg.get("logging_steps", 50)),
        "report_to": train_cfg.get("report_to", []),
        "save_strategy": "no",
    }
    for key in ("bf16", "fp16", "dataloader_persistent_workers", "dataloader_prefetch_factor", "disable_tqdm"):
        if train_cfg.get(key) is not None:
            candidates[key] = train_cfg[key]
    allowed = {field.name for field in fields(Seq2SeqTrainingArguments) if getattr(field, "init", True)}
    return {key: value for key, value in candidates.items() if key in allowed}


def build_datamodule(cfg: dict[str, Any], tokenizer: Any, processor: Any, model: torch.nn.Module):
    data_cfg = dict(cfg.get("data") or {})
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("train") or {})
    seed = int(cfg.get("seed", data_cfg.get("seed", 42)))

    data_cfg.setdefault("template", "llava")
    data_args = DataArguments(**keep_dataclass_kwargs(DataArguments, data_cfg))

    model_kwargs = keep_dataclass_kwargs(ModelArguments, model_cfg)
    if "model_name_or_path" not in model_kwargs and model_cfg.get("name_or_path"):
        model_kwargs["model_name_or_path"] = model_cfg["name_or_path"]
    model_args = ModelArguments(**model_kwargs)
    compute_dtype = resolve_torch_dtype(model_cfg.get("torch_dtype"))
    if isinstance(compute_dtype, torch.dtype):
        model_args.compute_dtype = compute_dtype
    elif train_cfg.get("bf16"):
        model_args.compute_dtype = torch.bfloat16
    elif train_cfg.get("fp16"):
        model_args.compute_dtype = torch.float16
    model_args.model_max_length = int(data_cfg.get("cutoff_len", 2048))

    hf_training_args = Seq2SeqTrainingArguments(**training_args_kwargs(train_cfg, seed=seed))
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    return ParquetSFTDataModule(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=hf_training_args,
        tokenizer=tokenizer,
        processor=processor,
        model=model,
        train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        shuffle=bool(train_cfg.get("shuffle", True)),
    )


def build_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or any(mark in name.lower() for mark in ("bias", "norm.weight", "layernorm.weight")):
            no_decay.append(param)
        else:
            decay.append(param)
    kwargs = {
        "lr": float(train_cfg.get("learning_rate", 5e-5)),
        "betas": (float(train_cfg.get("adam_beta1", 0.9)), float(train_cfg.get("adam_beta2", 0.999))),
        "eps": float(train_cfg.get("adam_epsilon", train_cfg.get("adam_eps", 1e-8))),
    }
    if str(train_cfg.get("optim", "adamw_torch")).lower() == "adamw_torch_fused" and torch.cuda.is_available():
        kwargs["fused"] = True
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(train_cfg.get("weight_decay", 0.0))}, {"params": no_decay, "weight_decay": 0.0}],
        **kwargs,
    )


class VLSFTModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, train_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.train_cfg = dict(train_cfg)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.model(**batch).loss
        self.log("train/loss", loss, prog_bar=True, logger=True, batch_size=batch["input_ids"].shape[0], sync_dist=True)
        return loss

    def on_train_epoch_start(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        if hasattr(datamodule, "set_epoch"):
            datamodule.set_epoch(int(self.current_epoch))

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model, self.train_cfg)
        scheduler_name = str(self.train_cfg.get("lr_scheduler_type", "linear")).lower()
        if scheduler_name in {"none", "no", "null"}:
            return optimizer
        estimated_steps = self.trainer.estimated_stepping_batches
        if isinstance(estimated_steps, float) and math.isinf(estimated_steps):
            if self.train_cfg.get("max_steps") is None:
                return optimizer
            estimated_steps = int(self.train_cfg["max_steps"])
        total_steps = max(1, int(estimated_steps))
        warmup = self.train_cfg.get("warmup_steps")
        warmup_steps = int(warmup) if warmup is not None else int(float(self.train_cfg.get("warmup_ratio", 0.0)) * total_steps)
        scheduler = get_scheduler(scheduler_name, optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}


def wants_wandb(report_to: Any) -> bool:
    values = report_to if isinstance(report_to, (list, tuple, set)) else [report_to]
    return "wandb" in {str(x).lower() for x in values if x is not None}


def build_logger(cfg: dict[str, Any], train_cfg: dict[str, Any]):
    if not wants_wandb(train_cfg.get("report_to", ["wandb"])):
        return False
    project = cfg.get("wandb_project") or train_cfg.get("wandb_project") or os.getenv("WANDB_PROJECT")
    if project:
        os.environ.setdefault("WANDB_PROJECT", str(project))
    return WandbLogger(project=str(project) if project else None, name=train_cfg.get("run_name"), save_dir=str(train_cfg.get("output_dir", ".")))


def build_precision(train_cfg: dict[str, Any]) -> str:
    if train_cfg.get("bf16"):
        return "bf16-mixed"
    if train_cfg.get("fp16"):
        return "16-mixed"
    return str(train_cfg.get("precision", "32-true"))


def build_trainer(cfg: dict[str, Any], train_cfg: dict[str, Any]) -> pl.Trainer:
    max_steps = train_cfg.get("max_steps")
    kwargs: dict[str, Any] = {
        "default_root_dir": str(train_cfg.get("output_dir", "saves/lightning_sft")),
        "accelerator": train_cfg.get("accelerator", "auto"),
        "devices": train_cfg.get("devices", "auto"),
        "num_nodes": int(train_cfg.get("num_nodes", 1)),
        "precision": build_precision(train_cfg),
        "max_epochs": int(math.ceil(float(train_cfg.get("num_train_epochs", 1)))),
        "max_steps": int(max_steps) if max_steps is not None else -1,
        "accumulate_grad_batches": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "logger": build_logger(cfg, train_cfg),
        "enable_checkpointing": False,
        "log_every_n_steps": int(train_cfg.get("logging_steps", 50)),
        "strategy": train_cfg.get("strategy") or "auto",
    }
    if train_cfg.get("gradient_clip_val") is not None:
        kwargs["gradient_clip_val"] = float(train_cfg["gradient_clip_val"])
    if train_cfg.get("enable_progress_bar") is not None:
        kwargs["enable_progress_bar"] = bool(train_cfg["enable_progress_bar"])
    return pl.Trainer(**kwargs)


def save_final_model(trainer: pl.Trainer, module: VLSFTModule, processor: Any, output_dir: str) -> None:
    if not trainer.is_global_zero:
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    module.model.save_pretrained(output_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir)


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    seed = int(cfg.get("seed", cfg.get("data", {}).get("seed", 42)))
    set_seed(seed)
    pl.seed_everything(seed, workers=True)
    apply_runtime_flags(cfg.get("train", {}))

    model, processor, tokenizer = load_vision_language_model(cfg["model"])
    model = apply_lora(model, cfg.get("lora", {}))
    patch_processor_for_llava(processor, model, str(cfg.get("data", {}).get("template", "llava")))

    datamodule = build_datamodule(cfg, tokenizer=tokenizer, processor=processor, model=model)
    module = VLSFTModule(model=model, train_cfg=cfg["train"])
    trainer = build_trainer(cfg, cfg["train"])
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg["train"].get("resume_from_checkpoint") or None)

    if cfg["train"].get("save_model_at_end", False):
        save_final_model(trainer, module, processor, str(cfg["train"]["output_dir"]))
    elif trainer.is_global_zero:
        print("Training finished. Final model saving is disabled by train.save_model_at_end=false.")


if __name__ == "__main__":
    main()
