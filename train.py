#!/usr/bin/env python3
"""Minimal Lightning Trainer script for LLaVA LoRA SFT.

Example:
    WANDB_PROJECT=CL-debug python train.py --config configs/demo2k.yaml
    python train.py --config configs/demo2k.yaml train.num_train_epochs=1 data.max_samples=64
"""

from __future__ import annotations

import argparse
import math
from datetime import timedelta
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_scheduler, set_seed

try:
    from lightning.pytorch import LightningDataModule, LightningModule, Trainer as LightningTrainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy
except ImportError:  # pragma: no cover - compatibility for environments that install pytorch-lightning only
    from pytorch_lightning import LightningDataModule, LightningModule, Trainer as LightningTrainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from data import IGNORE_INDEX, ShareGPTLlavaDataset, load_raw_dataset, read_dataset_spec, split_train_eval


EXCLUDE_LORA_KEYWORDS = (
    "vision_tower",
    "vision_model",
    "visual",
    "multi_modal_projector",
    "mm_projector",
    "projector",
    "lm_head",
)


HF_TRAINING_ARG_KEYS_TO_IGNORE = {
    # Kept in YAML for HF-Trainer compatibility. Lightning handles these concepts directly below.
    "remove_unused_columns",
    "save_checkpoint",
    "save_strategy",
    "save_steps",
    "save_total_limit",
    "save_model_at_end",
    "report_to",
    "run_name",
    "output_dir",
    "resume_from_checkpoint",
    "logging_steps",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "num_train_epochs",
    "max_steps",
    "lr_scheduler_type",
    "warmup_ratio",
    "warmup_steps",
    "bf16",
    "fp16",
    "tf32",
    "ddp_timeout",
    "ddp_find_unused_parameters",
    "eval_strategy",
    "evaluation_strategy",
    "eval_steps",
    "dataloader_num_workers",
    "dataloader_drop_last",
    "dataloader_pin_memory",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "max_grad_norm",
    "optim",
    "deepspeed",
    "pad_to_multiple_of",
    "use_cache",
}


NO_DECAY_KEYWORDS = (
    "bias",
    "layernorm",
    "layer_norm",
    "rmsnorm",
    "rms_norm",
    "ln_",
    ".ln",
    "norm.weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional dot-list overrides, e.g. train.learning_rate=1e-4 data.max_samples=128",
    )
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
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
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        cursor = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = parse_value(raw_value)
    return cfg


def resolve_torch_dtype(name: Any):
    if name is None or name == "auto":
        return "auto"
    if isinstance(name, torch.dtype):
        return name
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping[str(name).lower()]


def apply_runtime_flags(train_cfg: dict[str, Any]) -> None:
    tf32 = bool(train_cfg.get("tf32", True))
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = tf32


def set_use_cache(model: torch.nn.Module, value: bool = False) -> None:
    """Set KV-cache flag on wrapper, inner language model, configs and generation_config."""
    seen: set[int] = set()

    def _set(obj: Any) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))

        if hasattr(obj, "use_cache"):
            obj.use_cache = value

        for attr in ("config", "generation_config", "text_config", "language_config"):
            _set(getattr(obj, attr, None))

    _set(model)

    # Cover nested modules such as model.language_model / base_model / PEFT wrapper.
    for module in model.modules():
        _set(getattr(module, "config", None))
        _set(getattr(module, "generation_config", None))


def load_vision_language_model(model_cfg: dict[str, Any]):
    """Load processor + LLaVA model while keeping imports compatible across Transformers versions."""
    model_name = model_cfg["model_name_or_path"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("AutoProcessor has no tokenizer; this script expects a LLaVA-style processor.")
    tokenizer.padding_side = str(model_cfg.get("padding_side", "right"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": resolve_torch_dtype(model_cfg.get("torch_dtype", "bfloat16")),
    }
    if model_cfg.get("device_map") is not None:
        model_kwargs["device_map"] = model_cfg["device_map"]
    if model_cfg.get("attn_implementation") is not None:
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    if model_cfg.get("load_in_4bit"):
        model_kwargs["load_in_4bit"] = True
    if model_cfg.get("load_in_8bit"):
        model_kwargs["load_in_8bit"] = True

    try:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs)
    except Exception:
        from transformers import LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(model_name, **model_kwargs)

    # Training defaults to no KV cache; gradient checkpointing requires cache to be disabled.
    if not bool(model_cfg.get("use_cache", False)):
        set_use_cache(model, False)

    if bool(model_cfg.get("gradient_checkpointing", False)):
        set_use_cache(model, False)  # must be before first forward
        model.gradient_checkpointing_enable()
        set_use_cache(model, False)  # cover nested configs again

    return model, processor, tokenizer


def freeze_by_keywords(model: torch.nn.Module, keywords: tuple[str, ...]) -> None:
    for name, param in model.named_parameters():
        if any(key in name for key in keywords):
            param.requires_grad = False


def find_linear_lora_targets(model: torch.nn.Module, exclude_keywords: tuple[str, ...] = EXCLUDE_LORA_KEYWORDS) -> list[str]:
    """Return full Linear module names outside vision tower/projector/lm_head.

    Full names are used instead of suffixes so PEFT does not accidentally inject LoRA into
    vision modules with the same names, such as q_proj/v_proj.
    """
    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(key in name for key in exclude_keywords):
            continue
        targets.append(name)
    if not targets:
        raise ValueError("No Linear modules found for LoRA. Set lora.target_modules explicitly in YAML.")
    return targets


def apply_lora(model: torch.nn.Module, lora_cfg: dict[str, Any]) -> torch.nn.Module:
    if not bool(lora_cfg.get("enable", True)):
        return model

    target_modules = lora_cfg.get("target_modules", "all")
    if target_modules in {"all", "all-linear", "all_linear"}:
        target_modules = find_linear_lora_targets(model)
    elif isinstance(target_modules, str):
        target_modules = [x.strip() for x in target_modules.split(",") if x.strip()]

    if lora_cfg.get("freeze_vision_tower", True):
        freeze_by_keywords(model, ("vision_tower", "vision_model", "visual"))
    if lora_cfg.get("freeze_multi_modal_projector", True):
        freeze_by_keywords(model, ("multi_modal_projector", "mm_projector", "projector"))

    if lora_cfg.get("prepare_kbit", False):
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def infer_image_seq_len(processor, model, data_cfg: dict[str, Any]) -> int:
    if data_cfg.get("image_seq_len") is not None:
        return int(data_cfg["image_seq_len"])

    image_processor = getattr(processor, "image_processor", None)
    crop_size = getattr(image_processor, "crop_size", None) or {}
    size = getattr(image_processor, "size", None) or {}

    height = crop_size.get("height") or size.get("height") or size.get("shortest_edge") or 336
    width = crop_size.get("width") or size.get("width") or size.get("shortest_edge") or 336

    patch_size = getattr(processor, "patch_size", None)
    if patch_size is None:
        vision_cfg = getattr(getattr(model, "config", None), "vision_config", None)
        patch_size = getattr(vision_cfg, "patch_size", 14)

    additional = getattr(processor, "num_additional_image_tokens", None)
    if additional is None:
        additional = 1

    strategy = getattr(processor, "vision_feature_select_strategy", None)
    if strategy is None:
        strategy = getattr(getattr(model, "config", None), "vision_feature_select_strategy", "default")

    seq_len = (int(height) // int(patch_size)) * (int(width) // int(patch_size)) + int(additional)
    if strategy == "default":
        seq_len -= 1
    return int(seq_len)


class LlavaDataCollator:
    """Pad text labels and build image tensors through the HF image processor."""

    def __init__(self, tokenizer, processor, pad_to_multiple_of: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.pad_to_multiple_of = pad_to_multiple_of

    def _pad_labels(self, labels: list[list[int]], max_len: int) -> torch.Tensor:
        padded = []
        for row in labels:
            pad_len = max_len - len(row)
            if self.tokenizer.padding_side == "right":
                padded.append(row + [IGNORE_INDEX] * pad_len)
            else:
                padded.append([IGNORE_INDEX] * pad_len + row)
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = []
        text_features = []
        label_rows = []
        for feature in features:
            images.extend(feature.get("images") or [])
            text_features.append(
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
            )
            label_rows.append(feature["labels"])

        batch = self.tokenizer.pad(
            text_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = self._pad_labels(label_rows, max_len=batch["input_ids"].shape[1])

        if images:
            image_inputs = self.processor.image_processor(images, return_tensors="pt")
            batch.update(image_inputs)
        return batch


class LlavaLightningDataModule(LightningDataModule):
    """Lightning DataModule that mirrors the Hugging Face Trainer dataloader settings."""

    def __init__(
        self,
        train_dataset: ShareGPTLlavaDataset,
        eval_dataset: ShareGPTLlavaDataset | None,
        data_collator: LlavaDataCollator,
        train_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.train_cfg = train_cfg

    def _common_loader_kwargs(self) -> dict[str, Any]:
        num_workers = int(self.train_cfg.get("dataloader_num_workers", 0))
        kwargs: dict[str, Any] = {
            "num_workers": num_workers,
            "pin_memory": bool(self.train_cfg.get("dataloader_pin_memory", True)),
            "collate_fn": self.data_collator,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(self.train_cfg.get("dataloader_persistent_workers", False))
            if self.train_cfg.get("dataloader_prefetch_factor") is not None:
                kwargs["prefetch_factor"] = int(self.train_cfg["dataloader_prefetch_factor"])
        return kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.train_cfg.get("per_device_train_batch_size", 8)),
            shuffle=True,
            drop_last=bool(self.train_cfg.get("dataloader_drop_last", False)),
            **self._common_loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.eval_dataset is None:
            return None
        return DataLoader(
            self.eval_dataset,
            batch_size=int(self.train_cfg.get("per_device_eval_batch_size", 8)),
            shuffle=False,
            drop_last=False,
            **self._common_loader_kwargs(),
        )


def _is_no_decay_parameter(name: str, param: torch.nn.Parameter) -> bool:
    lowered = name.lower()
    return param.ndim < 2 or any(keyword in lowered for keyword in NO_DECAY_KEYWORDS)


def build_optimizer_grouped_parameters(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_no_decay_parameter(name, param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


class LlavaLightningModule(LightningModule):
    """LightningModule that keeps the HF Trainer optimization/scheduler semantics."""

    def __init__(self, model: torch.nn.Module, train_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.train_cfg = train_cfg
        self.save_hyperparameters({"train": dict(train_cfg)})

    def forward(self, **batch: torch.Tensor):
        return self.model(**batch)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        batch_size = int(batch["input_ids"].shape[0]) if "input_ids" in batch else None
        self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        batch_size = int(batch["input_ids"].shape[0]) if "input_ids" in batch else None
        self.log("eval_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if optimizer.param_groups:
            self.log(
                "learning_rate",
                optimizer.param_groups[0]["lr"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

    def configure_optimizers(self):
        learning_rate = float(self.train_cfg.get("learning_rate", 5e-5))
        weight_decay = float(self.train_cfg.get("weight_decay", 0.0))
        betas = (
            float(self.train_cfg.get("adam_beta1", 0.9)),
            float(self.train_cfg.get("adam_beta2", 0.999)),
        )
        eps = float(self.train_cfg.get("adam_epsilon", 1e-8))
        optim_name = str(self.train_cfg.get("optim", "adamw_torch")).lower()

        if optim_name in {"adamw", "adamw_torch", "adamw_hf"}:
            optimizer = torch.optim.AdamW(
                build_optimizer_grouped_parameters(self.model, weight_decay=weight_decay),
                lr=learning_rate,
                betas=betas,
                eps=eps,
            )
        else:
            raise ValueError(
                f"Unsupported train.optim={optim_name!r} in Lightning port. "
                "Use adamw_torch/adamw, or add the optimizer mapping here."
            )

        lr_scheduler_type = str(self.train_cfg.get("lr_scheduler_type", "linear"))
        estimated_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
        explicit_max_steps = self.train_cfg.get("max_steps")
        if explicit_max_steps is not None and int(explicit_max_steps) > 0:
            num_training_steps = int(explicit_max_steps)
        else:
            num_training_steps = max(estimated_steps, 1)

        warmup_steps = self.train_cfg.get("warmup_steps")
        if warmup_steps is not None and int(warmup_steps) > 0:
            num_warmup_steps = int(warmup_steps)
        else:
            num_warmup_steps = int(math.ceil(num_training_steps * float(self.train_cfg.get("warmup_ratio", 0.0))))

        scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def normalize_report_to(report_to: Any) -> list[str]:
    if report_to is None:
        return []
    if isinstance(report_to, str):
        lowered = report_to.lower()
        if lowered in {"none", "null", "false", "off", ""}:
            return []
        return [item.strip().lower() for item in report_to.split(",") if item.strip()]
    if isinstance(report_to, (list, tuple, set)):
        values = [str(item).lower() for item in report_to]
        return [] if any(item in {"none", "null", "false", "off"} for item in values) else values
    return [str(report_to).lower()]


def build_logger(train_cfg: dict[str, Any], wandb_project: str | None) -> Any:
    report_to = normalize_report_to(train_cfg.get("report_to", ["wandb"]))
    if not report_to:
        return False

    output_dir = str(train_cfg.get("output_dir", "outputs"))
    run_name = train_cfg.get("run_name")

    if "wandb" in report_to:
        return WandbLogger(
            project=wandb_project or os.environ.get("WANDB_PROJECT"),
            name=str(run_name) if run_name else None,
            save_dir=output_dir,
            log_model=False,
        )

    # Lightning's default logger covers unsupported HF report targets without changing training.
    return True


def resolve_precision(train_cfg: dict[str, Any]) -> str:
    if bool(train_cfg.get("bf16", False)):
        return "bf16-mixed"
    if bool(train_cfg.get("fp16", False)):
        return "16-mixed"
    return "32-true"


def train_batches_per_epoch(train_dataset: ShareGPTLlavaDataset, train_cfg: dict[str, Any]) -> int:
    batch_size = max(int(train_cfg.get("per_device_train_batch_size", 8)), 1)
    dataset_size = len(train_dataset)
    if bool(train_cfg.get("dataloader_drop_last", False)):
        return max(dataset_size // batch_size, 1)
    return max(math.ceil(dataset_size / batch_size), 1)


def resolve_max_epochs_and_steps(train_dataset: ShareGPTLlavaDataset, train_cfg: dict[str, Any]) -> tuple[int, int]:
    explicit_max_steps = train_cfg.get("max_steps")
    if explicit_max_steps is not None and int(explicit_max_steps) > 0:
        return -1, int(explicit_max_steps)

    num_train_epochs = float(train_cfg.get("num_train_epochs", 3.0))
    grad_accum = max(int(train_cfg.get("gradient_accumulation_steps", 1)), 1)

    if not num_train_epochs.is_integer():
        update_steps_per_epoch = max(math.ceil(train_batches_per_epoch(train_dataset, train_cfg) / grad_accum), 1)
        return int(math.ceil(num_train_epochs)), int(math.ceil(num_train_epochs * update_steps_per_epoch))

    return max(int(num_train_epochs), 1), -1


def build_callbacks(train_cfg: dict[str, Any], logger_enabled: bool) -> list[Any]:
    callbacks: list[Any] = []

    save_checkpoint = bool(train_cfg.get("save_checkpoint", True))
    save_strategy = str(train_cfg.get("save_strategy", "steps")).lower()
    if save_checkpoint and save_strategy != "no":
        output_dir = Path(str(train_cfg.get("output_dir", "outputs")))
        checkpoint_dir = output_dir / "checkpoints"
        common_kwargs: dict[str, Any] = {
            "dirpath": str(checkpoint_dir),
            "filename": "step-{step}",
            "save_last": True,
            "save_top_k": -1,
            "auto_insert_metric_name": False,
        }
        if save_strategy == "steps":
            common_kwargs["every_n_train_steps"] = int(train_cfg.get("save_steps", 500))
        elif save_strategy == "epoch":
            common_kwargs["every_n_epochs"] = 1
        callbacks.append(ModelCheckpoint(**common_kwargs))

    return callbacks


def resolve_validation_kwargs(train_cfg: dict[str, Any], has_eval_dataset: bool) -> dict[str, Any]:
    if not has_eval_dataset:
        return {"limit_val_batches": 0}

    eval_strategy = str(train_cfg.get("eval_strategy", train_cfg.get("evaluation_strategy", "no"))).lower()
    if eval_strategy in {"no", "none", "false", "off"}:
        return {"limit_val_batches": 0}

    if eval_strategy == "steps":
        # HF Trainer's eval_steps counts optimizer steps; Lightning's integer val_check_interval
        # counts train batches, so multiply by grad accumulation to preserve the same cadence.
        interval_batches = int(train_cfg.get("eval_steps", 500)) * max(int(train_cfg.get("gradient_accumulation_steps", 1)), 1)
        return {
            "val_check_interval": max(interval_batches, 1),
            "check_val_every_n_epoch": None,
        }

    if eval_strategy == "epoch":
        return {"check_val_every_n_epoch": 1}

    raise ValueError(f"Unsupported eval_strategy/evaluation_strategy={eval_strategy!r}")


def devices_request_multiple(train_cfg: dict[str, Any]) -> bool:
    devices = train_cfg.get("devices")
    if isinstance(devices, int):
        return devices > 1
    if isinstance(devices, str):
        lowered = devices.lower()
        if lowered in {"auto", "-1"}:
            return False
        try:
            return int(lowered) > 1
        except ValueError:
            return "," in lowered
    if isinstance(devices, (list, tuple, set)):
        return len(devices) > 1
    return False


def build_ddp_strategy(train_cfg: dict[str, Any]) -> DDPStrategy:
    kwargs: dict[str, Any] = {}
    if train_cfg.get("ddp_find_unused_parameters") is not None:
        kwargs["find_unused_parameters"] = bool(train_cfg["ddp_find_unused_parameters"])
    if train_cfg.get("ddp_timeout") is not None:
        kwargs["timeout"] = timedelta(seconds=int(train_cfg["ddp_timeout"]))
    return DDPStrategy(**kwargs)


def resolve_strategy(train_cfg: dict[str, Any]) -> Any:
    if train_cfg.get("deepspeed") is not None:
        return DeepSpeedStrategy(config=train_cfg["deepspeed"])

    strategy = train_cfg.get("strategy")
    if strategy is not None:
        if str(strategy).lower() == "ddp":
            return build_ddp_strategy(train_cfg)
        return strategy

    # Keep single-device behavior unchanged. If the user explicitly asks Lightning for
    # multiple devices, carry over HF-style DDP knobs.
    if devices_request_multiple(train_cfg) and (
        train_cfg.get("ddp_find_unused_parameters") is not None or train_cfg.get("ddp_timeout") is not None
    ):
        return build_ddp_strategy(train_cfg)

    return "auto"


def warn_unused_train_keys(train_cfg: dict[str, Any]) -> None:
    extra_keys = sorted(set(train_cfg) - HF_TRAINING_ARG_KEYS_TO_IGNORE - {"accelerator", "devices", "strategy"})
    if extra_keys:
        print(f"Warning: train keys are present but not explicitly mapped in the Lightning port: {extra_keys}")


def build_lightning_trainer(
    train_dataset: ShareGPTLlavaDataset,
    eval_dataset: ShareGPTLlavaDataset | None,
    train_cfg: dict[str, Any],
    wandb_project: str | None,
) -> LightningTrainer:
    warn_unused_train_keys(train_cfg)

    max_epochs, max_steps = resolve_max_epochs_and_steps(train_dataset, train_cfg)
    logger = build_logger(train_cfg, wandb_project=wandb_project)
    checkpointing_enabled = bool(train_cfg.get("save_checkpoint", True)) and str(train_cfg.get("save_strategy", "steps")).lower() != "no"

    trainer_kwargs: dict[str, Any] = {
        "default_root_dir": str(train_cfg.get("output_dir", "outputs")),
        "accelerator": train_cfg.get("accelerator", "auto"),
        "devices": train_cfg.get("devices", "auto"),
        "precision": resolve_precision(train_cfg),
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "accumulate_grad_batches": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "gradient_clip_val": float(train_cfg.get("max_grad_norm", 1.0)),
        "log_every_n_steps": int(train_cfg.get("logging_steps", 500)),
        "logger": logger,
        "callbacks": build_callbacks(train_cfg, logger_enabled=bool(logger)),
        "enable_checkpointing": checkpointing_enabled,
        "num_sanity_val_steps": 0,
        "strategy": resolve_strategy(train_cfg),
    }
    trainer_kwargs.update(resolve_validation_kwargs(train_cfg, has_eval_dataset=eval_dataset is not None))
    return LightningTrainer(**trainer_kwargs)


def save_final_model(model: torch.nn.Module, processor: Any, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir)


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.overrides)

    seed = int(cfg.get("seed", cfg.get("data", {}).get("seed", 42)))
    set_seed(seed)
    seed_everything(seed, workers=True)
    apply_runtime_flags(cfg.get("train", {}))

    wandb_project = cfg.get("wandb_project") or cfg.get("train", {}).get("wandb_project")
    if wandb_project:
        os.environ.setdefault("WANDB_PROJECT", str(wandb_project))

    model, processor, tokenizer = load_vision_language_model(cfg["model"])
    model = apply_lora(model, cfg.get("lora", {}))

    if not bool(cfg["model"].get("use_cache", False)):
        set_use_cache(model, False)

    data_cfg = cfg["data"]
    spec = read_dataset_spec(data_cfg)
    raw_ds = load_raw_dataset(data_cfg)
    train_raw, eval_raw = split_train_eval(raw_ds, data_cfg)

    image_seq_len = infer_image_seq_len(processor, model, data_cfg)
    print(f"Using image_seq_len={image_seq_len}")
    print(f"Loaded dataset={spec.dataset_name}, train={len(train_raw)}, eval={len(eval_raw) if eval_raw is not None else 0}")

    train_ds = ShareGPTLlavaDataset(
        train_raw,
        tokenizer=tokenizer,
        spec=spec,
        cutoff_len=int(data_cfg.get("cutoff_len", 2048)),
        image_seq_len=image_seq_len,
        image_token=str(data_cfg.get("image_token", "<image>")),
        add_default_system=bool(data_cfg.get("add_default_system", True)),
        train_on_prompt=bool(data_cfg.get("train_on_prompt", False)),
    )
    eval_ds = None
    if eval_raw is not None:
        eval_ds = ShareGPTLlavaDataset(
            eval_raw,
            tokenizer=tokenizer,
            spec=spec,
            cutoff_len=int(data_cfg.get("cutoff_len", 2048)),
            image_seq_len=image_seq_len,
            image_token=str(data_cfg.get("image_token", "<image>")),
            add_default_system=bool(data_cfg.get("add_default_system", True)),
            train_on_prompt=bool(data_cfg.get("train_on_prompt", False)),
        )

    train_cfg = cfg["train"]
    collator = LlavaDataCollator(
        tokenizer=tokenizer,
        processor=processor,
        pad_to_multiple_of=train_cfg.get("pad_to_multiple_of"),
    )
    data_module = LlavaLightningDataModule(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        train_cfg=train_cfg,
    )
    lightning_module = LlavaLightningModule(model=model, train_cfg=train_cfg)
    lightning_trainer = build_lightning_trainer(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        train_cfg=train_cfg,
        wandb_project=str(wandb_project) if wandb_project else None,
    )

    resume_from_checkpoint = train_cfg.get("resume_from_checkpoint")
    lightning_trainer.fit(lightning_module, datamodule=data_module, ckpt_path=resume_from_checkpoint)

    if bool(train_cfg.get("save_model_at_end", False)):
        save_final_model(lightning_module.model, processor, str(train_cfg.get("output_dir", "outputs")))
    else:
        print("Training finished. Checkpoint/model saving disabled; metrics/logs are sent to W&B if enabled.")


if __name__ == "__main__":
    main()
