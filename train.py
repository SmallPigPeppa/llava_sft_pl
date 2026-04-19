#!/usr/bin/env python3
"""Minimal Lightning Trainer script for LLaVA LoRA SFT.

Example:
    WANDB_PROJECT=CL-debug python train.py --config configs/demo2k.yaml
    python train.py --config configs/demo2k.yaml train.num_train_epochs=1 data.max_samples=64
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_scheduler, set_seed

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy
except ImportError as exc:  # pragma: no cover - fallback for older installs
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.loggers import WandbLogger
        from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
    except ImportError as fallback_exc:  # pragma: no cover
        raise ImportError(
            "Lightning is required. Install it with `pip install lightning` "
            "or install the updated requirements.txt."
        ) from fallback_exc

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
    model_name = model_cfg.get("model_name_or_path") or model_cfg.get("name_or_path")
    if not model_name:
        raise KeyError("model.model_name_or_path is required.")
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

    # 训练默认不使用 KV cache；gradient checkpointing 下必须关闭。
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
    elif isinstance(target_modules, str) and "," in target_modules:
        target_modules = [x.strip() for x in target_modules.split(",") if x.strip()]
    # A single string is kept as a string so PEFT can treat regex targets such as
    # `.*language_model.*(q_proj|k_proj|v_proj)$` correctly.

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


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [value]


def wants_wandb(report_to: Any) -> bool:
    values = [str(x).lower() for x in _as_list(report_to)]
    if not values:
        return False
    disabled = {"none", "no", "false", "null", "[]"}
    return "wandb" in values and not all(x in disabled for x in values)


def build_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    optim_name = str(train_cfg.get("optim", "adamw_torch")).lower()
    learning_rate = float(train_cfg.get("learning_rate", 5e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    beta1 = float(train_cfg.get("adam_beta1", 0.9))
    beta2 = float(train_cfg.get("adam_beta2", 0.999))
    eps = float(train_cfg.get("adam_epsilon", train_cfg.get("adam_eps", 1e-8)))

    decay_parameters: list[torch.nn.Parameter] = []
    no_decay_parameters: list[torch.nn.Parameter] = []
    no_decay_markers = ("bias", "layer_norm.weight", "layernorm.weight", "norm.weight")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if param.ndim <= 1 or any(marker in lowered for marker in no_decay_markers):
            no_decay_parameters.append(param)
        else:
            decay_parameters.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]

    if optim_name in {"adamw", "adamw_torch", "adamw_hf"}:
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, betas=(beta1, beta2), eps=eps)
    if optim_name == "adamw_torch_fused":
        kwargs: dict[str, Any] = {"lr": learning_rate, "betas": (beta1, beta2), "eps": eps}
        if torch.cuda.is_available():
            kwargs["fused"] = True
        return torch.optim.AdamW(optimizer_grouped_parameters, **kwargs)

    raise ValueError(
        f"Unsupported train.optim={optim_name!r} in the Lightning script. "
        "Supported values: adamw, adamw_torch, adamw_hf, adamw_torch_fused."
    )


class LlavaSFTLightningModule(pl.LightningModule):
    """LightningModule wrapper around the Hugging Face LLaVA model."""

    def __init__(self, model: torch.nn.Module, train_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.train_cfg = dict(train_cfg)

    def forward(self, **batch):
        return self.model(**batch)

    @staticmethod
    def _loss_from_outputs(outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "loss"):
            return outputs.loss
        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        raise ValueError("Model output does not contain a loss. Make sure labels are present in the batch.")

    @staticmethod
    def _batch_size(batch: dict[str, torch.Tensor]) -> int:
        input_ids = batch.get("input_ids")
        if isinstance(input_ids, torch.Tensor) and input_ids.ndim > 0:
            return int(input_ids.shape[0])
        return 1

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = self._loss_from_outputs(outputs)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self._batch_size(batch),
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = self._loss_from_outputs(outputs)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self._batch_size(batch),
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model, self.train_cfg)

        scheduler_name = str(self.train_cfg.get("lr_scheduler_type", "linear")).lower()
        if scheduler_name in {"none", "no", "null"}:
            return optimizer

        estimated_steps = self.trainer.estimated_stepping_batches
        if isinstance(estimated_steps, float) and math.isinf(estimated_steps):
            max_steps = self.train_cfg.get("max_steps")
            if max_steps is None:
                return optimizer
            estimated_steps = int(max_steps)
        num_training_steps = max(1, int(estimated_steps))

        warmup_steps = self.train_cfg.get("warmup_steps")
        if warmup_steps is None:
            warmup_steps = int(float(self.train_cfg.get("warmup_ratio", 0.0)) * num_training_steps)
        else:
            warmup_steps = int(warmup_steps)

        scheduler = get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": scheduler_name,
            },
        }


def build_dataloader(dataset, collator: LlavaDataCollator, train_cfg: dict[str, Any], train: bool) -> DataLoader:
    if train:
        batch_size = int(train_cfg.get("per_device_train_batch_size", 1))
        shuffle = bool(train_cfg.get("shuffle", True))
    else:
        batch_size = int(train_cfg.get("per_device_eval_batch_size", train_cfg.get("per_device_train_batch_size", 1)))
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        collate_fn=collator,
        pin_memory=bool(train_cfg.get("dataloader_pin_memory", torch.cuda.is_available())),
        drop_last=bool(train_cfg.get("dataloader_drop_last", False)) if train else False,
    )


def build_precision(train_cfg: dict[str, Any]) -> str:
    if bool(train_cfg.get("bf16", False)):
        return "bf16-mixed"
    if bool(train_cfg.get("fp16", False)):
        return "16-mixed"
    return str(train_cfg.get("precision", "32-true"))


def build_logger(cfg: dict[str, Any], train_cfg: dict[str, Any]):
    report_to = train_cfg.get("report_to", ["wandb"])
    if not wants_wandb(report_to):
        return False

    project = cfg.get("wandb_project") or train_cfg.get("wandb_project") or os.environ.get("WANDB_PROJECT")
    if project:
        os.environ.setdefault("WANDB_PROJECT", str(project))

    return WandbLogger(
        project=str(project) if project else None,
        name=train_cfg.get("run_name"),
        save_dir=str(train_cfg.get("output_dir", ".")),
    )


def build_checkpoint_callbacks(train_cfg: dict[str, Any], has_eval: bool) -> list[ModelCheckpoint]:
    save_strategy = str(train_cfg.get("save_strategy", "no")).lower()
    save_checkpoint = bool(train_cfg.get("save_checkpoint", False)) and save_strategy not in {"no", "none", "false"}
    if not save_checkpoint:
        return []

    output_dir = str(train_cfg.get("output_dir", "lightning_output"))
    save_total_limit = train_cfg.get("save_total_limit")
    save_top_k = int(save_total_limit) if save_total_limit is not None else -1
    monitor = "val_loss" if has_eval else None
    checkpoint_kwargs: dict[str, Any] = {
        "dirpath": output_dir,
        "filename": "epoch={epoch}-step={step}",
        "auto_insert_metric_name": False,
        "save_last": True,
        "save_weights_only": False,
    }
    if monitor is None:
        checkpoint_kwargs["monitor"] = None
        checkpoint_kwargs["save_top_k"] = -1 if save_top_k != 0 else 0
    else:
        checkpoint_kwargs["monitor"] = monitor
        checkpoint_kwargs["mode"] = "min"
        checkpoint_kwargs["save_top_k"] = save_top_k

    if save_strategy in {"steps", "step"}:
        every_n_train_steps = int(train_cfg.get("save_steps", 0) or 0)
        if every_n_train_steps <= 0:
            return []
        return [
            ModelCheckpoint(
                every_n_train_steps=every_n_train_steps,
                every_n_epochs=0,
                **checkpoint_kwargs,
            )
        ]
    if save_strategy in {"epoch", "epochs"}:
        return [
            ModelCheckpoint(
                every_n_epochs=1,
                every_n_train_steps=None,
                **checkpoint_kwargs,
            )
        ]

    return []


def _devices_request_multiple_devices(devices: Any) -> bool:
    if devices is None:
        return False
    if isinstance(devices, int):
        return devices > 1 or devices == -1
    if isinstance(devices, (list, tuple)):
        return len(devices) > 1
    if isinstance(devices, str):
        normalized = devices.strip().lower()
        if normalized in {"auto"}:
            return int(os.environ.get("WORLD_SIZE", "1")) > 1
        if normalized in {"-1", "all"}:
            return True
        if normalized.isdigit():
            return int(normalized) > 1
        if "," in normalized:
            return len([x for x in normalized.split(",") if x.strip()]) > 1
    return False


def build_strategy(train_cfg: dict[str, Any]):
    if train_cfg.get("strategy") is not None:
        return train_cfg["strategy"]

    if train_cfg.get("deepspeed") is not None:
        return DeepSpeedStrategy(config=train_cfg["deepspeed"])

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    devices = train_cfg.get("devices")
    use_ddp = world_size > 1 or _devices_request_multiple_devices(devices)
    if not use_ddp:
        return "auto"

    kwargs: dict[str, Any] = {}
    if train_cfg.get("ddp_find_unused_parameters") is not None:
        kwargs["find_unused_parameters"] = bool(train_cfg.get("ddp_find_unused_parameters"))
    if train_cfg.get("ddp_timeout") is not None:
        kwargs["timeout"] = timedelta(seconds=int(train_cfg["ddp_timeout"]))
    return DDPStrategy(**kwargs)


def build_trainer(cfg: dict[str, Any], train_cfg: dict[str, Any], has_eval: bool, num_train_batches: int | None = None):
    callbacks = build_checkpoint_callbacks(train_cfg, has_eval=has_eval)
    logger = build_logger(cfg, train_cfg)

    max_steps = train_cfg.get("max_steps")
    max_epochs_value = train_cfg.get("num_train_epochs", 1)
    max_epochs = int(math.ceil(float(max_epochs_value))) if max_epochs_value is not None else None

    trainer_kwargs: dict[str, Any] = {
        "default_root_dir": str(train_cfg.get("output_dir", "lightning_output")),
        "accelerator": train_cfg.get("accelerator", "auto"),
        "devices": train_cfg.get("devices", "auto"),
        "num_nodes": int(train_cfg.get("num_nodes", 1)),
        "precision": build_precision(train_cfg),
        "max_epochs": max_epochs,
        "max_steps": int(max_steps) if max_steps is not None else -1,
        "accumulate_grad_batches": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "logger": logger,
        "callbacks": callbacks,
        "enable_checkpointing": bool(callbacks),
        "log_every_n_steps": int(train_cfg.get("logging_steps", 50)),
        "num_sanity_val_steps": int(train_cfg.get("num_sanity_val_steps", 0)),
        "strategy": build_strategy(train_cfg),
    }

    if train_cfg.get("gradient_clip_val") is not None:
        trainer_kwargs["gradient_clip_val"] = float(train_cfg["gradient_clip_val"])
    if train_cfg.get("gradient_clip_algorithm") is not None:
        trainer_kwargs["gradient_clip_algorithm"] = train_cfg["gradient_clip_algorithm"]
    if train_cfg.get("deterministic") is not None:
        trainer_kwargs["deterministic"] = bool(train_cfg["deterministic"])
    if train_cfg.get("enable_progress_bar") is not None:
        trainer_kwargs["enable_progress_bar"] = bool(train_cfg["enable_progress_bar"])

    eval_strategy = str(train_cfg.get("eval_strategy", train_cfg.get("evaluation_strategy", "no"))).lower()
    if not has_eval or eval_strategy in {"no", "none", "false"}:
        trainer_kwargs["limit_val_batches"] = 0
    elif eval_strategy in {"steps", "step"} and train_cfg.get("eval_steps") is not None:
        # Lightning's integer val_check_interval is counted in train batches.
        # Keep the same numeric config value when valid; if a small debug run has
        # fewer batches than eval_steps, validate at epoch end instead of crashing.
        eval_steps = int(train_cfg["eval_steps"])
        if num_train_batches is not None and eval_steps > num_train_batches:
            trainer_kwargs["val_check_interval"] = 1.0
        else:
            trainer_kwargs["val_check_interval"] = eval_steps
    elif eval_strategy in {"epoch", "epochs"}:
        trainer_kwargs["check_val_every_n_epoch"] = 1

    return pl.Trainer(**trainer_kwargs)


def resolve_resume_checkpoint(path_like: Any) -> str | None:
    if path_like is None:
        return None
    if isinstance(path_like, str) and path_like.lower() in {"last", "hpc"}:
        return path_like

    path = Path(str(path_like)).expanduser()
    if path.is_dir():
        candidates = sorted(path.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(
                f"resume_from_checkpoint points to a directory with no Lightning .ckpt files: {path}"
            )
        return str(candidates[-1])
    return str(path)


def save_final_pretrained(trainer: Any, lightning_module: LlavaSFTLightningModule, processor: Any, output_dir: str) -> None:
    if not trainer.is_global_zero:
        return
    os.makedirs(output_dir, exist_ok=True)
    model = lightning_module.model
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir)


def normalize_lora_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Keep demo2k.yaml unchanged while accepting the older top-level model LoRA aliases."""
    lora_cfg = dict(cfg.get("lora") or {})
    model_cfg = cfg.get("model", {})
    if "use_lora" in model_cfg and "enable" not in lora_cfg:
        lora_cfg["enable"] = model_cfg.get("use_lora")
    alias_map = {
        "lora_r": "r",
        "lora_alpha": "alpha",
        "lora_dropout": "dropout",
        "lora_target": "target_modules",
    }
    for source_key, target_key in alias_map.items():
        if source_key in model_cfg and target_key not in lora_cfg:
            lora_cfg[target_key] = model_cfg[source_key]
    return lora_cfg


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.overrides)

    seed = int(cfg.get("seed", cfg.get("data", {}).get("seed", 42)))
    set_seed(seed)
    pl.seed_everything(seed, workers=True)
    apply_runtime_flags(cfg.get("train", {}))

    wandb_project = cfg.get("wandb_project") or cfg.get("train", {}).get("wandb_project")
    if wandb_project:
        os.environ.setdefault("WANDB_PROJECT", str(wandb_project))

    model, processor, tokenizer = load_vision_language_model(cfg["model"])
    model = apply_lora(model, normalize_lora_config(cfg))

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
    train_loader = build_dataloader(train_ds, collator, train_cfg, train=True)
    eval_loader = build_dataloader(eval_ds, collator, train_cfg, train=False) if eval_ds is not None else None

    lightning_module = LlavaSFTLightningModule(model=model, train_cfg=train_cfg)
    trainer = build_trainer(cfg, train_cfg, has_eval=eval_ds is not None, num_train_batches=len(train_loader))
    ckpt_path = resolve_resume_checkpoint(train_cfg.get("resume_from_checkpoint"))
    trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=eval_loader, ckpt_path=ckpt_path)

    if bool(train_cfg.get("save_model_at_end", False)):
        save_final_pretrained(trainer, lightning_module, processor, str(train_cfg["output_dir"]))
    elif trainer.is_global_zero:
        print("Training finished. Checkpoint/model saving disabled; metrics/logs are sent to W&B if enabled.")


if __name__ == "__main__":
    main()
