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
except ImportError:  # pragma: no cover - compatibility for older installations.
    import pytorch_lightning as pl
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

    if bool(model_cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

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



class LlavaLightningModule(pl.LightningModule):
    """Lightning wrapper that keeps the original HF model forward/loss behavior."""

    def __init__(self, model: torch.nn.Module, train_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.train_cfg = dict(train_cfg)

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=_infer_batch_size(batch),
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "eval/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=_infer_batch_size(batch),
        )
        return loss

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model, self.train_cfg)

        num_training_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
        max_steps = self.train_cfg.get("max_steps")
        if max_steps is not None and int(max_steps) > 0:
            num_training_steps = int(max_steps)
        if num_training_steps <= 0:
            num_training_steps = 1

        warmup_steps = self.train_cfg.get("warmup_steps")
        if warmup_steps is None:
            warmup_steps = 0
        warmup_steps = int(warmup_steps)
        if warmup_steps <= 0:
            warmup_steps = int(math.ceil(num_training_steps * float(self.train_cfg.get("warmup_ratio", 0.0))))

        scheduler = get_scheduler(
            name=str(self.train_cfg.get("lr_scheduler_type", "linear")),
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
            },
        }


def _infer_batch_size(batch: dict[str, Any]) -> int | None:
    input_ids = batch.get("input_ids") if isinstance(batch, dict) else None
    if isinstance(input_ids, torch.Tensor) and input_ids.ndim > 0:
        return int(input_ids.shape[0])
    return None


def normalize_report_to(value: Any) -> list[str]:
    if value is None:
        return ["wandb"]
    if isinstance(value, str):
        parts = [x.strip() for x in value.split(",") if x.strip()]
    else:
        parts = [str(x).strip() for x in value if str(x).strip()]

    disabled = {"none", "no", "false", "null", "[]"}
    if not parts or any(x.lower() in disabled for x in parts):
        return []
    return parts


def build_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    decay_parameters: set[str]
    try:
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        from transformers.trainer_pt_utils import get_parameter_names

        decay_parameters = set(get_parameter_names(model, ALL_LAYERNORM_LAYERS))
        decay_parameters = {name for name in decay_parameters if "bias" not in name}
    except Exception:
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
        decay_parameters = {
            name
            for name, param in model.named_parameters()
            if param.requires_grad and not any(nd in name for nd in no_decay)
        }

    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    optimizer_grouped_parameters = [
        {
            "params": [param for name, param in named_params if name in decay_parameters],
            "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        },
        {
            "params": [param for name, param in named_params if name not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer_grouped_parameters = [group for group in optimizer_grouped_parameters if group["params"]]

    optim_name = str(train_cfg.get("optim", "adamw_torch")).lower()
    betas = (float(train_cfg.get("adam_beta1", 0.9)), float(train_cfg.get("adam_beta2", 0.999)))
    eps = float(train_cfg.get("adam_epsilon", 1e-8))
    lr = float(train_cfg.get("learning_rate", 5e-5))

    if optim_name in {"adamw", "adamw_hf", "adamw_torch", "adamw_torch_fused"}:
        kwargs: dict[str, Any] = {"lr": lr, "betas": betas, "eps": eps}
        if optim_name == "adamw_torch_fused":
            kwargs["fused"] = True
        return torch.optim.AdamW(optimizer_grouped_parameters, **kwargs)
    if optim_name == "sgd":
        return torch.optim.SGD(optimizer_grouped_parameters, lr=lr)

    raise ValueError(f"Unsupported optimizer for Lightning path: {optim_name!r}")


def build_dataloader(
    dataset,
    collator: LlavaDataCollator,
    train_cfg: dict[str, Any],
    *,
    train: bool,
) -> DataLoader:
    batch_key = "per_device_train_batch_size" if train else "per_device_eval_batch_size"
    default_batch_size = train_cfg.get("per_device_train_batch_size", 1)
    batch_size = int(train_cfg.get(batch_key, default_batch_size))
    num_workers = int(train_cfg.get("dataloader_num_workers", 0))
    pin_memory = bool(train_cfg.get("dataloader_pin_memory", True))
    persistent_workers = bool(train_cfg.get("dataloader_persistent_workers", False)) and num_workers > 0
    drop_last = bool(train_cfg.get("dataloader_drop_last", False)) if train else False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )


def configure_tf32(train_cfg: dict[str, Any]) -> None:
    tf32 = train_cfg.get("tf32")
    if tf32 is None or not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)


def resolve_precision(train_cfg: dict[str, Any]) -> str:
    if bool(train_cfg.get("bf16", False)):
        return "bf16-mixed"
    if bool(train_cfg.get("fp16", False)):
        return "16-mixed"
    return "32-true"


def resolve_devices(train_cfg: dict[str, Any]) -> Any:
    if "devices" in train_cfg:
        return train_cfg["devices"]
    if "num_devices" in train_cfg:
        return train_cfg["num_devices"]
    # HF Trainer uses one device for a normal `python train.py` launch. With torchrun,
    # WORLD_SIZE is set and Lightning should attach to the distributed environment.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return "auto"
    return 1


def resolve_strategy(train_cfg: dict[str, Any], devices: Any):
    deepspeed_config = train_cfg.get("deepspeed")
    if deepspeed_config:
        return DeepSpeedStrategy(config=deepspeed_config)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1 or (isinstance(devices, int) and devices > 1) or devices == "auto"
    if not distributed:
        return "auto"

    find_unused = train_cfg.get("ddp_find_unused_parameters")
    timeout_seconds = int(train_cfg.get("ddp_timeout", 1800))
    kwargs: dict[str, Any] = {"timeout": timedelta(seconds=timeout_seconds)}
    if find_unused is not None:
        kwargs["find_unused_parameters"] = bool(find_unused)
    return DDPStrategy(**kwargs)


def build_logger(cfg: dict[str, Any]):
    train_cfg = cfg.get("train", {})
    report_to = normalize_report_to(train_cfg.get("report_to"))
    if "wandb" not in {x.lower() for x in report_to}:
        return False

    return WandbLogger(
        project=os.environ.get("WANDB_PROJECT") or cfg.get("wandb_project") or train_cfg.get("wandb_project"),
        name=train_cfg.get("run_name"),
        save_dir=train_cfg.get("output_dir", "."),
    )


def build_checkpoint_callbacks(train_cfg: dict[str, Any]) -> list[ModelCheckpoint]:
    save_checkpoint = bool(train_cfg.get("save_checkpoint", False))
    save_strategy = str(train_cfg.get("save_strategy", "steps")).lower()
    if not save_checkpoint or save_strategy == "no":
        return []

    output_dir = str(train_cfg.get("output_dir", "outputs"))
    save_total_limit = train_cfg.get("save_total_limit")
    save_top_k = int(save_total_limit) if save_total_limit is not None else -1
    common = {
        "dirpath": os.path.join(output_dir, "checkpoints"),
        "filename": "epoch={epoch}-step={step}",
        "save_top_k": save_top_k,
        "save_last": False,
        "auto_insert_metric_name": False,
    }

    if save_strategy == "epoch":
        return [ModelCheckpoint(every_n_epochs=1, **common)]
    if save_strategy == "steps":
        return [ModelCheckpoint(every_n_train_steps=int(train_cfg.get("save_steps", 500)), **common)]

    raise ValueError(f"Unsupported save_strategy for Lightning path: {save_strategy!r}")


def build_lightning_trainer(cfg: dict[str, Any], train_loader: DataLoader, has_eval: bool) -> pl.Trainer:
    train_cfg = cfg.get("train", {})
    configure_tf32(train_cfg)

    callbacks = build_checkpoint_callbacks(train_cfg)
    devices = resolve_devices(train_cfg)

    num_train_epochs = float(train_cfg.get("num_train_epochs", 1.0))
    max_epochs = int(math.ceil(num_train_epochs))
    accumulate_grad_batches = int(train_cfg.get("gradient_accumulation_steps", 1))

    max_steps = train_cfg.get("max_steps")
    if max_steps is None:
        # Lightning's max_epochs is integer-only; compute max_steps for fractional
        # epochs so HF-style values such as 0.5 or 1.5 keep the same meaning.
        if not num_train_epochs.is_integer():
            updates_per_epoch = max(1, math.ceil(len(train_loader) / accumulate_grad_batches))
            max_steps = int(math.ceil(num_train_epochs * updates_per_epoch))
        else:
            max_steps = -1
    else:
        max_steps = int(max_steps)
    if max_steps > 0:
        # HF TrainingArguments.max_steps overrides num_train_epochs.
        max_epochs = -1

    # Lightning accepts integer val_check_interval as a number of train batches. HF's
    # eval_steps are optimizer-update steps, so multiply by gradient accumulation.
    eval_strategy = str(train_cfg.get("eval_strategy", train_cfg.get("evaluation_strategy", "no"))).lower()
    val_check_interval: int | float | None = None
    check_val_every_n_epoch: int | None = 1
    limit_val_batches: int | float | None = None
    if not has_eval or eval_strategy == "no":
        limit_val_batches = 0
    elif eval_strategy == "steps":
        val_check_interval = max(1, int(train_cfg.get("eval_steps", 500)) * accumulate_grad_batches)
        check_val_every_n_epoch = None
    elif eval_strategy == "epoch":
        check_val_every_n_epoch = 1
    else:
        raise ValueError(f"Unsupported eval_strategy for Lightning path: {eval_strategy!r}")

    trainer_kwargs: dict[str, Any] = {
        "default_root_dir": train_cfg.get("output_dir", "outputs"),
        "accelerator": train_cfg.get("accelerator", "auto"),
        "devices": devices,
        "strategy": resolve_strategy(train_cfg, devices),
        "precision": resolve_precision(train_cfg),
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "accumulate_grad_batches": accumulate_grad_batches,
        "gradient_clip_val": float(train_cfg.get("max_grad_norm", 1.0)),
        "logger": build_logger(cfg),
        "callbacks": callbacks,
        "enable_checkpointing": bool(callbacks),
        "log_every_n_steps": int(train_cfg.get("logging_steps", 50)),
        "num_sanity_val_steps": int(train_cfg.get("num_sanity_val_steps", 0)),
        "use_distributed_sampler": bool(train_cfg.get("use_distributed_sampler", True)),
    }
    if val_check_interval is not None:
        trainer_kwargs["val_check_interval"] = val_check_interval
    if check_val_every_n_epoch is not None:
        trainer_kwargs["check_val_every_n_epoch"] = check_val_every_n_epoch
    if limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = limit_val_batches

    return pl.Trainer(**trainer_kwargs)


def maybe_save_final_model(trainer: pl.Trainer, lightning_module: LlavaLightningModule, processor, output_dir: str) -> None:
    if not trainer.is_global_zero:
        return
    os.makedirs(output_dir, exist_ok=True)
    lightning_module.model.save_pretrained(output_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir)


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.overrides)

    seed = int(cfg.get("seed", cfg.get("data", {}).get("seed", 42)))
    set_seed(seed)
    pl.seed_everything(seed, workers=True)

    wandb_project = cfg.get("wandb_project") or cfg.get("train", {}).get("wandb_project")
    if wandb_project:
        os.environ.setdefault("WANDB_PROJECT", str(wandb_project))

    model, processor, tokenizer = load_vision_language_model(cfg["model"])
    model = apply_lora(model, cfg.get("lora", {}))

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

    lightning_module = LlavaLightningModule(model=model, train_cfg=train_cfg)
    trainer = build_lightning_trainer(cfg, train_loader, has_eval=eval_loader is not None)
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=eval_loader,
        ckpt_path=train_cfg.get("resume_from_checkpoint"),
    )

    output_dir = str(train_cfg.get("output_dir", "outputs"))
    if bool(train_cfg.get("save_model_at_end", False)):
        maybe_save_final_model(trainer, lightning_module, processor, output_dir)
    elif trainer.is_global_zero:
        print("Training finished. Checkpoint/model saving disabled; metrics/logs are sent to W&B when enabled.")


if __name__ == "__main__":
    main()
