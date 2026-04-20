from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from peft import PeftModel
from PIL import Image
from transformers import DataCollatorForSeq2Seq

from ..extras import IGNORE_INDEX, IMAGE_PLACEHOLDER

if TYPE_CHECKING:
    from transformers import ProcessorMixin
    from .template import Template


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """Image-only SFT collator.

    Real image samples keep their images. Pure-text samples get one white fake image
    plus fake image token ids masked out from attention/loss, so mixed batches still
    exercise the same multimodal forward path without training on the fake pixels.
    """

    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None
    fake_image_size: int = 64

    def __post_init__(self) -> None:
        if self.template is None:
            raise ValueError("Template is required for image SFT collation.")
        if isinstance(self.model, PeftModel):
            self.model = self.model.base_model.model
        if self.model is not None and hasattr(self.model, "get_rope_index"):
            self.get_rope_func = self.model.get_rope_index
        elif self.model is not None and hasattr(self.model, "model") and hasattr(self.model.model, "get_rope_index"):
            self.get_rope_func = self.model.model.get_rope_index
        else:
            self.get_rope_func = None

    def _fake_image_and_ids(self) -> tuple[Image.Image, list[int]]:
        fake_image = Image.new("RGB", (self.fake_image_size, self.fake_image_size), (255, 255, 255))
        messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
        messages = self.template.mm_plugin.process_messages(messages, [fake_image], self.processor)
        fake_ids = self.tokenizer.encode(messages[0]["content"], add_special_tokens=False)
        fake_ids, _ = self.template.mm_plugin.process_token_ids(fake_ids, None, [fake_image], self.tokenizer, self.processor)
        return fake_image, fake_ids

    def _add_fake_image_to_text_sample(self, feature: dict[str, Any], fake_ids: list[int]) -> None:
        if self.tokenizer.padding_side == "right":
            feature["input_ids"] = feature["input_ids"] + fake_ids
            feature["attention_mask"] = feature["attention_mask"] + [0] * len(fake_ids)
            feature["labels"] = feature["labels"] + [IGNORE_INDEX] * len(fake_ids)
        else:
            feature["input_ids"] = fake_ids + feature["input_ids"]
            feature["attention_mask"] = [0] * len(fake_ids) + feature["attention_mask"]
            feature["labels"] = [IGNORE_INDEX] * len(fake_ids) + feature["labels"]

    def _compute_rope_position_ids(self, features: dict[str, torch.Tensor], mm_inputs: dict[str, Any]) -> None:
        if self.get_rope_func is None:
            return
        params = inspect.signature(self.get_rope_func).parameters
        rope_kwargs: dict[str, Any] = {
            "input_ids": features["input_ids"],
            "image_grid_thw": mm_inputs.get("image_grid_thw"),
            "attention_mask": (features["attention_mask"] >= 1).float(),
        }
        for name in params:
            if name not in rope_kwargs and (name.endswith("_grid_thw") or name.endswith("_per_grid_ts")):
                rope_kwargs[name] = None
        if "mm_token_type_ids" in params:
            image_token_id = getattr(self.model.config, "image_token_id", None)
            token_type_ids = torch.zeros_like(features["input_ids"])
            if image_token_id is not None:
                token_type_ids[features["input_ids"] == image_token_id] = 1
            rope_kwargs["mm_token_type_ids"] = token_type_ids
        features["position_ids"], features["rope_deltas"] = self.get_rope_func(**rope_kwargs)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_images: list[Any] = []
        batch_imglens: list[int] = []
        batch_input_ids: list[list[int]] = []
        fake_image: Image.Image | None = None
        fake_ids: list[int] | None = None

        for feature in features:
            images = feature.pop("images", None) or []
            if not images and self.template.mm_plugin.image_token is not None:
                if fake_image is None or fake_ids is None:
                    fake_image, fake_ids = self._fake_image_and_ids()
                images = [fake_image]
                self._add_fake_image_to_text_sample(feature, fake_ids)

            batch_images.extend(images)
            batch_imglens.append(len(images))
            batch_input_ids.append(feature["input_ids"])

        mm_inputs = self.template.mm_plugin.get_mm_inputs(batch_images, batch_imglens, batch_input_ids, self.processor)
        token_type_ids = mm_inputs.pop("token_type_ids", None)
        if token_type_ids is not None:
            for feature, token_types in zip(features, token_type_ids):
                feature["token_type_ids"] = token_types

        padded: dict[str, torch.Tensor] = super().__call__(features)
        self._compute_rope_position_ids(padded, mm_inputs)
        padded.update(mm_inputs)
        return padded


@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    """Compatibility name kept; packed/4D attention was removed."""

    compute_dtype: torch.dtype = torch.float32

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        features = super().__call__(features)
        for key, value in list(features.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                features[key] = value.to(self.compute_dtype)
        return features
