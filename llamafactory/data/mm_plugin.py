from __future__ import annotations

import math
import os
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, BinaryIO, Union

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as ImageObject
from transformers.image_utils import get_image_size, make_flat_list_of_images, to_numpy_array

from ..extras import IMAGE_PLACEHOLDER

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

ImageInput = Union[str, bytes, dict[str, Any], BinaryIO, ImageObject]


def _as_int(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _prod(value: Any) -> int:
    if hasattr(value, "prod"):
        return _as_int(value.prod())
    return int(np.prod(value))


def _to_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [_as_int(x) for x in value]


def _square_or_product(value: Any, default: int = 2) -> int:
    value = default if value is None else value
    if isinstance(value, (tuple, list)):
        return int(math.prod(value))
    return int(value) ** 2


@dataclass
class BasePlugin:
    image_token: str | None = None
    expand_mm_tokens: bool = True

    def _validate_input(self, processor: "ProcessorMixin | None", images: list[ImageInput]) -> None:
        if images and self.image_token is None:
            raise ValueError("This template does not support image input.")
        if images and (processor is None or getattr(processor, "image_processor", None) is None):
            raise ValueError("Image processor was not found for image SFT.")

    def _validate_messages(self, messages: list[dict[str, str]], images: list[ImageInput]) -> None:
        placeholder_count = sum(message.get("content", "").count(IMAGE_PLACEHOLDER) for message in messages)
        if placeholder_count != len(images):
            raise ValueError(
                f"Image count ({len(images)}) does not match {IMAGE_PLACEHOLDER} count ({placeholder_count})."
            )

    def _preprocess_image(self, image: ImageObject, image_max_pixels: int, image_min_pixels: int) -> ImageObject:
        pixels = image.width * image.height
        if pixels > image_max_pixels:
            ratio = math.sqrt(image_max_pixels / pixels)
            image = image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))))
        elif pixels < image_min_pixels:
            ratio = math.sqrt(image_min_pixels / pixels)
            image = image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))))
        return image.convert("RGB") if image.mode != "RGB" else image

    def _regularize_images(self, images: list[ImageInput], processor: "ProcessorMixin") -> list[ImageObject]:
        results: list[ImageObject] = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                raw_bytes = image.get("bytes")
                image = Image.open(BytesIO(raw_bytes)) if raw_bytes is not None else Image.open(image["path"])
            elif not isinstance(image, ImageObject):
                image = Image.open(image)

            results.append(
                self._preprocess_image(
                    image,
                    image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                    image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
                )
            )
        return results

    def _get_mm_inputs(self, images: list[ImageInput], processor: "ProcessorMixin") -> dict[str, torch.Tensor]:
        if not images:
            return {}
        image_processor = getattr(processor, "image_processor")
        images = self._regularize_images(images, processor)
        return dict(image_processor(images, return_tensors="pt"))

    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        processor: "ProcessorMixin | None",
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images)
        self._validate_messages(messages, images)
        return messages

    def process_token_ids(
        self,
        input_ids: list[int],
        labels: list[int] | None,
        images: list[ImageInput],
        tokenizer: "PreTrainedTokenizer",
        processor: "ProcessorMixin | None",
    ) -> tuple[list[int], list[int] | None]:
        self._validate_input(processor, images)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: list[ImageInput],
        imglens: list[int],
        batch_ids: list[list[int]],
        processor: "ProcessorMixin | None",
    ) -> dict[str, Any]:
        self._validate_input(processor, images)
        return self._get_mm_inputs(images, processor) if images else {}


@dataclass
class LlavaPlugin(BasePlugin):
    def process_messages(self, messages: list[dict[str, str]], images: list[ImageInput], processor: "ProcessorMixin | None") -> list[dict[str, str]]:
        self._validate_input(processor, images)
        self._validate_messages(messages, images)
        messages = deepcopy(messages)

        image_seqlen = 1
        if self.expand_mm_tokens and images:
            mm_inputs = self._get_mm_inputs(images, processor)
            pixel_values = mm_inputs.get("pixel_values")
            if pixel_values is not None:
                height, width = get_image_size(to_numpy_array(pixel_values[0]))
                patch_size = getattr(processor, "patch_size", None) or getattr(processor.image_processor, "patch_size", 14)
                if isinstance(patch_size, dict):
                    patch_size = next(iter(patch_size.values()))
                image_seqlen = (height // int(patch_size)) * (width // int(patch_size))
                image_seqlen += int(getattr(processor, "num_additional_image_tokens", 1))
                if getattr(processor, "vision_feature_select_strategy", "default") == "default":
                    image_seqlen -= 1

        for message in messages:
            message["content"] = message["content"].replace(IMAGE_PLACEHOLDER, self.image_token * image_seqlen)
        return messages


@dataclass
class Qwen3VLPlugin(BasePlugin):
    vision_bos_token: str = "<|vision_start|>"
    vision_eos_token: str = "<|vision_end|>"

    def _preprocess_image(self, image: ImageObject, image_max_pixels: int, image_min_pixels: int) -> ImageObject:
        image = super()._preprocess_image(image, image_max_pixels, image_min_pixels)
        if min(image.width, image.height) < 28:
            image = image.resize((max(image.width, 28), max(image.height, 28)))
        if image.width / image.height > 200:
            image = image.resize((image.height * 180, image.height))
        if image.height / image.width > 200:
            image = image.resize((image.width, image.width * 180))
        return image

    def process_messages(self, messages: list[dict[str, str]], images: list[ImageInput], processor: "ProcessorMixin | None") -> list[dict[str, str]]:
        self._validate_input(processor, images)
        self._validate_messages(messages, images)
        messages = deepcopy(messages)
        image_grid_thw = self._get_mm_inputs(images, processor).get("image_grid_thw", []) if self.expand_mm_tokens else [None] * len(images)
        merge_length = _square_or_product(getattr(processor.image_processor, "merge_size", 2))

        image_idx = 0
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = _prod(image_grid_thw[image_idx]) // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                image_idx += 1
            message["content"] = content
        return messages


@dataclass
class InternVLPlugin(BasePlugin):
    def _get_mm_inputs(self, images: list[ImageInput], processor: "ProcessorMixin") -> dict[str, torch.Tensor]:
        if not images:
            return {}
        image_processor = getattr(processor, "image_processor")
        kwargs: dict[str, Any] = {}
        if getattr(processor, "crop_to_patches", False):
            kwargs.update(crop_to_patches=True, max_patches=12, min_patches=1)

        images = make_flat_list_of_images(
            self._regularize_images(
                images,
                processor,
            )
        )
        image_inputs = dict(image_processor(images=images, return_tensors="pt", **kwargs))
        num_patches = image_inputs.pop("num_patches", None)
        pixel_values = image_inputs.pop("pixel_values")
        if num_patches is None:
            num_patches = torch.ones(len(images), dtype=torch.long)
        image_inputs["pixel_values"] = pixel_values
        image_inputs["image_num_patches"] = num_patches
        return image_inputs

    def process_messages(self, messages: list[dict[str, str]], images: list[ImageInput], processor: "ProcessorMixin | None") -> list[dict[str, str]]:
        self._validate_input(processor, images)
        self._validate_messages(messages, images)
        messages = deepcopy(messages)
        patch_counts = _to_int_list(self._get_mm_inputs(images, processor).get("image_num_patches"))
        image_seqlen = int(getattr(processor, "image_seq_length", 256)) if self.expand_mm_tokens else 1

        image_idx = 0
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                count = patch_counts[image_idx] if self.expand_mm_tokens else 1
                content = content.replace(IMAGE_PLACEHOLDER, f"<img>{'<IMG_CONTEXT>' * image_seqlen * count}</img>", 1)
                image_idx += 1
            message["content"] = content
        return messages

    def get_mm_inputs(self, images: list[ImageInput], imglens: list[int], batch_ids: list[list[int]], processor: "ProcessorMixin | None") -> dict[str, Any]:
        self._validate_input(processor, images)
        mm_inputs = self._get_mm_inputs(images, processor) if images else {}
        mm_inputs.pop("image_num_patches", None)
        return mm_inputs


@dataclass
class KimiVLPlugin(BasePlugin):
    def process_messages(self, messages: list[dict[str, str]], images: list[ImageInput], processor: "ProcessorMixin | None") -> list[dict[str, str]]:
        self._validate_input(processor, images)
        self._validate_messages(messages, images)
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, processor) if self.expand_mm_tokens else {}
        image_grid_hws = mm_inputs.get("image_grid_hws") or mm_inputs.get("image_grid_thw") or [None] * len(images)
        merge_length = _square_or_product(getattr(processor.image_processor, "merge_kernel_size", 2))

        image_idx = 0
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = _prod(image_grid_hws[image_idx]) // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"<|media_start|>image<|media_content|>{self.image_token * image_seqlen}<|media_end|>",
                    1,
                )
                image_idx += 1
            message["content"] = content
        return messages


PLUGINS: dict[str, type[BasePlugin]] = {
    "base": BasePlugin,
    "llava": LlavaPlugin,
    "qwen3vl": Qwen3VLPlugin,
    "internvl": InternVLPlugin,
    "kimivl": KimiVLPlugin,
}


def get_mm_plugin(name: str, image_token: str | None = None, **kwargs: Any) -> BasePlugin:
    if name not in PLUGINS:
        raise ValueError(f"Only image SFT plugins are kept: {sorted(PLUGINS)}")
    return PLUGINS[name](image_token=image_token, **kwargs)
