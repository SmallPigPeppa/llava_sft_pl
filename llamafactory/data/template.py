from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

IMAGE_PLACEHOLDER = "<image>"
VIDEO_PLACEHOLDER = "<video>"
AUDIO_PLACEHOLDER = "<audio>"
IGNORE_INDEX = -100
BOS = "__bos_token__"
EOS = "__eos_token__"

Slot = str


def _replace(slot: str, content: str) -> str:
    return slot.replace("{{content}}", content)


def _load_image(image: Any):
    try:
        from PIL import Image
        from PIL.Image import Image as PILImage
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("Pillow is required for image training data. Install `pillow`.") from exc

    if isinstance(image, PILImage):
        img = image
    elif isinstance(image, bytes):
        img = Image.open(BytesIO(image))
    elif isinstance(image, dict):
        if image.get("bytes") is not None:
            img = Image.open(BytesIO(image["bytes"]))
        elif image.get("path") is not None:
            img = Image.open(image["path"])
        else:
            raise ValueError(f"Unsupported image dict: {image!r}")
    else:
        img = Image.open(image)

    return img.convert("RGB") if getattr(img, "mode", "RGB") != "RGB" else img


def _resize_for_limits(image: Any, max_pixels: int, min_pixels: int):
    pixels = image.width * image.height
    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    return image


@dataclass
class Template:
    """Compact chat template with the LLaVA-family multimodal handling kept intact."""

    name: str
    user: list[Slot]
    assistant: list[Slot] = field(default_factory=lambda: ["{{content}}", EOS])
    system: list[Slot] = field(default_factory=lambda: ["{{content}}"])
    prefix: list[Slot] = field(default_factory=list)
    default_system: str = ""
    stop_words: list[str] = field(default_factory=list)
    replace_eos: bool = False
    efficient_eos: bool = False
    image_token: str | None = IMAGE_PLACEHOLDER
    video_token: str | None = None
    image_strategy: str = "llava"  # llava | llava_next | none

    def clone(self) -> "Template":
        return copy.deepcopy(self)

    def _slot_to_ids(self, tokenizer: Any, slot: Slot, content: str = "") -> list[int]:
        if slot == BOS:
            return [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
        if slot == EOS:
            return [] if tokenizer.eos_token_id is None else [tokenizer.eos_token_id]
        text = _replace(slot, content)
        return tokenizer.encode(text, add_special_tokens=False) if text else []

    def _encode_slots(self, tokenizer: Any, slots: list[Slot], content: str = "") -> list[int]:
        ids: list[int] = []
        for slot in slots:
            ids.extend(self._slot_to_ids(tokenizer, slot, content))
        return ids

    def encode_multiturn(
        self,
        tokenizer: Any,
        messages: list[dict[str, str]],
        system: str | None = None,
        tools: str | None = None,
    ) -> list[tuple[list[int], list[int]]]:
        if len(messages) % 2 != 0:
            raise ValueError("SFT messages must be user/assistant pairs.")

        system_text = self.default_system if system is None or system == "" else system
        if tools:
            system_text = (system_text + "\n" if system_text else "") + str(tools)

        encoded: list[list[int]] = []
        for i, message in enumerate(messages):
            role, content = message["role"], message["content"]
            if i % 2 == 0 and role != "user":
                raise ValueError(f"Expected user message at turn {i}, got {role!r}.")
            if i % 2 == 1 and role != "assistant":
                raise ValueError(f"Expected assistant message at turn {i}, got {role!r}.")

            if role == "user":
                ids: list[int] = []
                if i == 0:
                    ids.extend(self._encode_slots(tokenizer, self.prefix))
                    if system_text:
                        ids.extend(self._encode_slots(tokenizer, self.system, system_text))
                ids.extend(self._encode_slots(tokenizer, self.user, content))
            else:
                ids = self._encode_slots(tokenizer, self.assistant, content)
            encoded.append(ids)

        pairs = [(encoded[i], encoded[i + 1]) for i in range(0, len(encoded), 2)]
        return pairs

    def fix_special_tokens(self, tokenizer: Any) -> None:
        if self.replace_eos and self.stop_words:
            eos = self.stop_words[0]
            if tokenizer.eos_token != eos:
                tokenizer.add_special_tokens({"eos_token": eos})

        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        extra = self.stop_words[1:] if self.replace_eos else self.stop_words
        if extra:
            try:
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": extra},
                    replace_additional_special_tokens=False,
                )
            except TypeError:  # older transformers
                tokenizer.add_special_tokens({"additional_special_tokens": extra})

    def _regularize_images(self, images: list[Any], processor: Any) -> list[Any]:
        max_pixels = int(getattr(processor, "image_max_pixels", 768 * 768))
        min_pixels = int(getattr(processor, "image_min_pixels", 32 * 32))
        return [_resize_for_limits(_load_image(image), max_pixels, min_pixels) for image in images]

    def image_inputs(self, images: list[Any], processor: Any) -> dict[str, Any]:
        if not images:
            return {}
        if processor is None or getattr(processor, "image_processor", None) is None:
            raise ValueError("Image data requires a Hugging Face processor with an image_processor.")
        return processor.image_processor(self._regularize_images(images, processor), return_tensors="pt")

    def _image_token_counts(self, images: list[Any], processor: Any) -> list[int]:
        if not images:
            return []
        if self.image_strategy == "none":
            return [1] * len(images)

        mm_inputs = self.image_inputs(images, processor)
        strategy = getattr(processor, "vision_feature_select_strategy", "default")
        patch_size = int(getattr(processor, "patch_size", 14))
        additional = int(getattr(processor, "num_additional_image_tokens", 1))

        if self.image_strategy == "llava_next" and "image_sizes" in mm_inputs:
            pixel_values = mm_inputs["pixel_values"]
            # LlavaNext image processor returns [num_images, num_patches, C, H, W].
            first_tile = pixel_values[0][0] if getattr(pixel_values, "ndim", 0) == 5 else pixel_values[0]
            height, width = int(first_tile.shape[-2]), int(first_tile.shape[-1])
            counts = []
            for orig_height, orig_width in mm_inputs["image_sizes"].tolist():
                count = int(processor._get_number_of_features(orig_height, orig_width, height, width))
                counts.append(count - 1 if strategy == "default" else count)
            return counts

        pixel_values = mm_inputs.get("pixel_values")
        if pixel_values is None:
            return [1] * len(images)
        height, width = int(pixel_values[0].shape[-2]), int(pixel_values[0].shape[-1])
        count = (height // patch_size) * (width // patch_size) + additional
        if strategy == "default":
            count -= 1
        return [max(1, count)] * len(images)

    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[Any],
        videos: list[Any] | None,
        audios: list[Any] | None,
        processor: Any,
    ) -> list[dict[str, str]]:
        if videos:
            raise ValueError("This slim build keeps image SFT only; video preprocessing was removed.")
        if audios:
            raise ValueError("This slim build keeps image SFT only; audio preprocessing was removed.")
        if images and self.image_token is None:
            raise ValueError(f"Template {self.name!r} does not support images.")

        expected_images = sum(m["content"].count(IMAGE_PLACEHOLDER) for m in messages)
        if expected_images != len(images):
            raise ValueError(f"Found {expected_images} <image> placeholders but {len(images)} images.")

        counts = iter(self._image_token_counts(images, processor))
        processed = copy.deepcopy(messages)
        for message in processed:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                replacement = (self.image_token or IMAGE_PLACEHOLDER) * next(counts)
                content = content.replace(IMAGE_PLACEHOLDER, replacement, 1)
            message["content"] = content
        return processed


VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

CHATML_USER = ["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]
CHATML_ASSISTANT = ["{{content}}<|im_end|>\n"]
CHATML_SYSTEM = ["<|im_start|>system\n{{content}}<|im_end|>\n"]

TEMPLATES: dict[str, Template] = {
    "empty": Template(name="empty", user=["{{content}}"], image_token=None, image_strategy="none"),
    "default": Template(name="default", user=["{{content}}"], image_token=None, image_strategy="none"),
    "vicuna": Template(name="vicuna", user=["USER: {{content}} ASSISTANT:"], default_system=VICUNA_SYSTEM),
    "llava": Template(
        name="llava",
        user=["USER: {{content}} ASSISTANT:"],
        default_system=VICUNA_SYSTEM,
        image_token="<image>",
        image_strategy="llava",
    ),
    "llava_next": Template(
        name="llava_next",
        user=["USER: {{content}} ASSISTANT:"],
        default_system=VICUNA_SYSTEM,
        image_token="<image>",
        image_strategy="llava_next",
    ),
    "llava_next_llama3": Template(
        name="llava_next_llama3",
        user=["<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"],
        assistant=["{{content}}<|eot_id|>"],
        system=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"],
        prefix=[BOS],
        stop_words=["<|eot_id|>", "<|eom_id|>"],
        replace_eos=True,
        image_token="<image>",
        image_strategy="llava_next",
    ),
    "llava_next_mistral": Template(
        name="llava_next_mistral",
        user=["[INST] {{content}}[/INST]"],
        assistant=[" {{content}}", EOS],
        system=["{{content}}\n\n"],
        prefix=[BOS],
        image_token="<image>",
        image_strategy="llava_next",
    ),
    "llava_next_qwen": Template(
        name="llava_next_qwen",
        user=CHATML_USER,
        assistant=CHATML_ASSISTANT,
        system=CHATML_SYSTEM,
        default_system="You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        replace_eos=True,
        image_token="<image>",
        image_strategy="llava_next",
    ),
    "llava_next_yi": Template(
        name="llava_next_yi",
        user=CHATML_USER,
        assistant=CHATML_ASSISTANT,
        system=CHATML_SYSTEM,
        stop_words=["<|im_end|>"],
        image_token="<image>",
        image_strategy="llava_next",
    ),
    "yi_vl": Template(
        name="yi_vl",
        user=["### Human: {{content}}\n### Assistant:"],
        assistant=["{{content}}\n"],
        default_system=(
            "This is a chat between an inquisitive human and an AI assistant. "
            "Assume the role of the AI assistant. Read all the images carefully, "
            "and respond to the human's questions with informative, helpful, detailed and polite answers. "
            "这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。"
            "仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。\n\n"
        ),
        stop_words=["###"],
        efficient_eos=True,
        image_token="<image>",
        image_strategy="llava",
    ),
}

# Keep the original LLaVA-video template names discoverable. The slim build rejects
# actual video columns at preprocessing time with a clear error instead of carrying
# thousands of lines of video/audio code.
for _name, _base in {
    "video_llava": "llava",
    "llava_next_video": "llava_next",
    "llava_next_video_mistral": "llava_next_mistral",
    "llava_next_video_yi": "llava_next_yi",
}.items():
    TEMPLATES[_name] = TEMPLATES[_base].clone()
    TEMPLATES[_name].name = _name
    TEMPLATES[_name].video_token = VIDEO_PLACEHOLDER


def get_template_and_fix_tokenizer(tokenizer: Any, data_args: Any) -> Template:
    name = data_args.template or "empty"
    if name not in TEMPLATES:
        supported = ", ".join(sorted(TEMPLATES))
        raise ValueError(f"Template {name!r} was removed from this LLaVA-only build. Supported: {supported}.")

    template = TEMPLATES[name].clone()
    if data_args.default_system is not None:
        template.default_system = data_args.default_system
    if getattr(data_args, "tool_format", None) is not None:
        raise ValueError("Tool/function-call formatting was removed from this SFT-only project.")

    template.fix_special_tokens(tokenizer)
    return template
