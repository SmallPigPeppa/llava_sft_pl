from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

from ..extras import logging
from ..extras.constants import IMAGE_PLACEHOLDER
from .data_utils import Role

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments
    from ..hparams import DataArguments
    from .mm_plugin import AudioInput, ImageInput, VideoInput
    from .parser import DatasetAttr

    MediaType = Union[ImageInput, VideoInput, AudioInput]


logger = logging.get_logger(__name__)


@dataclass
class SharegptDatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"

    def _resolve_media(self, media: "MediaType") -> "MediaType":
        """Resolve relative local media paths while preserving bytes/PIL/HF image dicts."""
        if isinstance(media, str):
            candidate = os.path.join(self.data_args.media_dir, media) if not os.path.isabs(media) else media
            return candidate if os.path.isfile(candidate) else media

        if isinstance(media, dict) and media.get("path") is not None:
            path = str(media["path"])
            candidate = os.path.join(self.data_args.media_dir, path) if not os.path.isabs(path) else path
            if os.path.isfile(candidate):
                media = dict(media)
                media["path"] = candidate
            return media

        if isinstance(media, list):
            return [self._resolve_media(item) for item in media]  # type: ignore[list-item]

        return media

    def _find_medias(self, medias: Union["MediaType", list["MediaType"], None]) -> list["MediaType"] | None:
        if medias is None:
            return None
        if not isinstance(medias, list):
            medias = [medias]
        if len(medias) == 0:
            return None
        return [self._resolve_media(media) for media in medias]

    def _load_messages(self, raw_messages: Any) -> list[dict[str, Any]]:
        if isinstance(raw_messages, str):
            try:
                raw_messages = json.loads(raw_messages)
            except json.JSONDecodeError as exc:
                raise ValueError("ShareGPT messages column is a string but not valid JSON.") from exc
        if not isinstance(raw_messages, list):
            raise TypeError(f"ShareGPT messages must be a list, got {type(raw_messages)!r}.")
        return raw_messages

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        attr = self.dataset_attr
        messages = self._load_messages(example[attr.messages])

        tag_mapping = {
            attr.user_tag: Role.USER.value,
            attr.assistant_tag: Role.ASSISTANT.value,
            attr.observation_tag: Role.OBSERVATION.value,
            attr.function_tag: Role.FUNCTION.value,
            attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (attr.user_tag, attr.observation_tag)
        even_tags = (attr.assistant_tag, attr.function_tag)
        accept_tags = (odd_tags, even_tags)

        if attr.system_tag and len(messages) != 0 and messages[0][attr.role_tag] == attr.system_tag:
            system = messages[0][attr.content_tag]
            messages = messages[1:]
        else:
            system = example[attr.system] if attr.system else ""

        aligned_messages: list[dict[str, str]] = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            role = message[attr.role_tag]
            if role not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken_data = True
                break
            aligned_messages.append({"role": tag_mapping[role], "content": message[attr.content_tag]})

        if len(aligned_messages) % 2 != 0:
            logger.warning_rank0(f"Invalid message count in {messages}.")
            broken_data = True

        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        else:
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        images = self._find_medias(example[attr.images]) if attr.images else None
        if images:
            # Compatibility with earlier local wrapper: if image bytes/paths exist but
            # the text forgot logical <image> placeholders, add the missing ones.
            placeholder_count = sum(message["content"].count(IMAGE_PLACEHOLDER) for message in prompt + response)
            missing = max(0, len(images) - placeholder_count)
            if missing > 0:
                for message in prompt:
                    if message["role"] == Role.USER.value:
                        message["content"] = (IMAGE_PLACEHOLDER + "\n") * missing + message["content"]
                        break

        tools = example[attr.tools] if attr.tools else ""
        if isinstance(tools, (dict, list)):
            tools = json.dumps(tools, ensure_ascii=False)

        return {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": tools,
            "_images": images,
            "_videos": self._find_medias(example[attr.videos]) if attr.videos else None,
            "_audios": self._find_medias(example[attr.audios]) if attr.audios else None,
        }


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting ShareGPT format",
        )

    converter = SharegptDatasetConverter(dataset_attr, data_args)
    return dataset.map(converter, batched=False, remove_columns=column_names, **kwargs)
