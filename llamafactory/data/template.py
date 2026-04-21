from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from ..extras import get_logger
from .mm_plugin import BasePlugin, get_mm_plugin

logger = get_logger(__name__)
SLOTS = list[Union[str, set[str], dict[str, str]]]


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Formatter:
    slots: SLOTS = field(default_factory=list)

    def apply(self, **kwargs: Any) -> SLOTS:
        rendered: SLOTS = []
        for slot in self.slots:
            if isinstance(slot, str):
                text = slot
                for key, value in kwargs.items():
                    text = text.replace("{{" + key + "}}", "" if value is None else str(value))
                rendered.append(text)
            else:
                rendered.append(slot)
        return rendered


class EmptyFormatter(Formatter):
    pass


class StringFormatter(Formatter):
    pass


@dataclass
class Template:
    format_user: Formatter
    format_assistant: Formatter
    format_system: Formatter
    format_prefix: Formatter = field(default_factory=EmptyFormatter)
    default_system: str = ""
    stop_words: list[str] = field(default_factory=list)
    thought_words: tuple[str, str] = ("<think>\n", "\n</think>\n\n")
    efficient_eos: bool = False
    replace_eos: bool = False
    enable_thinking: Optional[bool] = True
    mm_plugin: BasePlugin = field(default_factory=lambda: get_mm_plugin("base"))

    def encode_multiturn(
        self,
        tokenizer: Any,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> list[tuple[list[int], list[int]]]:
        encoded = self._encode(tokenizer, messages, system)
        return [(encoded[i], encoded[i + 1]) for i in range(0, len(encoded), 2)]

    def _convert_elements_to_ids(self, tokenizer: Any, elements: SLOTS) -> list[int]:
        token_ids: list[int] = []
        for element in elements:
            if isinstance(element, str):
                token_ids.extend(tokenizer.encode(element, add_special_tokens=False))
            elif isinstance(element, set):
                if "bos_token" in element and tokenizer.bos_token_id is not None:
                    token_ids.append(tokenizer.bos_token_id)
                if "eos_token" in element and tokenizer.eos_token_id is not None:
                    token_ids.append(tokenizer.eos_token_id)
            elif isinstance(element, dict) and element.get("token"):
                token_ids.append(tokenizer.convert_tokens_to_ids(element["token"]))
        return token_ids

    def _encode(self, tokenizer: Any, messages: list[dict[str, str]], system: Optional[str]) -> list[list[int]]:
        system = self.default_system if system is None else system
        encoded: list[list[int]] = []
        for idx, message in enumerate(messages):
            elements: SLOTS = []
            if idx == 0:
                elements += self.format_prefix.apply()
                if system:
                    elements += self.format_system.apply(content=system)

            role, content = message["role"], message.get("content", "")
            if role == Role.USER.value:
                elements += self.format_user.apply(content=content)
            elif role == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=content)
            else:
                raise ValueError(f"Unexpected message role in SFT data: {role}")
            encoded.append(self._convert_elements_to_ids(tokenizer, elements))
        return encoded

    def add_thought(self, content: str = "") -> str:
        return f"{self.thought_words[0]}{self.thought_words[1]}" + content

    def remove_thought(self, content: str) -> str:
        pattern = re.compile(f"{re.escape(self.thought_words[0])}(.*?){re.escape(self.thought_words[1])}", re.DOTALL)
        return re.sub(pattern, "", content).lstrip("\n")

    def get_thought_word_ids(self, tokenizer: Any) -> list[int]:
        return tokenizer.encode(self.add_thought(), add_special_tokens=False)

    def fix_special_tokens(self, tokenizer: Any) -> None:
        stop_words = list(self.stop_words)
        if self.replace_eos:
            self._add_or_replace_eos_token(tokenizer, stop_words.pop(0))
        if tokenizer.eos_token_id is None:
            self._add_or_replace_eos_token(tokenizer, "<|endoftext|>")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info_rank0("Set pad token to eos token: %s", tokenizer.pad_token)
        if stop_words:
            try:
                tokenizer.add_special_tokens({"additional_special_tokens": stop_words}, replace_additional_special_tokens=False)
            except TypeError:
                tokenizer.add_special_tokens({"additional_special_tokens": stop_words})

    @staticmethod
    def _add_or_replace_eos_token(tokenizer: Any, eos_token: str) -> None:
        if tokenizer.eos_token != eos_token:
            tokenizer.add_special_tokens({"eos_token": eos_token})
            logger.info_rank0("Set eos token: %s", tokenizer.eos_token)


@dataclass
class ReasoningTemplate(Template):
    def encode_multiturn(
        self,
        tokenizer: Any,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> list[tuple[list[int], list[int]]]:
        messages = deepcopy(messages)
        if self.enable_thinking is False:
            for idx in range(1, len(messages), 2):
                messages[idx]["content"] = self.remove_thought(messages[idx]["content"])
        encoded = self._encode(tokenizer, messages, system)
        for idx in range(0, len(messages), 2):
            answer = messages[idx + 1]["content"]
            has_thought = self.thought_words[0].strip() in answer and self.thought_words[1].strip() in answer
            if not has_thought:
                if self.enable_thinking is False:
                    encoded[idx] += self.get_thought_word_ids(tokenizer)
                else:
                    encoded[idx + 1] = self.get_thought_word_ids(tokenizer) + encoded[idx + 1]
        return [(encoded[i], encoded[i + 1]) for i in range(0, len(encoded), 2)]


TEMPLATES: dict[str, Template] = {}


def register_template(
    name: str,
    format_user: Formatter,
    format_assistant: Formatter | None = None,
    format_system: Formatter | None = None,
    format_prefix: Formatter | None = None,
    default_system: str = "",
    stop_words: list[str] | None = None,
    thought_words: tuple[str, str] | None = None,
    efficient_eos: bool = False,
    replace_eos: bool = False,
    enable_thinking: Optional[bool] = True,
    mm_plugin: BasePlugin | None = None,
    template_class: type[Template] = Template,
) -> None:
    assistant_slots: SLOTS = ["{{content}}"] if efficient_eos else ["{{content}}", {"eos_token"}]
    TEMPLATES[name] = template_class(
        format_user=format_user,
        format_assistant=format_assistant or StringFormatter(assistant_slots),
        format_system=format_system or format_user,
        format_prefix=format_prefix or EmptyFormatter(),
        default_system=default_system,
        stop_words=stop_words or [],
        thought_words=thought_words or ("<think>\n", "\n</think>\n\n"),
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        enable_thinking=enable_thinking,
        mm_plugin=mm_plugin or get_mm_plugin("base"),
    )


def _chatml_template(
    name: str,
    plugin: BasePlugin,
    default_system: str = "",
    reasoning: bool = False,
    enable_thinking: Optional[bool] = True,
) -> None:
    register_template(
        name=name,
        format_user=StringFormatter(["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_assistant=StringFormatter(["{{content}}<|im_end|>\n"]),
        format_system=StringFormatter(["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        default_system=default_system,
        stop_words=["<|im_end|>"],
        replace_eos=True,
        enable_thinking=enable_thinking,
        mm_plugin=plugin,
        template_class=ReasoningTemplate if reasoning else Template,
    )


def get_template_and_fix_tokenizer(tokenizer: Any, data_args: Any) -> Template:
    name = data_args.template or "llava"
    if name not in TEMPLATES:
        raise ValueError(f"Kept templates are {sorted(TEMPLATES)}; got {name!r}.")
    template = deepcopy(TEMPLATES[name])
    if data_args.default_system is not None:
        template.default_system = data_args.default_system
    if isinstance(template, ReasoningTemplate):
        template.enable_thinking = data_args.enable_thinking
    template.fix_special_tokens(tokenizer)
    return template


_llava_system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
register_template(
    "llava",
    format_user=StringFormatter(["USER: {{content}} ASSISTANT:"]),
    default_system=_llava_system,
    mm_plugin=get_mm_plugin("llava", image_token="<image>"),
)

_chatml_template("qwen3vl", get_mm_plugin("qwen3vl", image_token="<|image_pad|>"), reasoning=True)
_chatml_template("qwen3vl_nothink", get_mm_plugin("qwen3vl", image_token="<|image_pad|>"), enable_thinking=False)
_chatml_template(
    "internvl",
    get_mm_plugin("internvl", image_token="<image>"),
    default_system="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
)
register_template(
    "kimivl",
    format_user=StringFormatter(["<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|>"]),
    format_assistant=StringFormatter(["{{content}}<|im_end|>"]),
    format_system=StringFormatter(["<|im_system|>system<|im_middle|>{{content}}<|im_end|>"]),
    default_system="You are a helpful assistant",
    stop_words=["<|im_end|>"],
    thought_words=("◁think▷", "◁/think▷"),
    mm_plugin=get_mm_plugin("kimivl", image_token="<|media_pad|>"),
    template_class=ReasoningTemplate,
)
