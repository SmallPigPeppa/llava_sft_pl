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
    FUNCTION = "function"
    OBSERVATION = "observation"


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

    def extract(self, content: str) -> str:
        return content


class EmptyFormatter(Formatter):
    pass


class StringFormatter(Formatter):
    pass


class FunctionFormatter(StringFormatter):
    def __init__(self, slots: SLOTS | None = None, tool_format: str | None = None) -> None:
        super().__init__(slots or ["{{content}}", {"eos_token"}])
        self.tool_format = tool_format


class ToolFormatter(Formatter):
    def __init__(self, slots: SLOTS | None = None, tool_format: str | None = None) -> None:
        super().__init__(slots or ["\n{{content}}"])
        self.tool_format = tool_format

    def apply(self, **kwargs: Any) -> SLOTS:
        content = kwargs.get("content")
        return super().apply(content=content) if content else []


@dataclass
class Template:
    format_user: Formatter
    format_assistant: Formatter
    format_system: Formatter
    format_function: Formatter
    format_observation: Formatter
    format_tools: Formatter
    format_prefix: Formatter
    default_system: str = ""
    stop_words: list[str] = field(default_factory=list)
    thought_words: tuple[str, str] = ("<think>\n", "\n</think>\n\n")
    tool_call_words: tuple[str, str] = ("<tool_call>", "</tool_call>")
    efficient_eos: bool = False
    replace_eos: bool = False
    replace_jinja_template: bool = False
    enable_thinking: Optional[bool] = True
    mm_plugin: BasePlugin = field(default_factory=lambda: get_mm_plugin("base"))

    def encode_oneturn(
        self,
        tokenizer: Any,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> tuple[list[int], list[int]]:
        encoded = self._encode(tokenizer, messages, system, tools)
        return sum(encoded[:-1], []), encoded[-1]

    def encode_multiturn(
        self,
        tokenizer: Any,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> list[tuple[list[int], list[int]]]:
        encoded = self._encode(tokenizer, messages, system, tools)
        return [(encoded[i], encoded[i + 1]) for i in range(0, len(encoded), 2)]

    def extract_tool(self, content: str) -> str:
        return self.format_tools.extract(content)

    def get_stop_token_ids(self, tokenizer: Any) -> list[int]:
        token_ids = {tokenizer.eos_token_id}
        for token in self.stop_words:
            token_ids.add(tokenizer.convert_tokens_to_ids(token))
        return list(token_ids)

    def add_thought(self, content: str = "") -> str:
        return f"{self.thought_words[0]}{self.thought_words[1]}" + content

    def remove_thought(self, content: str) -> str:
        pattern = re.compile(f"{re.escape(self.thought_words[0])}(.*?){re.escape(self.thought_words[1])}", re.DOTALL)
        return re.sub(pattern, "", content).lstrip("\n")

    def get_thought_word_ids(self, tokenizer: Any) -> list[int]:
        return tokenizer.encode(self.add_thought(), add_special_tokens=False)

    def _convert_elements_to_ids(self, tokenizer: Any, elements: SLOTS) -> list[int]:
        token_ids: list[int] = []
        for element in elements:
            if isinstance(element, str):
                if element:
                    token_ids.extend(tokenizer.encode(element, add_special_tokens=False))
            elif isinstance(element, set):
                if "bos_token" in element and tokenizer.bos_token_id is not None:
                    token_ids.append(tokenizer.bos_token_id)
                if "eos_token" in element and tokenizer.eos_token_id is not None:
                    token_ids.append(tokenizer.eos_token_id)
            elif isinstance(element, dict):
                token = element.get("token")
                if token:
                    token_ids.append(tokenizer.convert_tokens_to_ids(token))
            else:
                raise TypeError(f"Unsupported template slot: {element!r}")
        return token_ids

    def _encode(self, tokenizer: Any, messages: list[dict[str, str]], system: Optional[str], tools: Optional[str]) -> list[list[int]]:
        system = system if system is not None else self.default_system
        encoded_messages: list[list[int]] = []
        for i, message in enumerate(messages):
            elements: SLOTS = []
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    elements += self.format_system.apply(content=(system or "") + tool_text)

            role = message["role"]
            content = message.get("content", "")
            if role == Role.USER.value:
                elements += self.format_user.apply(content=content, idx=str(i // 2))
            elif role == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=content)
            elif role == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=content)
            elif role == Role.FUNCTION.value:
                elements += self.format_function.apply(content=content, thought_words=self.thought_words, tool_call_words=self.tool_call_words)
            else:
                raise ValueError(f"Unexpected message role: {role}")
            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))
        return encoded_messages

    def fix_special_tokens(self, tokenizer: Any) -> None:
        stop_words = list(self.stop_words)
        if self.replace_eos:
            if not stop_words:
                raise ValueError("Stop words are required to replace EOS.")
            self._add_or_replace_eos_token(tokenizer, stop_words.pop(0))
        if tokenizer.eos_token_id is None:
            self._add_or_replace_eos_token(tokenizer, "<|endoftext|>")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info_rank0(f"Add pad token: {tokenizer.pad_token}")
        if stop_words:
            try:
                tokenizer.add_special_tokens({"additional_special_tokens": stop_words}, replace_additional_special_tokens=False)
            except TypeError:
                tokenizer.add_special_tokens({"additional_special_tokens": stop_words})
            logger.info_rank0("Add stop words: %s", ",".join(stop_words))

    @staticmethod
    def _add_or_replace_eos_token(tokenizer: Any, eos_token: str) -> None:
        if tokenizer.eos_token == eos_token:
            return
        tokenizer.add_special_tokens({"eos_token": eos_token})
        logger.info_rank0(f"Set eos token: {tokenizer.eos_token}")

    def fix_jinja_template(self, tokenizer: Any) -> None:
        # Training only needs token ids built by this module. Keeping the tokenizer
        # chat_template untouched avoids carrying the large original Jinja generator.
        return None


@dataclass
class ReasoningTemplate(Template):
    def encode_oneturn(self, tokenizer: Any, messages: list[dict[str, str]], system: Optional[str] = None, tools: Optional[str] = None) -> tuple[list[int], list[int]]:
        messages = deepcopy(messages)
        if self.enable_thinking is False and len(messages) >= 2:
            messages[-1]["content"] = self.remove_thought(messages[-1]["content"])
        prompt_ids, response_ids = super().encode_oneturn(tokenizer, messages, system, tools)
        if len(messages) >= 2 and self.thought_words[0].strip() not in messages[-1]["content"] and self.thought_words[1].strip() not in messages[-1]["content"]:
            if not self.enable_thinking:
                prompt_ids += self.get_thought_word_ids(tokenizer)
            else:
                response_ids = self.get_thought_word_ids(tokenizer) + response_ids
        return prompt_ids, response_ids

    def encode_multiturn(self, tokenizer: Any, messages: list[dict[str, str]], system: Optional[str] = None, tools: Optional[str] = None) -> list[tuple[list[int], list[int]]]:
        messages = deepcopy(messages)
        if self.enable_thinking is False:
            for i in range(1, len(messages), 2):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])
        encoded = self._encode(tokenizer, messages, system, tools)
        for i in range(0, len(messages), 2):
            answer = messages[i + 1]["content"]
            if self.thought_words[0].strip() not in answer and self.thought_words[1].strip() not in answer:
                if not self.enable_thinking:
                    encoded[i] += self.get_thought_word_ids(tokenizer)
                else:
                    encoded[i + 1] = self.get_thought_word_ids(tokenizer) + encoded[i + 1]
        return [(encoded[i], encoded[i + 1]) for i in range(0, len(encoded), 2)]


TEMPLATES: dict[str, Template] = {}


def register_template(
    name: str,
    format_user: Formatter | None = None,
    format_assistant: Formatter | None = None,
    format_system: Formatter | None = None,
    format_function: Formatter | None = None,
    format_observation: Formatter | None = None,
    format_tools: Formatter | None = None,
    format_prefix: Formatter | None = None,
    default_system: str = "",
    stop_words: list[str] | None = None,
    thought_words: tuple[str, str] | None = None,
    tool_call_words: tuple[str, str] | None = None,
    efficient_eos: bool = False,
    replace_eos: bool = False,
    replace_jinja_template: bool = False,
    enable_thinking: Optional[bool] = True,
    mm_plugin: BasePlugin | None = None,
    template_class: type[Template] = Template,
) -> None:
    if name in TEMPLATES:
        raise ValueError(f"Template {name} already exists.")
    default_slots: SLOTS = ["{{content}}"] if efficient_eos else ["{{content}}", {"eos_token"}]
    assistant = format_assistant or StringFormatter(default_slots)
    user = format_user or StringFormatter(["{{content}}"])
    TEMPLATES[name] = template_class(
        format_user=user,
        format_assistant=assistant,
        format_system=format_system or user,
        format_function=format_function or FunctionFormatter(assistant.slots),
        format_observation=format_observation or user,
        format_tools=format_tools or ToolFormatter(),
        format_prefix=format_prefix or EmptyFormatter(),
        default_system=default_system,
        stop_words=stop_words or [],
        thought_words=thought_words or ("<think>\n", "\n</think>\n\n"),
        tool_call_words=tool_call_words or ("<tool_call>", "</tool_call>"),
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        replace_jinja_template=replace_jinja_template,
        enable_thinking=enable_thinking,
        mm_plugin=mm_plugin or get_mm_plugin("base"),
    )


def get_template_and_fix_tokenizer(tokenizer: Any, data_args: Any) -> Template:
    name = data_args.template or "empty"
    if name not in TEMPLATES:
        raise ValueError(f"Template {name!r} is not kept in this simplified build. Available: {sorted(TEMPLATES)}")
    template = deepcopy(TEMPLATES[name])
    if data_args.default_system is not None:
        template.default_system = data_args.default_system
    if data_args.tool_format is not None:
        logger.warning_rank0("tool_format=%s is treated as plain text in this train-only simplified build.", data_args.tool_format)
    if isinstance(template, ReasoningTemplate):
        template.enable_thinking = data_args.enable_thinking
    template.fix_special_tokens(tokenizer)
    template.fix_jinja_template(tokenizer)
    return template


def _chatml_plugin(name: str, plugin: BasePlugin | None = None, default_system: str = "You are a helpful assistant.", reasoning: bool = False) -> None:
    register_template(
        name=name,
        format_user=StringFormatter(["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_assistant=StringFormatter(["{{content}}<|im_end|>\n"]),
        format_system=StringFormatter(["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        default_system=default_system,
        stop_words=["<|im_end|>"],
        replace_eos=True,
        mm_plugin=plugin,
        template_class=ReasoningTemplate if reasoning else Template,
    )


register_template("empty", format_assistant=StringFormatter(["{{content}}"]))
register_template(
    "default",
    format_user=StringFormatter(["Human: {{content}}", {"eos_token"}, "\nAssistant:"]),
    format_assistant=StringFormatter(["{{content}}", {"eos_token"}, "\n"]),
    format_system=StringFormatter(["System: {{content}}", {"eos_token"}, "\n"]),
)

_llava_system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
register_template("llava", format_user=StringFormatter(["USER: {{content}} ASSISTANT:"]), default_system=_llava_system, mm_plugin=get_mm_plugin("llava", image_token="<image>"))
register_template("llava_next", format_user=StringFormatter(["USER: {{content}} ASSISTANT:"]), default_system=_llava_system, mm_plugin=get_mm_plugin("llava_next", image_token="<image>"))
register_template("llava_next_video", format_user=StringFormatter(["USER: {{content}} ASSISTANT:"]), default_system=_llava_system, mm_plugin=get_mm_plugin("llava_next_video", image_token="<image>", video_token="<video>"))
_chatml_plugin("llava_next_qwen", get_mm_plugin("llava_next", image_token="<image>"), default_system="You are a helpful assistant.")
register_template(
    "llava_next_llama3",
    format_user=StringFormatter(["<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"]),
    format_assistant=StringFormatter(["{{content}}<|eot_id|>"]),
    format_system=StringFormatter(["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_prefix=EmptyFormatter([{"bos_token"}]),
    stop_words=["<|eot_id|>", "<|eom_id|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin("llava_next", image_token="<image>"),
)
register_template(
    "llava_next_mistral",
    format_user=StringFormatter(["[INST] {{content}}[/INST]"]),
    format_assistant=StringFormatter([" {{content}}", {"eos_token"}]),
    format_system=StringFormatter(["{{content}}\n\n"]),
    format_prefix=EmptyFormatter([{"bos_token"}]),
    mm_plugin=get_mm_plugin("llava_next", image_token="<image>"),
)

_chatml_plugin("qwen", default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
_chatml_plugin("qwen3", reasoning=True)
_chatml_plugin("qwen3_nothink")
_chatml_plugin("qwen2_vl", get_mm_plugin("qwen2_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"))
_chatml_plugin("qwen3_vl", get_mm_plugin("qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"), default_system="", reasoning=True)
_chatml_plugin("qwen3_vl_nothink", get_mm_plugin("qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"), default_system="")
_chatml_plugin("qwen3_5", get_mm_plugin("qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"), default_system="", reasoning=True)
_chatml_plugin("qwen3_5_nothink", get_mm_plugin("qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"), default_system="")
_chatml_plugin("qwen2_audio", get_mm_plugin("qwen2_audio", audio_token="<|AUDIO|>"))
_chatml_plugin("qwen2_omni", get_mm_plugin("qwen2_omni", image_token="<|IMAGE|>", video_token="<|VIDEO|>", audio_token="<|AUDIO|>", vision_bos_token="<|vision_bos|>", vision_eos_token="<|vision_eos|>", audio_bos_token="<|audio_bos|>", audio_eos_token="<|audio_eos|>"))

_chatml_plugin(
    "intern_vl",
    get_mm_plugin("intern_vl", image_token="<image>", video_token="<video>"),
    default_system="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
)
_chatml_plugin("intern_s1", get_mm_plugin("intern_vl", image_token="<image>", video_token="<video>"), default_system="")

register_template(
    "kimi_vl",
    format_user=StringFormatter(["<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|>"]),
    format_assistant=StringFormatter(["{{content}}<|im_end|>"]),
    format_system=StringFormatter(["<|im_system|>system<|im_middle|>{{content}}<|im_end|>"]),
    default_system="You are a helpful assistant",
    stop_words=["<|im_end|>"],
    thought_words=("◁think▷", "◁/think▷"),
    mm_plugin=get_mm_plugin("kimi_vl", image_token="<|media_pad|>"),
    template_class=ReasoningTemplate,
)
