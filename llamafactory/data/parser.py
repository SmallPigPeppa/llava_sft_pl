from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from ..extras.constants import DATA_CONFIG


@dataclass
class DatasetAttr:
    """Local dataset_info.json entry used by the parquet-only DataModule."""

    load_from: Literal["file"]
    dataset_name: str
    formatting: Literal["sharegpt"] = "sharegpt"
    ranking: bool = False
    split: str = "train"
    num_samples: int | None = None

    system: str | None = None
    tools: str | None = None
    images: str | None = None
    videos: str | None = None
    audios: str | None = None

    messages: str | None = "conversations"
    role_tag: str | None = "from"
    content_tag: str | None = "value"
    user_tag: str | None = "human"
    assistant_tag: str | None = "gpt"
    observation_tag: str | None = "observation"
    function_tag: str | None = "function_call"
    system_tag: str | None = "system"

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: dict[str, Any], default: Any | None = None) -> None:
        setattr(self, key, obj.get(key, default))

    def join(self, attr: dict[str, Any]) -> None:
        self.set_attr("formatting", attr, default="sharegpt")
        if self.formatting != "sharegpt":
            raise ValueError("This simplified project keeps only ShareGPT/LLaVA SFT datasets.")
        self.set_attr("ranking", attr, default=False)
        if self.ranking:
            raise ValueError("Ranking/DPO-style datasets were removed; use standard ShareGPT SFT data.")
        self.set_attr("split", attr, default="train")
        self.set_attr("num_samples", attr)

        if "columns" in attr:
            for name in ["messages", "system", "tools", "images", "videos", "audios"]:
                self.set_attr(name, attr["columns"])

        if "tags" in attr:
            for name in [
                "role_tag",
                "content_tag",
                "user_tag",
                "assistant_tag",
                "observation_tag",
                "function_tag",
                "system_tag",
            ]:
                self.set_attr(name, attr["tags"])


def get_dataset_list(dataset_names: list[str] | None, dataset_dir: str | dict[str, Any]) -> list[DatasetAttr]:
    if dataset_names is None:
        return []

    if isinstance(dataset_dir, dict):
        dataset_info = dataset_dir
    else:
        config_path = os.path.join(dataset_dir, DATA_CONFIG)
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                dataset_info = json.load(f)
        except Exception as err:
            raise ValueError(f"Cannot open {config_path}: {err}") from err

    dataset_list: list[DatasetAttr] = []
    for name in dataset_names:
        if name not in dataset_info:
            raise ValueError(f"Undefined dataset {name} in {DATA_CONFIG}.")
        item = dataset_info[name]
        if "file_name" not in item:
            raise ValueError(f"Dataset {name} must define `file_name` in {DATA_CONFIG}.")
        attr = DatasetAttr("file", dataset_name=item["file_name"])
        attr.join(item)
        dataset_list.append(attr)
    return dataset_list
