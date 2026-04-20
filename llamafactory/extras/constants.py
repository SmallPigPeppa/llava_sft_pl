from __future__ import annotations

import os

AUDIO_PLACEHOLDER = os.getenv("AUDIO_PLACEHOLDER", "<audio>")
DATA_CONFIG = "dataset_info.json"
IGNORE_INDEX = -100
IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")
VIDEO_PLACEHOLDER = os.getenv("VIDEO_PLACEHOLDER", "<video>")

MROPE_MODELS = {
    "glm4v",
    "glm_ocr",
    "Keye",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen2_5_omni_thinker",
    "qwen3_omni_moe_thinker",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
}
