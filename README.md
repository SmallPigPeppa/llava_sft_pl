# llava_sft_pl_slim

这是一个进一步精简后的 Lightning 版 LLaVA LoRA SFT 项目。核心入口仍然是：

```bash
python train.py --config configs/demo2k.yaml
```

## 这次精简的重点

`llamafactory/` 被收缩成 SFT 训练所需的最小兼容层：

```text
llamafactory/
├── __init__.py
├── hparams/
│   └── __init__.py          # DataArguments / ModelArguments
└── data/
    ├── __init__.py
    ├── template.py          # LLaVA-family prompt template + image token expansion
    └── lightning_datamodule.py  # parquet + ShareGPT + online/offline tokenize + collator
```

相比上一版，删除了：

- `extras/`、`processor/`、`parser.py`、`converter.py`、`loader.py`、`collator.py`、`formatter.py`、`tool_utils.py`、`mm_plugin.py` 等拆得很碎的模块；
- RM / PPO / DPO / KTO / ranking / eval split / streaming / packing / tool-call / audio-video 预处理路径；
- `.git`、`.idea`、`__pycache__`、`__MACOSX` 等与运行无关的文件。

## 保留的能力

- LLaVA / LLaVA-NeXT 图像 SFT 模板与图像 token 展开；
- parquet 本地数据加载；
- `dataset_info.json` 数据集声明；
- ShareGPT 格式转换；
- online tokenization，即 DataLoader worker 中动态 tokenize；
- offline tokenization，即保存到 `data.tokenized_path`；
- multimodal collate：padding `input_ids` / `attention_mask` / `labels`，并用 Hugging Face processor 生成 `pixel_values` 等视觉输入；
- `python train.py --config ... key=value` 覆盖参数。

当前保留的模板名：

```text
llava, llava_next, llava_next_llama3, llava_next_mistral,
llava_next_qwen, llava_next_yi, yi_vl, vicuna, default, empty
```

另外保留了 `video_llava`、`llava_next_video*` 的模板名字用于兼容旧配置，但这个 slim 版本不再携带视频/音频预处理代码；如果数据列里真的包含视频或音频，会直接报出明确错误。

## 运行 demo2k

`configs/demo2k.yaml` 默认数据集：

```yaml
data:
  dataset_dir: data/llava_779k_demo
  dataset: demo_2000
  template: llava
  stage: sft
  preprocessing_mode: online
  val_size: 0.0
  packing: false
```

`data/llava_779k_demo/dataset_info.json` 中的 `file_name` 仍然指向本地 parquet 路径。确认路径可访问后运行：

```bash
export WANDB_API_KEY=你的_key
./run_demo2k.sh
```

覆盖参数示例：

```bash
python train.py --config configs/demo2k.yaml \
  data.max_samples=64 \
  train.learning_rate=1e-4 \
  train.num_train_epochs=1
```

## 注意

这个版本不是完整 LLaMA-Factory；它只保留本项目训练 LLaVA-family 图像 SFT 所需的代码。需要恢复 Qwen2-VL、MiniCPM-V、InternVL、audio/video 或 tool-call 等上游完整能力时，应使用完整版 LLaMA-Factory。
