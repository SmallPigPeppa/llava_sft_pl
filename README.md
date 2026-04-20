# llava_sft_pl_simplified

这是一个进一步精简后的 Lightning 多模态 LoRA SFT 项目。当前版本只保留 **train-only SFT** 主路径：本地 parquet 数据读取、ShareGPT/LLaVA 格式对齐、核心模板编码、多模态 collator、Lightning 训练循环和 LoRA 注入。

## 精简后的目录

```text
llava_sft_pl/
├── train.py
├── configs/demo2k.yaml
├── data/llava_779k_demo/dataset_info.json
├── llamafactory/
│   ├── data/
│   │   ├── collator.py
│   │   ├── lightning_datamodule.py
│   │   ├── mm_plugin.py
│   │   └── template.py
│   ├── extras/__init__.py
│   └── hparams/__init__.py
├── run_demo2k.sh
└── README.md
```

## 保留内容

- LLaVA / LLaVA-Next / LLaVA-Next-Video 模板与多模态插件；
- Qwen / Qwen2-VL / Qwen3-VL / Qwen3.5-VL / Qwen audio / Qwen omni 模板与插件；
- InternVL / InternS1 模板与插件；
- Kimi-VL 模板与插件；
- 本地 parquet + `dataset_info.json` 的 ShareGPT/LLaVA SFT 数据加载；
- online tokenization 与 offline tokenization；
- LoRA 注入、vision tower / multimodal projector 冻结、Lightning 训练、W&B logging、学习率调度。

## 删除/合并内容

- val/eval/test/predict dataloader 与相关参数；
- `data_utils.py`、`converter.py`、`parser.py`、`loader.py`、`processor/` 等拆得很碎的数据模块，已合并进 `data/lightning_datamodule.py`；
- `formatter.py` 与 `tool_utils.py`，已合并为 `data/template.py` 内的极简 formatter；
- `extras/constants.py`、`extras/logging.py`、`extras/misc.py`、`extras/packages.py`，已合并为 `extras/__init__.py`；
- `hparams/data_args.py`、`hparams/model_args.py`，已合并为 `hparams/__init__.py`；
- `.git`、`.idea`、`__pycache__`、`__MACOSX` 等与运行无关的文件。

## 运行 demo2k

确认 `data/llava_779k_demo/dataset_info.json` 里的 parquet 路径可访问。当前 `demo_2000` 指向：

```json
"file_name": "/ppio_net0/datasets/parquet/llava_779k_demo_2000"
```

启动：

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

## 数据配置

默认只保留 SFT 训练路径：

```yaml
data:
  dataset_dir: data/llava_779k_demo
  dataset: demo_2000
  template: llava
  stage: sft
  preprocessing_mode: online
  packing: false
```

`preprocessing_mode: online` 表示每个 epoch 在 DataLoader worker 中动态 tokenize，不额外落盘。若要离线 tokenize：

```bash
python train.py --config configs/demo2k.yaml \
  data.preprocessing_mode=offline \
  data.tokenized_path=data/tokenized/demo_2000_sft
```

## 模板选择

常用模板名：`llava`、`llava_next`、`llava_next_qwen`、`qwen`、`qwen2_vl`、`qwen3_vl`、`qwen3_vl_nothink`、`intern_vl`、`intern_s1`、`kimi_vl`。

## 保存策略

默认不保存 Lightning checkpoint：

```yaml
train:
  save_model_at_end: false
```

需要训练结束保存最终 LoRA adapter 和 processor 时，设置 `train.save_model_at_end=true`。
