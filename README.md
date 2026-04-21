# llava_sft_pl_simplified

这是按“保留原工程骨架、只做减法”精简后的 Lightning 多模态 LoRA SFT 项目。

当前范围非常窄：**train-only SFT + image-only + online tokenize**。模型侧只保留 `qwen3_vl`、`llava`、`intern_vl`、`kimi_vl` 四类模板/插件；数据侧支持图文样本和纯文本样本，纯文本样本会在 collator 中自动补一张白色 fake image，并把对应 fake image token 的 `attention_mask` 置 0、`labels` 置 `-100`，避免训练到假图。

## 目录

```text
llava_sft_pl/
├── train.py
├── run_demo2k.sh
├── configs/demo2k.yaml
├── data/llava_779k_demo/dataset_info.json
├── llamafactory/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collator.py      # image-only collator，含纯文本 fake image 补齐
│   │   ├── datamodule.py    # parquet + ShareGPT 对齐 + online tokenize
│   │   ├── mm_plugin.py     # 仅 qwen3vl / llava / internvl / kimivl 图片插件
│   │   └── template.py      # 仅保留四类训练模板
│   ├── extras/__init__.py
│   └── hparams/__init__.py
└── README.md
```

## 已删除/收窄

- 删除 `.git`、`.idea`、`__pycache__`、`__MACOSX` 等运行无关文件。
- 删除推理、聊天、导出、评测相关路径。
- 删除离线分词落盘和样本打包路径。
- 删除非图片多模态数据处理逻辑。
- 删除无关模型插件与模板，只保留 `llava`、`qwen3_vl`、`intern_vl`、`kimi_vl` 及别名 `qwen3vl`、`internvl`、`kimivl`。
- 删除大量为全量 LLaMA-Factory 兼容而存在的分支、断言和注册表。

## 数据格式

`dataset_info.json` 仍使用 LLaMA-Factory 风格：

```json
{
  "demo_2000": {
    "file_name": "/path/to/parquet_or_dir",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "image"
    }
  }
}
```

图文样本：`images` 列可以是图片路径、图片路径列表、`{"path": ...}` 或带 bytes 的字典。样本文本里如果缺少 `<image>`，代码会自动把缺失的 `<image>` 插到第一轮 user 前面。

纯文本样本：可以不配置 `images` 列，或该列为空。tokenize 时不会插入 `<image>`；collator 会给该样本追加 fake image token，并把 fake token mask 掉。

## 运行

确认 `data/llava_779k_demo/dataset_info.json` 里的 parquet 路径可访问（当前示例为 `/ppio_net0/datasets/parquet/llava779k_demo2k`、`/ppio_net0/datasets/parquet/llava779k_demo10k`），然后：

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

## 模板选择

```yaml
data:
  template: llava      # llava / qwen3_vl / intern_vl / kimi_vl
```

别名也可用：`qwen3vl`、`internvl`、`kimivl`。

## 保存策略

默认不保存最终 adapter：

```yaml
train:
  save_model_at_end: false
```

需要训练结束保存 LoRA adapter 和 processor 时设置：

```bash
python train.py --config configs/demo2k.yaml train.save_model_at_end=true
```
