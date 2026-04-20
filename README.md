# minimal_llava_trainer_lightning

这是一个 Lightning 版 LLaVA LoRA SFT 训练脚本。当前版本已经把原来项目里手写的 `ShareGPTLlavaDataset` / `LlavaDataCollator` 数据路径，切换为 `llamafactory_lightning_datamodule_portable`。

保留内容：

1. LLaVA 模型加载、LoRA 注入、vision tower / projector 冻结逻辑；
2. Lightning `Trainer`、W&B logging、checkpoint/save 配置映射；
3. `configs/demo2k.yaml` 仍然使用同一个 `dataset_info.json` 和同一个 parquet 数据源：`data/llava_779k_demo -> demo_2000`；
4. 命令行覆盖方式保持不变：`train.learning_rate=... data.max_samples=...`。

主要变化：

1. `train.py` 启动时自动把 `llamafactory_lightning_datamodule_portable/src` 放到 `sys.path` 前面；
2. SFT parquet 数据由 `LlamaFactoryLightningDataModule` 负责加载、ShareGPT 对齐、LLaVA template 编码、multimodal collate；
3. 默认使用 `data.preprocessing_mode: online`，不额外保存 tokenized dataset；
4. `llamafactory_lightning_datamodule_portable` 这个 portable slice 只提供 train dataloader，所以默认把 `data.val_size` 设为 `0.0`，并把 `train.eval_strategy` 设为 `"no"`；
5. 仍兼容原项目的数据容错：如果 parquet 行里有 image bytes/path，但第一轮 user 文本缺少 `<image>` placeholder，会在 ShareGPT converter 里自动补齐缺失的 `<image>`。

## 文件结构

```text
llava_sft_pl/
├── train.py
├── configs/demo2k.yaml
├── data.py                                      # 旧手写数据路径保留但 train.py 不再使用
├── data/llava_779k_demo/dataset_info.json       # 仍指向原来的 parquet
├── llamafactory_lightning_datamodule_portable/  # portable LlamaFactory DataModule slice
├── requirements.txt
├── run_demo2k.sh
└── README.md
```

## 安装

```bash
cd llava_sft_pl
pip install -r requirements.txt
```

`requirements.txt` 会递归安装 `llamafactory_lightning_datamodule_portable/requirements-datamodule.txt`，以匹配 portable DataModule 需要的 Transformers / Datasets / PEFT / TRL / OmegaConf 等依赖。

## 运行 demo2k

确认 `configs/demo2k.yaml` 里的数据路径可访问：

```yaml
data:
  dataset_dir: data/llava_779k_demo
  dataset: demo_2000
```

`data/llava_779k_demo/dataset_info.json` 中的 `demo_2000` 仍然指向：

```json
"file_name": "/ppio_net0/datasets/parquet/llava_779k_demo_2000"
```

启动：

```bash
export WANDB_API_KEY=你的_key
./run_demo2k.sh
```

也可以继续用原来的 YAML 覆盖方式：

```bash
python train.py --config configs/demo2k.yaml \
  train.learning_rate=1e-4 \
  data.max_samples=64 \
  train.num_train_epochs=1
```

## DataModule 配置

当前默认配置：

```yaml
data:
  stage: sft
  preprocessing_mode: online
  preprocessing_batch_size: 1000
  tokenized_path: null
  val_size: 0.0
  packing: false
```

如果要改成离线预处理，可以这样覆盖：

```bash
python train.py --config configs/demo2k.yaml \
  data.preprocessing_mode=offline \
  data.tokenized_path=data/tokenized/demo_2000_sft
```

离线模式第一次运行会把 tokenized train split 保存到 `data.tokenized_path`，之后会直接从该路径加载。

## W&B 与 checkpoint

当前默认配置仍然是：

```yaml
wandb_project: CL-debug
train:
  report_to: [wandb]
  save_checkpoint: false
  save_strategy: "no"
  save_model_at_end: false
```

这意味着训练日志会上报到 W&B；Lightning 不保存中间 checkpoint；训练结束也不额外保存 LoRA adapter。如果之后需要保存最终 LoRA 权重，把 `save_model_at_end` 改成 `true`。

## resume_from_checkpoint 注意

Lightning checkpoint 格式是 `.ckpt`。如果 `train.resume_from_checkpoint` 指向目录，脚本会自动在该目录下查找最近的 `.ckpt` 文件；如果该目录只有 Hugging Face Trainer 旧 checkpoint，则不能直接作为 Lightning checkpoint 恢复。
