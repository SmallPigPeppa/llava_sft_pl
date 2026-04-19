# minimal_llava_trainer_lightning

这是把原始最小版 LLaVA LoRA SFT 项目从 Hugging Face `Trainer` 改成 Lightning `Trainer` 后的版本。

保留内容：

1. ShareGPT/LLaVA 数据加载与图文 batch 组装逻辑；
2. LLaVA 模型加载、LoRA 注入、vision tower / projector 冻结逻辑；
3. `configs/demo2k.yaml` 与 `config.yaml` 的参数配置文件原样保留；
4. 命令行覆盖方式保持不变：`train.learning_rate=... data.max_samples=...`。

主要变化：

1. `train.py` 不再使用 `transformers.Trainer` / `TrainingArguments`；
2. 新增 `LlavaSFTLightningModule`，在 `training_step` / `validation_step` 中直接调用 HF LLaVA 模型并记录 loss；
3. 使用 Lightning `Trainer.fit(...)` 训练；
4. 使用 `torch.utils.data.DataLoader` 接入原有 `ShareGPTLlavaDataset` 和 `LlavaDataCollator`；
5. `train.report_to: [wandb]` 时使用 Lightning `WandbLogger`；
6. `save_checkpoint: false` / `save_strategy: "no"` 时禁用 Lightning checkpoint，与原配置一致；
7. `save_model_at_end: true` 时仍然使用 `save_pretrained` 保存最终 HF/PEFT 权重和 processor。

## 文件结构

```text
minimal_llava_trainer_lightning/
├── train.py              # LightningModule + Lightning Trainer 主训练入口
├── data.py               # dataset_info.json 读取、Parquet/JSON/HF 数据加载、ShareGPT->SFT 编码
├── configs/demo2k.yaml   # demo2k 的完整 YAML 配置，未改动
├── config.yaml           # 原配置，未改动
├── requirements.txt
├── run_demo2k.sh
└── README.md
```

## 安装

```bash
cd llava_sft_hf
pip install -r requirements.txt
```

## 运行 demo2k

确认 `configs/demo2k.yaml` 里的数据路径可访问：

```yaml
data:
  dataset_dir: data/llava_779k_demo
  dataset: demo_2000
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

## HF Trainer 参数到 Lightning 的映射

`configs/demo2k.yaml` 没有改字段名，`train.py` 内部做映射：

```yaml
train:
  per_device_train_batch_size -> DataLoader(batch_size=...)
  per_device_eval_batch_size  -> eval DataLoader(batch_size=...)
  gradient_accumulation_steps -> Trainer(accumulate_grad_batches=...)
  learning_rate               -> torch.optim.AdamW(lr=...)
  lr_scheduler_type           -> transformers.get_scheduler(...)
  warmup_ratio / warmup_steps -> scheduler warmup
  num_train_epochs            -> Trainer(max_epochs=...)
  max_steps                   -> Trainer(max_steps=...)
  bf16 / fp16                 -> Trainer(precision="bf16-mixed" / "16-mixed")
  logging_steps               -> Trainer(log_every_n_steps=...)
  eval_strategy/eval_steps    -> Trainer validation interval
  report_to: [wandb]          -> WandbLogger
  save_checkpoint: false      -> enable_checkpointing=False
```

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

## 数据格式要求

默认支持 ShareGPT/LLaVA 风格：

```json
{
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is in the image?"},
    {"from": "gpt", "value": "..."}
  ],
  "image": {"bytes": "...", "path": "xxx.jpg"}
}
```

`image` 也可以是 PIL image、bytes、路径字符串、`{"bytes": ..., "path": ...}`，或者它们的列表。路径类图片可通过 `data.media_dir` 指定根目录。

## 注意

`image_seq_len` 默认会从 processor/model 里推断。对 `llava-hf/llava-1.5-7b-hf` 通常是 576。如果你更换模型后遇到 image token 数与 image feature 数不匹配，可以在 YAML 中手动设置：

```yaml
data:
  image_seq_len: 576
```
