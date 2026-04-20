# llava_sft_pl_simplified

这是一个精简后的 Lightning 版 LLaVA LoRA SFT 项目。数据读取、ShareGPT 对齐、LLaVA template 编码、multimodal collate 全部走内置的 LLaMA-Factory 数据路径；原来的手写 `data.py` / `ShareGPTLlavaDataset` / 自定义 collator 已删除。

## 精简后的目录

```text
llava_sft_pl_simplified/
├── train.py
├── configs/demo2k.yaml
├── data/llava_779k_demo/dataset_info.json
├── llamafactory/                 # 只保留训练所需的 LLaMA-Factory data/template/collator/hparams 子集
├── requirements.txt
├── run_demo2k.sh
├── LICENSE.llamafactory
└── README.md
```

## 保留内容

- LLaVA 模型加载；
- LoRA 注入，以及 vision tower / multimodal projector 冻结；
- Lightning 训练循环、W&B logging、学习率调度；
- LLaMA-Factory SFT 数据加载：本地 parquet、`dataset_info.json`、ShareGPT 转换、LLaVA multimodal collator；
- `python train.py --config ... key=value` 的覆盖方式。

## 删除内容

- 顶层 `data.py` 手写数据集和旧数据加载函数；
- `llamafactory_lightning_datamodule_portable/` 外层包装目录、examples、manifest、filelist；
- LLaMA-Factory 中本项目不使用的 RM/PPO/KTO/feedback/pairwise/unsupervised/pretrain 数据处理器；
- evaluation split、checkpoint callback、DeepSpeed/DDP 自定义策略等训练脚本分支；
- 多余的 hparams/parser/training/eval/generation 参数文件。

## 安装

```bash
cd llava_sft_pl_simplified
pip install -r requirements.txt
```

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

也可以覆盖参数：

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
  val_size: 0.0
  packing: false
```

`preprocessing_mode: online` 表示每个 epoch 在 DataLoader worker 中动态 tokenize，不额外落盘。若要离线 tokenize：

```bash
python train.py --config configs/demo2k.yaml \
  data.preprocessing_mode=offline \
  data.tokenized_path=data/tokenized/demo_2000_sft
```

## 保存策略

精简版默认不保存 Lightning checkpoint：

```yaml
train:
  save_model_at_end: false
```

如果需要训练结束后保存最终 LoRA adapter 和 processor，把 `train.save_model_at_end=true`。
