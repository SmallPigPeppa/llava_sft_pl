#!/usr/bin/env bash
python train.py --config configs/demo2k.yaml 2>&1 | tee train_llava_1_5_demo2k.log
