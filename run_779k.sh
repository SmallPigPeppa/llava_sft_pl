#!/usr/bin/env bash
python train.py --config configs/779k.yaml 2>&1 | tee train_llava_1_5_779k.log
