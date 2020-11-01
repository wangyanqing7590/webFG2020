#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5
export PYTHONWARNINGS="ignore"

# resnet152, resnet50
export NET='resnet50'
export path='resnet50'
export data_base='/data1/wangyanqing/projects/webFG2020/data_sub'
export N_CLASSES=1000
export lr=0.01
export w_decay=1e-5
export label_weight=0.5
export epochs=30

python train.py --net ${NET} --n_classes ${N_CLASSES} --path ${path} --data_base ${data_base} --epochs ${epochs} --lr ${lr} --w_decay ${w_decay} --label_weight ${label_weight} --denoise --smooth --warm 5 --cos
