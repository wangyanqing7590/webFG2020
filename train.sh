#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONWARNINGS="ignore"

# resnet152, resnet50
export NET='resnet50'
export path='resnet50'
export data_base='/data1/wangyanqing/projects/webFG2020/data_5000'
export N_CLASSES=5000
export lr=0.0028
export w_decay=1e-8
export label_weight=0.5
export epochs=20
export batch_size=128
export step=1
export warmup=5
# export prt_model='/resnet50/resnet_30.pth'
# --denoise --smooth  --cos 

#python train.py --net ${NET}  --n_classes ${N_CLASSES} --path ${path} --batch_size=${batch_size} --data_base ${data_base} --epochs ${epochs} --lr ${lr} --w_decay ${w_decay} --label_weight ${label_weight} --step ${step} --warmup ${warmup}  --smooth  --cos 

#sleep 100

export lr=0.002
export w_decay=1e-5
export step=2


python train.py --net ${NET}  --n_classes ${N_CLASSES} --path ${path} --batch_size=${batch_size} --data_base ${data_base} --epochs ${epochs} --lr ${lr} --w_decay ${w_decay} --label_weight ${label_weight} --step ${step} --warmup ${warmup} --denoise --smooth  --cos
