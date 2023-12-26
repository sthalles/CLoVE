#!/bin/bash

set -e
set -x

data_dir="/home/thalles.silva/representation-learning/datasets/ILSVRC2012/"
output_dir="./output/clove_base_r50_400ep"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    main_pretrain.py \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --cache-mode no \
    --crop 0.2 \
    --aug BYOL \
    --dataset ImageNet \
    --batch-size 32 \
    \
    --model CLoVE \
    --arch resnet50 \
    --head-type early_return \
    \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 2e-5 \
    --warmup-epoch 5 \
    --epochs 200 \
    \
    --save-freq 10 \
    --auto-resume \
    --clove-momentum 0.99 \
    --clove-pos-ratio 2 \
    --grad-accumulation-steps 2