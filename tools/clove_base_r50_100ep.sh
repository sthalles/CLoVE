#!/bin/bash

set -e
set -x

data_dir="../../../../../../data/ImageNet2012/"
output_dir="./output/clove_base_r50_100ep"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    main_pretrain.py \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --cache-mode no \
    --crop 0.24 \
    --aug BYOL \
    --dataset ImageNet \
    --batch-size 256 \
    \
    --model PixPro \
    --arch resnet50 \
    --head-type early_return \
    \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 50 \
    --amp-opt-level O1 \
    \
    --save-freq 10 \
    --auto-resume \
    \
    --pixpro-p 2 \
    --pixpro-momentum 0.99 \
    --pixpro-pos-ratio 0.7 \
    --pixpro-transform-layer 1 \
    --pixpro-ins-loss-weight 0. \
    --lamb 2 \
    --mu 0.0 \
    --gamma 0.0 \
    --grad-accumulation-steps 4