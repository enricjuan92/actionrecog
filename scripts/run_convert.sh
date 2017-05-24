#!/usr/bin/env bash

#python convert.py prototxt/cuhk_action_spatial_vgg_16_deploy.prototxt \
# --caffemodel caffemodel/vgg_16_action_rgb_pretrain.caffemodel \
# --data-output-path=spatial_vgg16.npy
srun --pty --gres=gpu:1 --mem 12G  python convert.py $1 --caffemodel $2 --data-output-path=$3