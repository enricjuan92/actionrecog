#!/bin/bash
srun --gres=gpu:1 --mem 12G python run_vgg16.py
