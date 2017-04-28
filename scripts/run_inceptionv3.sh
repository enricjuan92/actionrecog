#!/bin/bash
srun --gres=gpu:1 --mem 12G python run_inceptionv3.py
