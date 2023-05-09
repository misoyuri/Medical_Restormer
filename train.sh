#!/usr/bin/env bash

CONFIG=/content/drive/MyDrive/Restormer/Medical_Restormer/Denoising/Options/Medical/GaussianGrayDenoising_Paired_Medical_Restormer.yml
torchrun --nproc_per_node=1 --master_port=14351 basicsr/train.py -opt $CONFIG --launcher pytorch