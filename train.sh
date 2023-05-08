#!/usr/bin/env bash

CONFIG=/home/oraclemiso/Restormer/Denoising/Options/Medical/GaussianGrayDenoising_Paired_Medical_Restormer.yml
python -m torch.distributed.launch --nproc_per_node=1 --master_port=14351 basicsr/train.py -opt $CONFIG --launcher pytorch