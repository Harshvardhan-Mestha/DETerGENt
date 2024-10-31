#!/bin/sh

#SBATCH --job-name=resnet_finetune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=./vision_FT/logs/%j.out
#SBATCH --error=./vision_FT/logs/%j.err
#SBATCH --time=05:00:00

# activate virtualenv
conda activate radio-lm

python -m torch.distributed.run --standalone --nproc-per-node=gpu --master-port=23456 vision_FT/trainer.py
