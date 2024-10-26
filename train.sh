#!/bin/bash
#SBATCH -p student # partition (queue).
#SBATCH --gres=gpu:01
#SBATCH --qos=normal
#SBATCH --job-name=data-create
#SBATCH --output=logs/results.txt
#SBATCH --error=logs/errors.txt

conda init bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate croco

torchrun --nproc_per_node=4 pretrain.py --output_dir ./output/pretraining/