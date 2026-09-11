#!/bin/bash
#SBATCH --job-name=chead-ft-vold
#SBATCH --partition=ava_m.p
#SBATCH --nodelist=ava-m6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/sbatch/chead_finetune_vicuna_old.out

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run scripts/finetune.py /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old \
    --checkpoint /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old/last.ckpt \
    --head-type c --wandb-project ptp-inf --batch-size 2 --num-completions 64 --max-steps 8000 \
    --self-mode --nucleus-threshold 0.8
