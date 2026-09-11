#!/bin/bash
#SBATCH --job-name=ptp-phead
#SBATCH --partition=ava_m.p
#SBATCH --nodelist=ava-m6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/sbatch/ptp_phead.out

uv run scripts/inference.py /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna --experiment vicuna --no-compile \
    --algorithm ptp --partial-mode phead \
    --phead-checkpoint /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna/last_phead_ft.ckpt
