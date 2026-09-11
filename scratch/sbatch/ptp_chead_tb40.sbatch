#!/bin/bash
#SBATCH --job-name=ptp-chead-tb40
#SBATCH --partition=ava_m.p,ava_s.p
#SBATCH --exclude=ava-m0,ava-m1,ava-m2,ava-m3,ava-m4,ava-s0,ava-s1,ava-s2,ava-s3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/sbatch/ptp_chead_tb40.out

uv run scripts/inference.py /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna --experiment vicuna --no-compile \
    --algorithm ptp --partial-mode chead --total-token-budget 40 \
    --save-dir ptp-chead-tb40 --chead-checkpoint /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna/last_chead_ft.ckpt
