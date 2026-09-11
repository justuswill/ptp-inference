#!/bin/bash
#SBATCH --job-name=seq-ptp-self-vicuna-tok
#SBATCH --partition=ava_m.p,ava_s.p
#SBATCH --exclude=ava-m0,ava-m1,ava-m2,ava-m3,ava-m4,ava-s0,ava-s1,ava-s2,ava-s3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/sbatch/seq_ptp_self_vicuna_tok.out

uv run scripts/inference.py /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna --experiment vicuna --no-compile \
    --algorithm seq-ptp-self --ar-mode tok \
    --output-dir /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna/inference/seq-ptp-self-tok
