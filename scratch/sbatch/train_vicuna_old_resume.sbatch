#!/bin/bash
#SBATCH --job-name=train-vold-ultrachat
#SBATCH --partition=ava_m.p
#SBATCH --exclude=ava-m0,ava-m1,ava-m2,ava-m3,ava-m4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/sbatch/train_vicuna_old_resume.out

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Continues vicuna_old on UltraChat -- its actual original training data, per the
# ckpt-embedded ModelCheckpoint dirpath (.../checkpoints/vicuna-7b-ultrachat-lora-r128).
# Restores base weights from vicuna_old/last-old.ckpt (the preserved original, step
# 1,000,000) via train.yaml's restore_ckpt, and writes new checkpoints as last.ckpt
# in the same dir, leaving last-old.ckpt untouched.
#
# On first run this computes packing metadata for ~174k items (cached to
# vicuna_old/data_cache/), which takes a while before step 0.
#
# NOTE: after the first checkpoint is written, point train.yaml's restore_ckpt at
# last.ckpt (or delete the line to use Lightning's default "last") before
# re-submitting, or the resubmission restarts from the original base checkpoint.
# --local-ckpt-tmp: /tmp is only ~10GB here, far too small to stage a 17GB atomic
# checkpoint save; this stages temp files in the target dir instead.
uv run ptp_train /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old --local-ckpt-tmp
