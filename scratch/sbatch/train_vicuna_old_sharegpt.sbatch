#!/bin/bash
#SBATCH --job-name=train-vold-sharegpt
#SBATCH --partition=ava_m.p
#SBATCH --exclude=ava-m0,ava-m1,ava-m2,ava-m3,ava-m4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/sbatch/train_vicuna_old_sharegpt.out

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Continues the vicuna_old checkpoint on ShareGPT instead of its original UltraChat data
# (see train_vicuna_old_resume.sbatch for the faithful continuation). Restores the base
# weights from vicuna_old/last-old.ckpt via train.yaml's restore_ckpt, then saves its own
# checkpoints into checkpoints/vicuna_old_sharegpt/, leaving vicuna_old untouched.
# Data is the pregenerated ShareGPT teacher-completion cache (8 completions/prompt),
# symlinked from /extra/ucibdl1/fast-llm/ptp-data/vicuna-7b-sharegpt/data.
#
# NOTE: after the first checkpoint is written, switch train.yaml's `restore_ckpt` to
# last.ckpt (or delete the line so Lightning's default "last" applies) before
# re-submitting, otherwise a resubmission restarts from the original base checkpoint.
# --local-ckpt-tmp: /tmp is only ~10GB here, far too small to stage a 17GB atomic
# checkpoint save; this stages temp files in the target dir instead.
uv run ptp_train /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old_sharegpt --local-ckpt-tmp
