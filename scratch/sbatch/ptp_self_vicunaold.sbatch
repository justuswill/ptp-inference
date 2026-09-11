#!/bin/bash
#SBATCH --job-name=ptp-self-old
#SBATCH --partition=ava_m.p,ava_s.p
#SBATCH --exclude=ava-m0,ava-m1,ava-m2,ava-m3,ava-m4,ava-s0,ava-s1,ava-s2,ava-s3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/sbatch/ptp_self_vicunaold.out

# vicuna_old/train.yaml's ckpt_dir points back at .../checkpoints/vicuna, so an explicit
# absolute --checkpoint is required to actually load vicuna_old's own last.ckpt weights.
# (The existing stale-ptp_self dir under vicuna_old/inference was actually run against
# vicuna's checkpoint by mistake, before this was caught.)
uv run scripts/inference.py /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old --experiment vicuna --no-compile \
    --checkpoint /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old/last.ckpt \
    --algorithm ptp_self \
    --output-dir /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old/inference/ptp_self
