#!/bin/bash
#SBATCH --job-name=seq-ptp-topp-self-old-tok
#SBATCH --partition=ava_m.p,ava_s.p
#SBATCH --exclude=ava-m0,ava-m1,ava-m2,ava-m3,ava-m4,ava-s0,ava-s1,ava-s2,ava-s3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/sbatch/seq_ptp_topp_self_vicunaold_tok.out

# vicuna_old/train.yaml's ckpt_dir points back at .../checkpoints/vicuna, so an explicit
# absolute --checkpoint is required to actually load vicuna_old's own last.ckpt weights.
uv run scripts/inference.py /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old --experiment vicuna --no-compile \
    --checkpoint /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old/last.ckpt \
    --algorithm seq-ptp-top-p-self --ar-mode tok --p 0.99 \
    --output-dir /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old/inference/seq-ptp-topp-self-0.99-tok
