#!/bin/bash
#SBATCH --job-name=smoke-phead-choice-k
#SBATCH --partition=ava_m.p,ava_s.p
#SBATCH --exclude=ava-m0,ava-m1,ava-m2,ava-m3,ava-m4,ava-s0,ava-s1,ava-s2,ava-s3,ava-s5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/sbatch/smoke_phead_choice_k.out

# Smoke test for the new --choice-mode phead (seq-ptp-top-p-choice-k), reusing
# vicuna's existing phead sidecar (vicuna_old has no phead/chead sidecar yet).
uv run scripts/inference.py /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna --experiment vicuna --no-compile \
    --checkpoint /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna/last.ckpt \
    --algorithm seq-ptp-top-p-choice-k --choice-mode phead \
    --phead-checkpoint /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna/last_phead_ft.ckpt \
    --k 50 --p 0.98 --n-examples 3 \
    --output-dir /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna/inference/tmp_smoke_phead
