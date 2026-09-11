#!/bin/bash
#SBATCH --job-name=seq-ptp-topp-choice-50-self-phead
#SBATCH --partition=ava_m.p
#SBATCH --nodelist=ava-m5,ava-m6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/sbatch/seq_ptp_topp_choice_50_self_phead.out

# Deliberately using the current (vicuna-derived, stale for vicuna_old) pooled
# _BRANCH_PI0/_BRANCH_RHO constants -- this run is the "stale baseline" to compare
# against once those constants are updated with vicuna_old's own fits.
uv run scripts/inference.py /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old --experiment vicuna --no-compile \
    --checkpoint /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old/last.ckpt \
    --algorithm seq-ptp-top-p-choice-k-self --choice-mode phead --ar-mode tok \
    --phead-checkpoint /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old/last_phead_ft_self_nt0.8.ckpt \
    --k 50 --p 0.8
