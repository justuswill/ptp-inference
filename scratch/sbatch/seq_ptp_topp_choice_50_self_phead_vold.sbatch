#!/bin/bash
#SBATCH --job-name=choice50-self-phead-vold
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s4,ava-s5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/sbatch/seq_ptp_topp_choice_50_self_phead_vold.out

# branch_fit=vicuna_old_self_topp0.8 -- pooled (pi0,p,rho) fit from vicuna_old's own
# self-speculative + nucleus(0.8) collection (matching this run's own acceptance
# criterion), replacing the earlier stale-baseline run that used vicuna's constants.
uv run scripts/inference.py /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old --experiment vicuna --no-compile \
    --checkpoint /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old/last.ckpt \
    --algorithm seq-ptp-top-p-choice-k-self --choice-mode phead --ar-mode tok \
    --phead-checkpoint /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old/last_phead_ft_self_nt0.8.ckpt \
    --k 50 --p 0.8 --branch-fit vicuna_old_self_topp0.8
