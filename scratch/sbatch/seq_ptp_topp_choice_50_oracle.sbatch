#!/bin/bash
#SBATCH --job-name=seq-ptp-topp-choice-50-oracle
#SBATCH --partition=ava_m.p,ava_s.p
#SBATCH --exclude=ava-m0,ava-m1,ava-m2,ava-m3,ava-m4,ava-s0,ava-s1,ava-s2,ava-s3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/sbatch/seq_ptp_topp_choice_50_oracle.out

# vicuna_old/train.yaml's ckpt_dir points back at .../checkpoints/vicuna, so an explicit
# absolute --checkpoint is required to actually load vicuna_old's own last.ckpt weights.
# --choice-mode oracle needs no sidecar checkpoint (just the precomputed per-question
# p_hat_i lookup in scratch/joint_per_question_results.json), so this is directly
# comparable to tmp_seq-ptp-topp-choice-50-0.98 / -bayes-conjugate / -bayes-p on this
# same vicuna_old checkpoint.
uv run scripts/inference.py /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old --experiment vicuna --no-compile \
    --checkpoint /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old/last.ckpt \
    --algorithm seq-ptp-top-p-choice-k --choice-mode oracle \
    --k 50 --p 0.98
