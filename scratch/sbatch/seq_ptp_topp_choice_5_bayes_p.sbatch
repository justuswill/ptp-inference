#!/bin/bash
#SBATCH --job-name=seq-ptp-topp-choice-5-bp
#SBATCH --partition=ava_m.p,ava_s.p
#SBATCH --exclude=ava-m0,ava-m1,ava-m2,ava-m3,ava-m4,ava-s0,ava-s1,ava-s2,ava-s3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/sbatch/seq_ptp_topp_choice_5_bayes_p.out

# NOTE: bayes-p's seed prior (_BAYES_P_FITS -> scratch/joint_per_question_results.json)
# is still fit from vicuna's own per-question data, not vicuna_old's -- same mismatch
# flagged on the oracle run. Rerun once vicuna_old's own joint_per_question_results
# lands (see stage1/stage2 analyze_branching_joint_*.py --prefix vicuna_old_).
uv run scripts/inference.py /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old --experiment vicuna --no-compile \
    --checkpoint /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old/last.ckpt \
    --algorithm seq-ptp-top-p-choice-k --choice-mode bayes-p \
    --k 5 --p 0.98
