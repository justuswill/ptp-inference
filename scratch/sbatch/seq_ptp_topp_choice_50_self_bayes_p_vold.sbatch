#!/bin/bash
#SBATCH --job-name=choice50-self-bp-vold
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s4,ava-s5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/sbatch/seq_ptp_topp_choice_50_self_bayes_p_vold.out

# branch_fit=vicuna_old_self_topp0.8 -- pooled (pi0,rho) fit AND the per-question Beta
# prior (a,b) both come from vicuna_old's own self+nucleus(0.8) collection now
# (scratch/vicuna_old_topp0.8_self_joint_per_question_results.json, refit after the
# k=1000 _generate_blocked self-mode bug fix), not vicuna's.
uv run scripts/inference.py /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old --experiment vicuna --no-compile \
    --checkpoint /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old/last.ckpt \
    --algorithm seq-ptp-top-p-choice-k-self --choice-mode bayes-p --ar-mode tok \
    --k 50 --p 0.8 --branch-fit vicuna_old_self_topp0.8
