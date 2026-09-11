#!/bin/bash
#SBATCH --job-name=collect-vold-choice2
#SBATCH --partition=ava_m.p,ava_s.p
#SBATCH --exclude=ava-m0,ava-m1,ava-m2,ava-m3,ava-m4,ava-s0,ava-s1,ava-s2,ava-s3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/collect_vicuna_old_choice2.log

uv run python scripts/collect_choicek_hist.py 2 vicuna_old
