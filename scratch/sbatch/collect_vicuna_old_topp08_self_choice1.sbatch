#!/bin/bash
#SBATCH --job-name=collect-vold-topp08-self-c1
#SBATCH --partition=ava_m.p,ava_s.p
#SBATCH --exclude=ava-m0,ava-m1,ava-m2,ava-m3,ava-m4,ava-s0,ava-s1,ava-s2,ava-s3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/collect_vicuna_old_topp0.8_self_choice1.log
#SBATCH --open-mode=append

uv run python scripts/collect_choicek_hist.py 1 vicuna_old 0.8 self
