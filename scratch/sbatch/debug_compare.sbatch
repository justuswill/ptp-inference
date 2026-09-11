#!/bin/bash
#SBATCH --job-name=debug-compare
#SBATCH --partition=ava_m.p,ava_s.p
#SBATCH --exclude=ava-m0,ava-m1,ava-m2,ava-m3,ava-m4,ava-s0,ava-s1,ava-s2,ava-s3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/sbatch/debug_compare.out

uv run python scratch/debug_compare.py
