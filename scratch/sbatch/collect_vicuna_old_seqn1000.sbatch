#!/bin/bash
#SBATCH --job-name=collect-vold-seqn1000
#SBATCH --partition=ava_m.p,ava_s.p
#SBATCH --exclude=ava-m0,ava-m1,ava-m2,ava-m3,ava-m4,ava-s0,ava-s1,ava-s2,ava-s3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --chdir=/home/jcwill/Projects/ptp
#SBATCH --output=/home/jcwill/Projects/ptp/scratch/collect_vicuna_old_seqn1000_full.log
#SBATCH --open-mode=append

# collect_seqn_choicek_hist.py resumes from this same log + its percall_full.json output
# if both already have partial data (see its resume_rows logic) -- so if 16h still isn't
# enough, just resubmit this same script again rather than raising the limit further.
uv run python scripts/collect_seqn_choicek_hist.py 1000 100 vicuna_old
