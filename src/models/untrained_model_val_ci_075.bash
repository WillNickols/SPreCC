#!/bin/bash
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 36
#SBATCH -t 6:00:00
#SBATCH --mem 30000
#SBATCH -o /n/holyscratch01/huttenhower_lab/wnickols/cryst/untrained_model_val_ci_075.out
#SBATCH -e /n/holyscratch01/huttenhower_lab/wnickols/cryst/untrained_model_val_ci_075.err

python /n/holylfs05/LABS/nguyen_lab/Everyone/wnickols/cryst/src/test_parallel.py --in-dir /n/holyscratch01/huttenhower_lab/wnickols/cryst/seqs/ --out-dir /n/holyscratch01/huttenhower_lab/wnickols/cryst/untrained_model_val_ci_075/ --metadata /n/holyscratch01/huttenhower_lab/wnickols/cryst/metadata_parsed_100.tsv --encodings /n/holyscratch01/huttenhower_lab/wnickols/cryst/encodings_100.tsv --threads 36 --w-0-bin -1 --w-1-bin 3 --w-0-cont -1 --w-1-cont -2
