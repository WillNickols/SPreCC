#!/bin/bash
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 36
#SBATCH -t 6:00:00
#SBATCH --mem 30000
#SBATCH -o /n/holyscratch01/huttenhower_lab/wnickols/cryst/null_model_val.out
#SBATCH -e /n/holyscratch01/huttenhower_lab/wnickols/cryst/null_model_val.err

python /n/holylfs05/LABS/nguyen_lab/Everyone/wnickols/cryst/src/test_parallel.py --in-dir /n/holyscratch01/huttenhower_lab/wnickols/cryst/seqs/ --out-dir /n/holyscratch01/huttenhower_lab/wnickols/cryst/null_model_val/ --metadata /n/holyscratch01/huttenhower_lab/wnickols/cryst/metadata_parsed_100.tsv --encodings /n/holyscratch01/huttenhower_lab/wnickols/cryst/encodings_100.tsv --threads 36 --w-0-bin 0.001 --w-1-bin 0.001 --w-0-cont 0.001 --w-1-cont 0.001
