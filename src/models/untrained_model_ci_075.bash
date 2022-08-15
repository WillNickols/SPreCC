#!/bin/bash
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 36
#SBATCH -t 24:00:00
#SBATCH --mem 30000
#SBATCH -o /n/holyscratch01/huttenhower_lab/wnickols/cryst/untrained_model_ci_075.out
#SBATCH -e /n/holyscratch01/huttenhower_lab/wnickols/cryst/untrained_model_ci_075.err

python /n/holylfs05/LABS/nguyen_lab/Everyone/wnickols/cryst/src/train_parallel.py --in-dir /n/holyscratch01/huttenhower_lab/wnickols/cryst/seqs/ --out-dir /n/holyscratch01/huttenhower_lab/wnickols/cryst/untrained_model_ci_075/ --metadata /n/holyscratch01/huttenhower_lab/wnickols/cryst/metadata_parsed_100.tsv --encodings /n/holyscratch01/huttenhower_lab/wnickols/cryst/encodings_100.tsv --weight-pkl /n/holyscratch01/huttenhower_lab/wnickols/cryst/untrained_model/untrained_model.pkl --threads 36 --convergence-radius 0.01 --convergence-length 10000 --save-freq 1000 --skip-weight-fitting --w-0-bin -1 --w-1-bin 3 --w-0-cont -1 --w-1-cont -2
