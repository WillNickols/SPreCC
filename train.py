import sys
from Bio import SeqIO
import subprocess
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
import os
import random
from contextlib import suppress
import time
import csv
import math
from scipy.stats import norm
import argparse
import pickle as pkl
from weight_models import *
import timeit
import copy
import json

parser = argparse.ArgumentParser()
parser.add_argument("--in-dir", help="input directory")
parser.add_argument("--out-dir", help="output directory")
parser.add_argument("--metadata", help="metadata tsv")
parser.add_argument("--encodings", help="encodings tsv", default=None)
parser.add_argument("--use-prev-weights", help="Use previous weights from pkl file", default=False)
parser.add_argument("--weight-pkl", help="pkl file for weights", default=None)
parser.add_argument("--epochs", help="epochs", type=int, default = 10000)
parser.add_argument("--output-file", help="specific output file", default = "")
parser.add_argument("--stop-early", help="stop early if converged", type=float, default=0)
parser.add_argument("--threads", help="threads to train on", default = 4, type=int)
parser.add_argument("--alpha", help="alpha", default=0.1, type=float)
args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir
metadata = args.metadata
encodings = args.encodings
epochs = args.epochs
threads = args.threads
output_file = args.output_file
stop_early = args.stop_early

out_dir = "/" + out_dir.strip("/") + "/"
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

db_dir = out_dir + "db/"

if not os.path.isdir(db_dir):
    os.makedirs(db_dir)

# Read in crystallization conditions
df = pd.read_csv(metadata, sep='\t', low_memory=False)

# Create one row in the metadata for every fasta file
copied_fastas = [str(f) for f in Path(in_dir + "train/").glob('*.fasta')]
seq_to_id = pd.DataFrame(
    {'seq_ID': [fasta.split("/")[-1].split(".")[0] for fasta in copied_fastas],
     'ID': [fasta.split("/")[-1].split("_")[0] for fasta in copied_fastas]
    })
df = pd.merge(seq_to_id, df, on=['ID'], how='left')

# Build a mash sketch of all proteins
if not os.path.exists(db_dir + 'combined_sketch.msh'):
    print("Building mash database...")

    copied_fastas = [str(f) for f in Path(in_dir + "train/").glob('*.fasta')]

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield (lst[i:i + n], i)

    fasta_chunks = list(chunks(copied_fastas, 1000))

    def sketch_chunk(chunk, i):
        subprocess.run(['mash sketch -o ' + db_dir + 'sketch_' + str(i) + ' ' + ' '.join(chunk) + ' -a -k 9 -s 5000'], shell=True)

    with mp.Pool(processes = threads) as p:
        p.starmap(sketch_chunk, fasta_chunks)

    subprocess.run(['mash paste ' + db_dir + 'combined_sketch ' + db_dir + 'sketch_*'], shell=True)

print("Initializing weights...")
if args.use_prev_weights:
    try:
        with open(args.weight_pkl, 'rb') as f:
            weights = pkl.load(f)
    except:
        raise ValueError("Unable to load weight JSON")
else:
    if encodings is not None:
        encodings_tmp = pd.read_csv(encodings, sep='\t')
        input_list = [(condition, encoding) for condition, encoding in zip(encodings_tmp['name'].tolist(), encodings_tmp['encoding'].tolist()) if encoding is not np.nan]
        def load_encodings(condition, encoding):
            return (condition, get_encoding(encoding, condition, df))
        with mp.Pool(processes = threads) as p:
            weights = dict(p.starmap(load_encodings, input_list))
        del encodings_tmp
    else:
        raise ValueError("No encodings provided")

# Set alphas for multithreading
for key in weights:
    weights[key].set_alpha(args.alpha * threads)

# Make copies of the weights for parallel training
print("Starting deepcopy for parallelization...")
with mp.Pool(processes = threads) as p:
    weights_copy_list = p.map(copy.deepcopy, [weights for i in range(threads)])

def train_single(weights, ID, n):
    print(ID)
    result = subprocess.run(['mash dist -v ' + str(1/n) + ' ' + db_dir + 'combined_sketch.msh ' + in_dir + "train/" + ID + '.fasta'], stdout=subprocess.PIPE, shell=True).stdout.decode("utf-8")

    if result == "":
        n_p = 0
    else:
        IDs, ss = zip(*[map(item.split("\t").__getitem__, [0,2]) for item in result.split("\n")[:-1]])
        IDs, ss = [ID.split('/')[-1].split('.')[0] for ID in IDs], list(ss)

        # Remove self match
        with suppress(ValueError):
            index_to_remove = IDs.index(ID)
            del IDs[index_to_remove]
            del ss[index_to_remove]

        n_p = len(IDs)

    print(n_p)
    if n_p > 0:
        ss = 1 - np.array(ss, dtype = 'f')
        return [weights[key].update(ID, IDs, ss, n_p) for key,weight in weights.items()]
    else:
        return [None] * len(weights.items())

def update_parallel_weights():
    for i in range(threads):
        for key in weights:
            if isinstance(weights_copy_list[i][key], Cat):
                weights_copy_list[i][key].w_0 = weights[key].w_0
                weights_copy_list[i][key].w_1 = weights[key].w_1
            elif isinstance(weights_copy_list[i][key], Cont):
                weights_copy_list[i][key].w_0 = weights[key].w_0
                weights_copy_list[i][key].w_1 = weights[key].w_1
                weights_copy_list[i][key].c = weights[key].c
            elif isinstance(weights[key], Contbin):
                weights_copy_list[i][key].w_0_cont = weights[key].w_0_cont
                weights_copy_list[i][key].w_1_cont = weights[key].w_1_cont
                weights_copy_list[i][key].c = weights[key].c
                weights_copy_list[i][key].w_0_bin = weights[key].w_0_bin
                weights_copy_list[i][key].w_1_bin = weights[key].w_1_bin
            elif isinstance(weights[key], Bicontbin):
                weights_copy_list[i][key].w_10_cont = weights[key].w_10_cont
                weights_copy_list[i][key].w_11_cont = weights[key].w_11_cont
                weights_copy_list[i][key].c_1 = weights[key].c_1
                weights_copy_list[i][key].w_20_cont = weights[key].w_20_cont
                weights_copy_list[i][key].w_21_cont = weights[key].w_21_cont
                weights_copy_list[i][key].c_2 = weights[key].c_2
                weights_copy_list[i][key].w_0_bin = weights[key].w_0_bin
                weights_copy_list[i][key].w_1_bin = weights[key].w_1_bin

def train_parallel(IDs, n):
    for_parallel = [(weight_copy, ID, n) for weight_copy, ID in zip(weights_copy_list, IDs)]
    with mp.Pool(processes = threads) as p:
        loss_on_iteration = p.starmap(train_single, for_parallel)

    for key in weights:
        if isinstance(weights[key], Cat):
            weights[key].w_0 = np.mean([weight_copy[key].w_0 for weight_copy in weights_copy_list])
            weights[key].w_1 = np.mean([weight_copy[key].w_1 for weight_copy in weights_copy_list])
        elif isinstance(weights[key], Cont):
            weights[key].w_0 = np.mean([weight_copy[key].w_0 for weight_copy in weights_copy_list])
            weights[key].w_1 = np.mean([weight_copy[key].w_1 for weight_copy in weights_copy_list])
            weights[key].c = np.mean([weight_copy[key].c for weight_copy in weights_copy_list])
        elif isinstance(weights[key], Contbin):
            weights[key].w_0_cont = np.mean([weight_copy[key].w_0_cont for weight_copy in weights_copy_list])
            weights[key].w_1_cont = np.mean([weight_copy[key].w_1_cont for weight_copy in weights_copy_list])
            weights[key].c = np.mean([weight_copy[key].c for weight_copy in weights_copy_list])
            weights[key].w_0_bin = np.mean([weight_copy[key].w_0_bin for weight_copy in weights_copy_list])
            weights[key].w_1_bin = np.mean([weight_copy[key].w_1_bin for weight_copy in weights_copy_list])
        elif isinstance(weights[key], Bicontbin):
            weights[key].w_10_cont = np.mean([weight_copy[key].w_10_cont for weight_copy in weights_copy_list])
            weights[key].w_11_cont = np.mean([weight_copy[key].w_11_cont for weight_copy in weights_copy_list])
            weights[key].c_1 = np.mean([weight_copy[key].c_1 for weight_copy in weights_copy_list])
            weights[key].w_20_cont = np.mean([weight_copy[key].w_20_cont for weight_copy in weights_copy_list])
            weights[key].w_21_cont = np.mean([weight_copy[key].w_21_cont for weight_copy in weights_copy_list])
            weights[key].c_2 = np.mean([weight_copy[key].c_2 for weight_copy in weights_copy_list])
            weights[key].w_0_bin = np.mean([weight_copy[key].w_0_bin for weight_copy in weights_copy_list])
            weights[key].w_1_bin = np.mean([weight_copy[key].w_1_bin for weight_copy in weights_copy_list])

    update_parallel_weights()

    return loss_on_iteration

print("Beginning training...")

def view_weights():
    for _, value in weights.items():
        try:
            print(value.w_0)
        except:
            try:
                print(value.c)
            except:
                print(value.c_1)

if output_file == "":
    output_file = "training.log"

colnames = [key for key, _ in weights.items()]
colnames.append("ID")

with open(out_dir + output_file, 'w') as f:
    writer = csv.DictWriter(f, delimiter='\t', fieldnames = colnames)
    writer.writerow(dict(zip(colnames, colnames)))

n = len(df.index) - 1

if args.weight_pkl is None:
    weight_pkl = out_dir + "weights.pkl"
else:
    weight_pkl = args.weight_pkl

for i in range(epochs):
    IDs = df['seq_ID'][np.random.randint(low = 0, high = n + 1, size = threads)]
    loss = train_parallel(IDs, n)

    print("Iteration: " + str(i) + " ID: " + str(IDs))

    loss.append(IDs)

    #with open(out_dir + output_file, 'a') as f:
    #    writer = csv.DictWriter(f, delimiter='\t', fieldnames = colnames)
    #    writer.writerows(loss)

    if i % 2 == 0:
        with open(weight_pkl, "wb") as f:
            pkl.dump(weights, f)
        view_weights()
