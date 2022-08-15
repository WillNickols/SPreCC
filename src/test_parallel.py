import sys
from Bio import SeqIO
import subprocess
import multiprocessing as mp
import numpy as np
from numpy.random import default_rng
import pandas as pd
from pathlib import Path
import os
import random
from contextlib import suppress
import csv
import math
from scipy.stats import norm
import argparse
import pickle as pkl
from weight_models import *
import timeit
import copy

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--in-dir", help="input directory")
parser.add_argument("--out-dir", help="output directory")
parser.add_argument("--metadata", help="metadata tsv")
parser.add_argument("--encodings", help="encodings tsv", default=None)
parser.add_argument("--use-prev-weights", help="Use previous weights from pkl file", action='store_true')
parser.add_argument("--w-0-bin", help="w_0_bin", default=-1, type=float)
parser.add_argument("--w-1-bin", help="w_1_bin", default=3, type=float)
parser.add_argument("--w-0-cont", help="w_0_cont", default=-1, type=float)
parser.add_argument("--w-1-cont", help="w_1_cont", default=-2, type=float)
parser.add_argument("--weight-pkl", help="pkl file for weights", default=None)
parser.add_argument("--threads", help="threads to train on", default = 4, type=int)
parser.add_argument("--CI", help="confidence interval from 0 to 1", default=0.9, type=float)
args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir
metadata = args.metadata
encodings = args.encodings
threads = args.threads
CI = args.CI

out_dir = "/" + out_dir.strip("/") + "/"
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

db_dir = out_dir + "db/"

if not os.path.isdir(db_dir):
    os.makedirs(db_dir)

# Read in crystallization conditions
df = pd.read_csv(metadata, sep='\t', low_memory=False)

# Create one row in the metadata for every fasta train file
copied_fastas = [str(f) for f in Path(in_dir + "train/").glob('*.fasta')]
seq_to_id = pd.DataFrame(
    {'seq_ID': [fasta.split("/")[-1].split(".")[0] for fasta in copied_fastas],
     'ID': [fasta.split("/")[-1].split("_")[0] for fasta in copied_fastas]
    })
metadata_train = pd.merge(seq_to_id, df, on=['ID'], how='left')

copied_fastas = [str(f) for f in Path(in_dir + "validation/").glob('*.fasta')]
seq_to_id = pd.DataFrame(
    {'seq_ID': [fasta.split("/")[-1].split(".")[0] for fasta in copied_fastas],
     'ID': [fasta.split("/")[-1].split("_")[0] for fasta in copied_fastas]
    })
metadata_validation = pd.merge(seq_to_id, df, on=['ID'], how='left')
del(df)

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

# Initialize weights as new or from file
print("Initializing weights...")
if args.use_prev_weights:
    try:
        with open(args.weight_pkl, 'rb') as f:
            fit_weights = pkl.load(f)
            def convert_weights(condition, weight):
                return (condition, get_test_encoding(None, None, metadata_train, metadata_validation, None, None, None, None, weight))
            with mp.Pool(processes = threads) as p:
                weights = dict(p.starmap(convert_weights, fit_weights.items()))
    except:
        raise ValueError("Unable to load weight PKL")
else:
    if encodings is not None:
        encodings_tmp = pd.read_csv(encodings, sep='\t')
        input_list = [(condition, encoding) for condition, encoding in zip(encodings_tmp['name'].tolist(), encodings_tmp['encoding'].tolist()) if encoding is not np.nan]
        def load_weight(condition, encoding):
            return (condition, get_test_encoding(encoding, condition, metadata_train, metadata_validation, args.w_0_bin, args.w_1_bin, args.w_0_cont, args.w_1_cont, None))
        with mp.Pool(processes = threads) as p:
            weights = dict(p.starmap(load_weight, input_list))
        del encodings_tmp
    else:
        raise ValueError("No encodings provided")

# Function to train weights in parallel
def evaluate_parallel(IDs_org, n, CI):
    global weights
    global threads
    input_list = [in_dir + "validation/" + ID + '.fasta' for ID in IDs_org]
    start_time = timeit.default_timer()
    result = subprocess.run(['mash dist -v ' + str(1/n) + ' ' + db_dir + 'combined_sketch.msh ' + ' '.join(input_list) + ' -p ' + str(threads)], stdout=subprocess.PIPE, shell=True).stdout.decode("utf-8")
    print("Mash time: " + str(timeit.default_timer() - start_time))

    if result == "":
        n_p = 0
        all_vals_out = []
        for ID in IDs_org:
            vals_out = []
            for key in weights:
                vals_out.append(weights[key].evaluate(ID, [], [], n_p, CI))

            all_vals_out.append(vals_out)
        return all_vals_out

    else:
        start_time = timeit.default_timer()
        IDs, searches, ss = zip(*[map(item.split("\t").__getitem__, [0,1,2]) for item in result.split("\n")[:-1]])
        IDs, searches, ss = [ID.split('/')[-1].split('.')[0] for ID in IDs], [ID.split('/')[-1].split('.')[0] for ID in searches], list(ss)

        all_vals_out = []
        searches_set = set(searches)
        for item in IDs_org:
            if item in searches_set:
                IDs_cur, ss_cur = zip(*[(ID, ss) for ID, search, ss in zip(IDs, searches, ss) if search == item])
                IDs_cur, ss_cur = list(IDs_cur), list(ss_cur)

                if item in IDs_cur:
                    raise ValueError("Train set includes test point")

                n_p = len(IDs_cur)

                ss_cur = 1 - np.array(ss_cur, dtype = 'f')
                vals_out = []
                for key in weights:
                    vals_out.append(weights[key].evaluate(item, IDs_cur, ss_cur, n_p, CI))

                all_vals_out.append(vals_out)
            else:
                vals_out = []
                for key in weights:
                    vals_out.append(weights[key].evaluate(item, [], [], 0, CI))

                all_vals_out.append(vals_out)
        print("Calculate time: " + str(timeit.default_timer() - start_time))
        return all_vals_out

# Evalaute the model on all test proteins to get thresholds and accuracy
print("Beginning evaluation...")
final_pass_file = "evaluation.log"

colnames = [key for key, _ in weights.items()]
colnames.append("ID")

with open(out_dir + final_pass_file, 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(colnames)

n = len(metadata_train.index)

IDs = metadata_validation['seq_ID'].tolist()
batches = [IDs[i:i+threads] for i in range(0, len(IDs), threads)]
for j, batch in enumerate(batches):
    print("Step " + str(j) + " of " + str(len(batches)) + ", IDs: " + " ".join(batch))
    vals_out = evaluate_parallel(batch, n, CI)

    for batch_iter, ID in enumerate(batch):
        vals_out[batch_iter].append(ID)

    with open(out_dir + final_pass_file, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(vals_out)
