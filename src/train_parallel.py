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

parser = argparse.ArgumentParser()
parser.add_argument("--in-dir", help="input directory")
parser.add_argument("--out-dir", help="output directory")
parser.add_argument("--metadata", help="metadata tsv")
parser.add_argument("--encodings", help="encodings tsv", default=None)
parser.add_argument("--use-prev-weights", help="Use previous weights from pkl file", action='store_true')
parser.add_argument("--weight-pkl", help="pkl file for weights", default=None)
parser.add_argument("--epochs", help="epochs", type=int, default = 10000)
parser.add_argument("--output-file", help="specific output file", default = "")
parser.add_argument("--stop-early", help="stop early if converged", type=float, default=0)
parser.add_argument("--threads", help="threads to train on", default = 4, type=int)
parser.add_argument("--alpha", help="alpha", default=0.1, type=float)
parser.add_argument("--beta-cont", help="regularization for continuous parameters", default=0, type=float)
parser.add_argument("--beta-bin", help="regularization for binary parameters", default=0, type=float)
parser.add_argument("--convergence-radius", help="distance between max and min in recent training history at which to end training early", default=0.1, type=float)
parser.add_argument("--convergence-length", help="number of training iterations over which to test the convergence radius", default=20000, type=int)
parser.add_argument("--seed", help="random seed", default=1, type=int)
parser.add_argument("--save-freq", help="save frequency", default=100, type=int)
args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir
metadata = args.metadata
encodings = args.encodings
epochs = args.epochs
threads = args.threads
output_file = args.output_file
stop_early = args.stop_early
seed = args.seed
save_freq = args.save_freq
alpha=args.alpha

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
        raise ValueError("Unable to load weight PKL")
else:
    if encodings is not None:
        encodings_tmp = pd.read_csv(encodings, sep='\t')
        input_list = [(condition, encoding) for condition, encoding in zip(encodings_tmp['name'].tolist(), encodings_tmp['encoding'].tolist()) if encoding is not np.nan]
        def load_encodings(condition, encoding):
            return (condition, get_encoding(encoding, condition, df, args.beta_cont, args.beta_bin, args.convergence_radius, args.convergence_length, alpha))
        with mp.Pool(processes = threads) as p:
            weights = dict(p.starmap(load_encodings, input_list))
        del encodings_tmp
    else:
        raise ValueError("No encodings provided")

def train_parallel(IDs, n, new_alpha):
    global weights
    input_list = [in_dir + "train/" + ID + '.fasta' for ID in IDs]
    result = subprocess.run(['mash dist -v ' + str(1/n) + ' ' + db_dir + 'combined_sketch.msh ' + ' '.join(input_list) + ' -p ' + str(threads)], stdout=subprocess.PIPE, shell=True).stdout.decode("utf-8")

    if result == "":
        n_p = 0
        return [], False
    else:
        IDs, searches, ss = zip(*[map(item.split("\t").__getitem__, [0,1,2]) for item in result.split("\n")[:-1]])
        IDs, searches, ss = [ID.split('/')[-1].split('.')[0] for ID in IDs], [ID.split('/')[-1].split('.')[0] for ID in searches], list(ss)

        all_loss = []
        all_done = True
        for item in set(searches):
            IDs_cur, ss_cur = zip(*[(ID, ss) for ID, search, ss in zip(IDs, searches, ss) if search == item])
            IDs_cur, ss_cur = list(IDs_cur), list(ss_cur)

            # Remove self match
            with suppress(ValueError):
                index_to_remove = IDs_cur.index(item)
                del IDs_cur[index_to_remove]
                del ss_cur[index_to_remove]

            n_p = len(IDs_cur)

            ss_cur = 1 - np.array(ss_cur, dtype = 'f')
            loss = []
            for key in weights:
                loss.append(weights[key].update(item, IDs_cur, ss_cur, n_p))
                weights[key].set_alpha(new_alpha)
                all_done = all_done and weights[key].done_updating

            all_loss.append(loss)
        return all_loss, all_done

print("Beginning training...")

def view_weights():
    for key, value in weights.items():
        try:
            print(key + ": w_0_bin:" + str(value.w_0_bin) + ", w_1_bin: " + str(value.w_1_bin) + ", w_10: " + str(value.w_10_cont) + ", w_11: " + str(value.w_11_cont) + ", c_1: " + str(value.c_1) + ", w_20: " + str(value.w_20_cont) + ", w_21: " + str(value.w_21_cont) + ", c_2: " + str(value.c_2) + ", eta_1: " + str(value.eta_1) + ", eta_2: " + str(value.eta_2))
        except:
            try:
                print(key + ": w_0_bin:" + str(value.w_0_bin) + ", w_1_bin: " + str(value.w_1_bin) + ", w_0_cont: " + str(value.w_0_cont) + ", w_1_cont: " + str(value.w_1_cont) + ", c: " + str(value.c) + ", eta: " + str(value.eta))
            except:
                try:
                    print(key + ": w_0:" + str(value.w_0) + ", w_1: " + str(value.w_1) + ", c: " + str(value.c) + ", eta: " + str(value.eta))
                except:
                    print(key + ": w_0:" + str(value.w_0) + ", w_1: " + str(value.w_1))

if output_file == "":
    output_file = "training.log"

colnames = [key for key, _ in weights.items()]
colnames.append("ID")

with open(out_dir + output_file, 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(colnames)

n = len(df.index) - 1

if args.weight_pkl is None:
    weight_pkl = out_dir + "weights.pkl"
else:
    weight_pkl = args.weight_pkl

np.random.seed(seed)
for i in range(epochs):
    rng = default_rng()
    numbers = rng.choice(n + 1, size = n + 1, replace = False)
    IDs = df['seq_ID'][numbers].tolist()

    batches = [IDs[i:i+threads] for i in range(0, len(IDs), 3)]
    for j, batch in enumerate(batches):
        new_alpha = ((1 - i / epochs) - (1 - j / len(batches))/epochs) * alpha
        all_loss, all_done = train_parallel(batch, n, new_alpha)
        print("Epoch: " + str(i) + " Iteration: " + str(j) + " IDs: " + " ".join(batch))

        if all_loss is not None:
            with open(out_dir + output_file, 'a') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(all_loss)

        if j % save_freq == 0:
            with open(weight_pkl, "wb") as f:
                pkl.dump(weights, f)
            view_weights()

        if all_done:
            print("Weights converged.  Saving weights and ending training...")
            with open(weight_pkl, "wb") as f:
                pkl.dump(weights, f)
            break
    else:
        continue
    break
