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

parser = argparse.ArgumentParser()
parser.add_argument("--in-dir", help="input directory")
parser.add_argument("--out-dir", help="output directory")
parser.add_argument("--metadata", help="metadata")
parser.add_argument("--condition", help="condition")
parser.add_argument("--alpha", help="learning rate", type=float, default = 0.1)
parser.add_argument("--w_0", help="w_0 initialization", type=float, default = -1)
parser.add_argument("--w_1", help="w_1 initialization", type=float, default = 3)
parser.add_argument("--epochs", help="epochs", type=int, default = 10000)
parser.add_argument("--threads", help="threads", type=int, default = 1)
parser.add_argument("--clean-fasta", help="run fasta cleaning", default = "Run")
parser.add_argument("--output-file", help="specific output file", default = "")
parser.add_argument("--stop-early", help="parameter difference at which to end", type=float, default = 0)
parser.add_argument("--pred", help="fasta file to test")
args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir
metadata = args.metadata
condition = args.condition
alpha = args.alpha
w_0 = args.w_0
w_1 = args.w_1
epochs = args.epochs
threads = args.threads
clean_fasta = args.clean_fasta
output_file = args.output_file
stop_early = args.stop_early
pred = args.pred

out_dir = "/" + out_dir.strip("/") + "/"
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

seq_dir = out_dir + "seqs/"
db_dir = out_dir + "db/"

if not os.path.isdir(seq_dir):
    os.makedirs(seq_dir)

if not os.path.isdir(db_dir):
    os.makedirs(db_dir)

# Get all fasta files
all_fastas_in = [str(f) for f in Path(in_dir).rglob('*.fasta')]

# Select just fasta files with non-NA values in the metadata
df = pd.read_csv(metadata, sep='\t')
df = df[df[condition].notna()]
tmp_dict = dict(zip(all_fastas_in, [fasta.split("/")[-1].split(".")[0] for fasta in all_fastas_in]))
all_fasta_names = set([fasta.split("/")[-1].split(".")[0] for fasta in all_fastas_in]).intersection(df['ID'].tolist())
all_fastas = [key for key, value in tmp_dict.items() if value in all_fasta_names]

def check_and_copy_single_fasta(fasta):
    for record in SeqIO.parse(fasta,'fasta'):
        # Only keep sequences longer than 20 amino acids
        if len(record.seq) > 20 and not os.path.exists(seq_dir + record.id.replace(":", "_") + ".fasta"):
            record.name = record.name.replace(":", "_")
            record.id = record.id.replace(":", "_")
            record.description = record.description.replace(":", "_")
            SeqIO.write(record, seq_dir + record.id.replace(":", "_") + ".fasta", "fasta")

# Copy non-zero fastas
def copy_nonzero_fasta(all_fastas, threads):
    with mp.Pool(processes = threads) as p:
        fastas = p.map(check_and_copy_single_fasta, all_fastas)

if clean_fasta == "Run":
    print("Cleaning fastas")
    copy_nonzero_fasta(all_fastas, threads)
else:
    print("Skipping cleaning")

# Create one row in the metadata for every fasta file
copied_fastas = [str(f) for f in Path(seq_dir).glob('*.fasta')]
seq_to_id = pd.DataFrame(
    {'seq_ID': copied_fastas,
     'ID': [fasta.split("/")[-1].split("_")[0] for fasta in copied_fastas]
    })
df = pd.merge(seq_to_id, df, on=['ID'], how='left')

# Remove duplicated conditions + amino acid sequence occurrences
if clean_fasta == "Run":
    check_for_delete = df[df['pdbx_details'].isin(df['pdbx_details'][df['pdbx_details'].duplicated()])]
    check_for_delete = check_for_delete[['seq_ID', 'pdbx_details']]

    def read_single_line_fasta(fasta):
        parsed = list(SeqIO.parse(fasta, 'fasta'))
        if len(parsed) > 0:
            return parsed[0].seq

    print("Reading fastas to ensure no duplications")
    with mp.Pool(processes = threads) as p:
        check_for_delete['seq'] = p.map(read_single_line_fasta, check_for_delete['seq_ID'].tolist())

    duplicated_prots = check_for_delete.loc[check_for_delete.duplicated(subset=['pdbx_details', 'seq'])]['seq_ID']
    del(check_for_delete)
    df = df[~df['seq_ID'].isin(duplicated_prots)]

    def delete_file(fasta):
        if os.path.exists(fasta):
            os.remove(fasta)

    print("Removing duplicated proteins")
    with mp.Pool(processes = threads) as p:
        p.map(delete_file, duplicated_prots.tolist())
else:
    print("Skipping duplicate deletion")

# Build a mash index from that file
if not os.path.exists(db_dir + 'combined_sketch.msh'):
    print("Building mash database")

    copied_fastas = [str(f) for f in Path(seq_dir).glob('*.fasta')]

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield (lst[i:i + n], i)

    fasta_chunks = list(chunks(copied_fastas, 1000))

    def sketch_chunk(chunk, i):
        subprocess.run(['mash sketch -o ' + db_dir + 'sketch_' + str(i) + ' ' + ' '.join(chunk) + ' -a -k 9 -s 5000'], shell=True)

    with mp.Pool(processes = threads) as p:
        p.starmap(sketch_chunk, fasta_chunks)

    subprocess.run(['mash paste ' + db_dir + 'combined_sketch ' + db_dir + 'sketch_*'], shell=True)

# Get a dictionary of conditions by sample
condition_dict = dict(zip([seq_id.split("/")[-1].split(".")[0] for seq_id in df['seq_ID'].tolist()], [float(item) for item in df[condition].tolist()]))

# Should be IDs, not filepaths now
df['seq_ID'] = [fasta.split("/")[-1].split(".")[0] for fasta in df['seq_ID']]

def sigmoid_array(x):
   return 1 / (1 + np.exp(-x))

def train(ID, condition_dict, w_0, w_1, alpha, mean_block, n):
    result = subprocess.run(['mash dist -v ' + str(1/n) + ' ' + db_dir + 'combined_sketch.msh ' + seq_dir + ID + '.fasta'], stdout=subprocess.PIPE, shell=True).stdout.decode("utf-8")

    if result == "":
        ni = 0
    else:
        IDs, ss = zip(*[map(item.split("\t").__getitem__, [0,2]) for item in result.split("\n")[:-1]])
        IDs, ss = [ID.split('/')[-1].split('.')[0] for ID in IDs], list(ss)

        # Remove self match
        with suppress(ValueError):
            index_to_remove = IDs.index(ID)
            del(IDs[index_to_remove])
            del(ss[index_to_remove])

        ni = len(IDs)

    y = condition_dict[ID]
    if ni > 0:
        ss = 1 - np.array(ss, dtype = 'f')
        xs = np.array([condition_dict[item] for item in IDs])
        sigmoids = sigmoid_array(w_1 * ss + w_0)
        sigminus = np.ones(ni) - sigmoids
        sigmoids_x = np.multiply(xs, sigmoids)
        sigmoids_x_sigminus = np.multiply(sigmoids_x, sigminus)
        sigmoids_sigminus = np.multiply(sigmoids, sigminus)
        sum_sigmoids = np.sum(sigmoids)
        sum_sigmoids_x = np.sum(sigmoids_x)
        yhat = 1/(ni + 1) * (mean_block - y/n) + ni/(ni + 1) * sum_sigmoids_x/sum_sigmoids
        first_chunk = (y/yhat - (1-y)/(1-yhat))
        dldw0 = -first_chunk * (sum_sigmoids * np.sum(sigmoids_x_sigminus) - sum_sigmoids_x * np.sum(sigmoids_sigminus)) / np.square(sum_sigmoids)
        dldw1 = -first_chunk * (sum_sigmoids * np.sum(np.multiply(sigmoids_x_sigminus, ss)) - sum_sigmoids_x * np.sum(np.multiply(sigmoids_sigminus, ss))) / np.square(sum_sigmoids)
        loss = y * np.log(yhat) + (1-y) * np.log(1-yhat)
        w_0 = w_0 - alpha * dldw0
        w_1 = w_1 - alpha * dldw1

    else:
        yhat = mean_block - y/n
        loss = y * np.log(yhat) + (1-y) * np.log(1-yhat)

    print("Y: " + str(y) + " Yhat: " + str(yhat))
    return loss, ni, w_0, w_1, y, yhat

epochs = 10000
w_0, w_1 = -1, 3
n = len(df.index) - 1
mean_block = 1/n * np.sum(df[condition].to_numpy())

print("Beginning training")

if output_file == "":
    output_file = "training.log"

with open(out_dir + output_file, 'w') as f:
    writer = csv.DictWriter(f, delimiter='\t', fieldnames = ['iteration', 'loss', 'n_p', 'w_0', 'w_1', 'ID', 'y', 'yhat'])
    writer.writerow({'iteration': 'iteration', 'loss': 'loss', 'n_p': 'n_p', 'w_0': 'w_0', 'w_1': 'w_1', 'ID':'ID', 'y':'y', 'yhat':'yhat'})

hist_results = np.array([[w_0, w_1]])

for i in range(epochs):
    ID_index = random.randint(0, n)
    loss, ni, w_0, w_1, y, yhat = train(df['seq_ID'][ID_index], condition_dict, w_0, w_1, alpha, mean_block, n)
    out_dict = {'iteration': i, 'loss': loss, 'n_p': ni, 'w_0': w_0, 'w_1': w_1, 'ID': df['seq_ID'][ID_index], 'y': y, 'yhat': yhat}
    with open(out_dir + output_file, 'a') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames = ['iteration', 'loss', 'n_p', 'w_0', 'w_1', 'ID', 'y', 'yhat'])
        writer.writerow(out_dict)

    hist_results = np.append(hist_results,[[w_0, w_1]], axis=0)
    if hist_results.shape[0] >= 1000:
        if np.all(np.max(hist_results[i-998:i+2,:], axis=0) - np.min(hist_results[i-998:i+2,:], axis=0) < args.stop_early):
            break
