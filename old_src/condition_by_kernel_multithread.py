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
parser.add_argument("--beta", help="regularization", type=float, default = 0)
parser.add_argument("--delta", help="range around y for loss", type=float, default = 0.25)
parser.add_argument("--w_0", help="w_0 initialization", type=float, default = -1.5)
parser.add_argument("--w_1", help="w_1 initialization", type=float, default = -5)
parser.add_argument("--c", help="c initialization", type=float, default = 2)
parser.add_argument("--epochs", help="epochs", type=int, default = 10000)
parser.add_argument("--threads", help="threads", type=int, default = 1)
parser.add_argument("--clean-fasta", help="run fasta cleaning", default = "Run")
parser.add_argument("--ci", help="confidence interval on 0 to 1", type=float, default = 0.95)
parser.add_argument("--output-file", help="specific output file", default = "")
parser.add_argument("--stop-early", help="stop early if converged", type=float, default=0)
parser.add_argument("--pred", help="fasta file to test")
parser.add_argument("--training-size", help="threads to train on", default = 4, type=int)
args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir
metadata = args.metadata
condition = args.condition
alpha = args.alpha
beta = args.beta
delta = args.delta
w_0 = args.w_0
w_1 = args.w_1
c = args.c
epochs = args.epochs
threads = args.threads
clean_fasta = args.clean_fasta
CI = args.ci
output_file = args.output_file
stop_early = args.stop_early
pred = args.pred
training_size = args.training_size

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

def std_norm_exp(x):
   return np.exp(-np.square(x)/2)

def zs(yhat, xs, hs):
    # Returns ndim(yhat) * ndim(xs)
    return np.divide(np.expand_dims(yhat,1) - xs, hs)

def train(ID, condition_dict, w_0, w_1, c, alpha, beta, n):
    result = subprocess.run(['mash dist -v ' + str(1/n) + ' ' + db_dir + 'combined_sketch.msh ' + seq_dir + ID + '.fasta'], stdout=subprocess.PIPE, shell=True).stdout.decode("utf-8")

    if result == "":
        n_p = 0
    else:
        IDs, ss = zip(*[map(item.split("\t").__getitem__, [0,2]) for item in result.split("\n")[:-1]])
        IDs, ss = [ID.split('/')[-1].split('.')[0] for ID in IDs], list(ss)

        # Remove self match
        with suppress(ValueError):
            index_to_remove = IDs.index(ID)
            del(IDs[index_to_remove])
            del(ss[index_to_remove])

        n_p = len(IDs)

    all_x = np.array([value for key, value in condition_dict.items() if key != ID])
    xbar = np.mean(all_x)
    eta = np.std(all_x)

    y = condition_dict[ID]
    if n_p > 0:
        ss = 1 - np.array(ss, dtype = 'f')
        xs = np.array([condition_dict[item] for item in IDs])
        sigmoids = sigmoid_array(w_1 * ss + w_0)

        U = (np.log(np.exp(-w_1) + np.exp(w_0)) - np.log(1 + np.exp(w_0))) / w_1 + 1
        hs = c * sigmoids / U

        norm_c = 1/(np.sqrt(2 * math.pi) * eta * (n_p + 1))

        def fyhat(yhat):
            std_norm_piece = std_norm_exp((yhat - xbar)/eta)
            std_phis = np.multiply(1/hs, std_norm_exp(zs(yhat, xs, hs)))
            return norm_c * std_norm_piece + 1/((n_p + 1) * np.sqrt(2 * math.pi)) * np.sum(std_phis, axis=1)

        int_space = np.arange(y-delta, y+delta * 101/100, delta / 100)

        loss = 1 - np.sum(fyhat(int_space)) * delta / 100
        mean = 1/(n_p + 1) * xbar + n_p/(n_p + 1) * np.mean(xs)
        mean_prob = fyhat([mean])[0]

        # Calculate gradients
        norm_d = 1/(c * (n_p + 1) * np.sqrt(2 * math.pi))
        m = np.exp(w_0) + np.exp(-w_1)

        dfydh = np.multiply(1 / np.square(hs), np.multiply(std_norm_exp(zs(int_space, xs, hs)), np.square(zs(int_space, xs, hs)) - 1)) # ndim(yhat) * ndim(xs)
        dhdw0 = np.add(c/U * np.multiply(sigmoids, 1-sigmoids), -np.multiply(c/np.multiply(w_1, np.square(U)) * sigmoids, np.exp(w_0) / m - (np.exp(w_0) / (1+np.exp(w_0))))) # ndim(xs)
        dhdw1 = np.add(np.multiply(ss, c/U * np.multiply(sigmoids, 1-sigmoids)), -np.multiply(c/np.square(np.multiply(w_1, U)) * sigmoids, -w_1 * np.exp(-w_1) / m - (np.log(m) - np.log(1 + np.exp(w_0))))) # ndim(xs)
        dhdc = hs/c # ndim(xs)

        dldw0 = - np.sum(norm_d * np.multiply(dfydh, dhdw0)) * delta / 100
        dldw1 = - np.sum(norm_d * np.multiply(dfydh, dhdw1)) * delta / 100
        dldc = - np.sum(norm_d * np.multiply(dfydh, dhdc)) * delta / 100 + 2 * (c - eta) * beta

        # Get mode
        for_search = np.append(xbar, xs)
        mode_start_search = min(for_search)
        mode_stop_search = max(for_search)
        search_dx = (mode_stop_search - mode_start_search)/1000
        search_space = np.arange(mode_start_search, mode_stop_search * (1 + search_dx), search_dx)
        searched = fyhat(search_space)
        mode = search_space[np.argmax(searched)]
        mode_prob = max(searched)

        # Get 95% confidence interval
        two_point_fiveth = norm.ppf(0.5 - CI/2)
        search_ph = max(min(all_x), min(two_point_fiveth * eta + xbar, min(two_point_fiveth*hs + xs)))
        while 1/(n_p + 1) * (norm.cdf((search_ph - xbar)/eta) + np.sum(norm.cdf(np.divide(search_ph - xs,hs)))) < 0.5 - CI/2:
            search_ph += search_dx
        lb = search_ph - search_dx
        ninety_seven_fiveth = norm.ppf(0.5 + CI/2)
        search_ph = min(max(all_x), max(ninety_seven_fiveth * eta + xbar, max(ninety_seven_fiveth*hs + xs)))
        while 1/(n_p + 1) * (norm.cdf((search_ph - xbar)/eta) + np.sum(norm.cdf(np.divide(search_ph - xs,hs)))) >= 0.5 + CI/2:
            search_ph -= search_dx
        ub = search_ph

        if y <= ub and y >= lb:
            capture = 1
        else:
            capture = 0

        w_0 = w_0 - alpha * dldw0
        w_1 = w_1 - alpha * dldw1
        c = c - alpha * dldc

    else:
        def fyhat(yhat):
            return 1/(np.sqrt(2 * math.pi) * eta) * std_norm_exp(zs(yhat, xbar, eta)[0])
        loss = 1 - np.sum(fyhat(np.arange(y-delta, y+delta * 101/100, delta / 100))) * delta / 100
        mean = xbar
        mode = xbar
        mean_prob = fyhat([xbar])[0]
        mode_prob = mean_prob
        lb = norm.ppf(0.5 - CI/2) * eta + xbar
        ub = norm.ppf(0.5 + CI/2) * eta + xbar
        if y <= ub and y >= lb:
            capture = 1
        else:
            capture = 0

    if loss > 1 or loss < 0:
        raise ValueError("Loss error: out of bounds")
    return round(loss,3), n_p, w_0, w_1, c, y, round(mean,3), round(mean_prob,3), round(mode,3), round(mode_prob,3), round(lb,3), round(ub,3), capture

def test(fasta, condition_dict, w_0, w_1, c, alpha, beta, n):
    result = subprocess.run(['mash dist -v ' + str(1/n) + ' ' + db_dir + 'combined_sketch.msh ' + fasta], stdout=subprocess.PIPE, shell=True).stdout.decode("utf-8")

    if result == "":
        n_p = 0
    else:
        IDs, ss = zip(*[map(item.split("\t").__getitem__, [0,2]) for item in result.split("\n")[:-1]])
        IDs, ss = [ID.split('/')[-1].split('.')[0] for ID in IDs], list(ss)

        n_p = len(IDs)

    all_x = np.array([value for key, value in condition_dict.items()])
    xbar = np.mean(all_x)
    eta = np.std(all_x)

    if n_p > 0:
        ss = 1 - np.array(ss, dtype = 'f')
        xs = np.array([condition_dict[item] for item in IDs])
        sigmoids = sigmoid_array(w_1 * ss + w_0)

        U = (np.log(np.exp(-w_1) + np.exp(w_0)) - np.log(1 + np.exp(w_0))) / w_1 + 1
        hs = c * sigmoids / U

        norm_c = 1/(np.sqrt(2 * math.pi) * eta * (n_p + 1))

        def fyhat(yhat):
            std_norm_piece = std_norm_exp((yhat - xbar)/eta)
            std_phis = np.multiply(1/hs, std_norm_exp(zs(yhat, xs, hs)))
            return norm_c * std_norm_piece + 1/((n_p + 1) * np.sqrt(2 * math.pi)) * np.sum(std_phis, axis=1)

        mean = 1/(n_p + 1) * xbar + n_p/(n_p + 1) * np.mean(xs)
        mean_prob = fyhat([mean])[0]

        # Get mode
        mode_start_search = min(all_x)
        mode_stop_search = max(all_x)
        search_dx = (mode_stop_search - mode_start_search)/1000
        search_space = np.arange(mode_start_search, mode_stop_search * (1 + search_dx), search_dx)
        searched = fyhat(search_space)
        mode = search_space[np.argmax(searched)]
        mode_prob = max(searched)

        # Get confidence interval
        two_point_fiveth = norm.ppf(0.5 - CI/2)
        search_ph = max(min(all_x), min(two_point_fiveth * eta + xbar, min(two_point_fiveth*hs + xs)))
        while 1/(n_p + 1) * (norm.cdf((search_ph - xbar)/eta) + np.sum(norm.cdf(np.divide(search_ph - xs,hs)))) < 0.5 - CI/2:
            search_ph += search_dx
        lb = search_ph - search_dx
        ninety_seven_fiveth = norm.ppf(0.5 + CI/2)
        search_ph = min(max(all_x), max(ninety_seven_fiveth * eta + xbar, max(ninety_seven_fiveth*hs + xs)))
        while 1/(n_p + 1) * (norm.cdf((search_ph - xbar)/eta) + np.sum(norm.cdf(np.divide(search_ph - xs,hs)))) >= 0.5 + CI/2:
            search_ph -= search_dx
        ub = search_ph

    else:
        def fyhat(yhat):
            return 1/(np.sqrt(2 * math.pi) * eta) * std_norm_exp(zs(yhat, xbar, eta)[0])
        mean = xbar
        mode = xbar
        mean_prob = fyhat([xbar])[0]
        mode_prob = mean_prob
        lb = norm.ppf(0.5 - CI/2) * eta + xbar
        ub = norm.ppf(0.5 + CI/2) * eta + xbar

    prediction_df = pd.DataFrame(
    {'pH': search_space,
     'probability': searched
    })

    prediction_df.to_csv(out_dir + output_file, sep='\t')

    return {'n_p':n_p, 'mean':mean, 'mean_prob':round(mean_prob,3), 'mode':round(mode,3), 'mode_prob':round(mode_prob,3), 'lb':round(lb,3), 'ub':round(ub,3), 'sim_prots': pd.DataFrame({'ID':IDs, 'seq_sim': ss, 'condition': xs})}

n = len(df.index) - 1

if pred:
    print(test(pred, condition_dict, w_0, w_1, c, alpha, beta, n))
    sys.exit()

print("Beginning training")

if output_file == "":
    output_file = "training.log"

with open(out_dir + output_file, 'w') as f:
    writer = csv.DictWriter(f, delimiter='\t', fieldnames = ['iteration', 'loss', 'n_p', 'w_0', 'w_1', 'c', 'IDs', 'interval', 'capture'])
    writer.writerow({'iteration': 'iteration', 'loss': 'loss', 'n_p': 'n_p', 'w_0': 'w_0', 'w_1': 'w_1', 'c':'c', 'IDs':'IDs', 'interval':'interval', 'capture':'capture'})


hist_results = np.array([[w_0, w_1, c]])

for i in range(epochs):
    ID_indexes = [ID_index for ID_index in np.random.randint(low = 0, high = n + 1, size = training_size)]
    input_tuples = [(df['seq_ID'][ID_index], condition_dict, w_0, w_1, c, alpha, beta, n) for ID_index in ID_indexes]

    with mp.Pool(processes = training_size) as p:
        loss, n_p, w_0, w_1, c, y, mean, mean_prob, mode, mode_prob, lb, ub, capture = np.mean(np.array(p.starmap(train, input_tuples)), axis = 0)

    loss = round(loss, 3)
    n_p = round(n_p, 3)
    interval = round(ub-lb, 3)
    capture = round(capture, 3)

    print("Loss: " + loss + " Range: " + str(interval) + " n_p: " + str(n_p))

    out_dict = {'iteration': i, 'loss': loss, 'n_p': n_p, 'w_0': w_0, 'w_1': w_1, 'c':c, 'IDs': ",".join([df['seq_ID'][ID_index] for ID_index in ID_indexes]), 'interval':interval, 'capture':capture}
    with open(out_dir + output_file, 'a') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames = ['iteration', 'loss', 'n_p', 'w_0', 'w_1', 'c', 'IDs', 'interval', 'capture'])
        writer.writerow(out_dict)

    hist_results = np.append(hist_results,[[w_0, w_1, c]], axis=0)
    if hist_results.shape[0] >= 1000:
        if np.all(np.max(hist_results[i-998:i+2,:], axis=0) - np.min(hist_results[i-998:i+2,:], axis=0) < args.stop_early):
            break
