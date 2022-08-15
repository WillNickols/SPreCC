import multiprocessing as mp
import pandas as pd
from pathlib import Path
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--in-dir", help="seqs directory")
parser.add_argument("--output", help="output file")
parser.add_argument("--threads", help="number of threads", default=1, type=int)
args = parser.parse_args()

paths = Path(args.in_dir).rglob('*.fasta')

def parse_path(path):
    return (str(path).split("/")[-2], str(path).split("/")[-1].split("_")[0])

with mp.Pool(processes = args.threads) as p:
    parsed_paths = p.map(parse_path, paths)

pd.DataFrame(parsed_paths, columns=['partition', 'ID']).to_csv(args.output, sep="\t", index=False)
