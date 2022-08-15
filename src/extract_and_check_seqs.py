import sys
from Bio import SeqIO
import multiprocessing as mp
import pandas as pd
import warnings
from pathlib import Path
import gzip
import os
import itertools

# Usage: python extract_and_check_seqs.py in_dir output_dir metadata threads > log.txt

print(" ".join(sys.argv))

in_dir = str(sys.argv[1])
output_dir = str(sys.argv[2])
df = pd.read_csv(str(sys.argv[3]), sep='\t', low_memory=False)
threads = int(sys.argv[4])

output_dir = "/" + output_dir.strip("/") + "/"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# Check how many CPUs are present
cpus = mp.cpu_count()
print("CPUs available: " + str(cpus) + "\nUsing: " + str(threads))

# Get all CIF files
all_cifs = [str(f) for f in Path(in_dir).rglob('*.cif.gz')]

# Create fasta sequence from CIF
def fasta_from_cif(pdb_filename):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with gzip.open(pdb_filename, mode='rt') as pdb_file:
                partition = pdb_filename.split("/")[-2]
                if not os.path.isdir(output_dir + partition + "/"):
                    os.makedirs(output_dir + partition + "/")
                records = [record for record in SeqIO.parse(pdb_file, 'cif-atom')]
                record_lengths = [len(record.seq) for record in records]
                index_max = max(range(len(record_lengths)), key=record_lengths.__getitem__)
                record = records[index_max]
                # Only keep sequences longer than 20 amino acids
                if len(record.seq) > 20 and not os.path.exists(output_dir + partition + "/" + record.id.replace(":", "_") + ".fasta"):
                    record.name = record.name.replace(":", "_")
                    record.id = record.id.replace(":", "_")
                    record.description = record.description.replace(":", "_")
                    SeqIO.write(record, output_dir + partition + "/" + record.id.replace(":", "_") + ".fasta", "fasta")
        except:
            pass
    return

print("Writing fastas to " + output_dir)
with mp.Pool(processes = threads) as p:
  p.map(fasta_from_cif, all_cifs)

copied_fastas = [str(f) for f in Path(output_dir).rglob('*.fasta')]
seq_to_id = pd.DataFrame(
    {'seq_ID': copied_fastas,
     'ID': [fasta.split("/")[-1].split("_")[0] for fasta in copied_fastas]
    })
df = pd.merge(seq_to_id, df, on=['ID'], how='left')

# Remove duplicated conditions + amino acid sequence occurrences to avoid double counting the same study's identical proteins
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
