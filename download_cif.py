import multiprocessing as mp
from datetime import date
import sys
import pandas as pd
import urllib.request
import os
import numpy as np
import socket

socket.setdefaulttimeout(15)

# Usage: python download_cif.py input.csv output_folder threads > log.txt

input_file = sys.argv[1]
output_dir = "/" + str(sys.argv[2]).strip("/") + "/"
threads = int(sys.argv[3])

# Check how many CPUs are present
cpus = mp.cpu_count()
print("CPUs available: " + str(cpus) + "\nUsing: " + str(threads))

# Read in list of proteins
df = pd.read_csv(input_file, sep='\t', usecols=[0,1])
df.columns = ['id', 'date']
df['partition'] = np.where(
     df['date'].str.contains('2020|2021|2022'),
    'test',
     np.where(
        df['date'].str.contains('2018|2019'), 'validation', 'train'
     )
     )


# Download cif files
def f(pdb_id, partition):
    if not os.path.isfile(output_dir + partition + "/" + pdb_id + '.cif.gz'):
        try:
            urllib.request.urlretrieve('http://files.rcsb.org/download/' + pdb_id + '.cif.gz', output_dir + partition + "/" + pdb_id + '.cif.gz')
        except:
            try:
                urllib.request.urlretrieve('http://files.rcsb.org/download/' + pdb_id + '.cif.gz', output_dir + partition + "/" + pdb_id + '.cif.gz')
            except:
                pass

# Make a folder of each space group and download cif.gz files
for partition in ['train', 'validation', 'test']:
    print("Downloading proteins for: " + partition)

    if not os.path.exists(output_dir + partition + "/"):
      os.makedirs(output_dir + partition + "/")

    download_list = df[df['partition']==partition].loc[:, 'id'].values.tolist()
    for_query = [(id, partition) for id in download_list]

    # Run job
    with mp.Pool(processes = threads) as p:
          p.starmap(f, for_query)
