from pypdb import *
import multiprocessing as mp
from datetime import date
import csv
import sys
import urllib.request
import pandas as pd
import numpy as np
import traceback
import warnings

# Usage: python crystal_metadata.py output.tsv threads > log.txt

print(" ".join(sys.argv))

output = str(sys.argv[1])
threads = int(sys.argv[2])

# Check how many CPUs are present
cpus = mp.cpu_count()
print("CPUs available: " + str(cpus) + "\nUsing: " + str(threads))

# List all proteins
found_pdbs = Query("*").search()
print("Total proteins: " + str(len(found_pdbs)) + "\nData collected on " + date.today().strftime('%m/%d/%Y'))

# Get crystallization and space group information for all crystals
def f(pdb_id):
  all_info = get_info(pdb_id)
  if 'rcsb_accession_info' in all_info.keys() and 'exptl_crystal_grow' in all_info.keys():
        info = ([pdb_id, all_info['rcsb_accession_info']['deposit_date'].split("T")[0], all_info['exptl_crystal_grow'][0].get('crystal_id'), all_info['exptl_crystal_grow'][0].get('details'), all_info['exptl_crystal_grow'][0].get('method'), all_info['exptl_crystal_grow'][0].get('p_h'), all_info['exptl_crystal_grow'][0].get('pdbx_details'), all_info['exptl_crystal_grow'][0].get('pdbx_phrange'), all_info['exptl_crystal_grow'][0].get('temp'), all_info['exptl_crystal_grow'][0].get('temp_details')])

        # Get better-parsed data if available
        try:
          fp = urllib.request.urlopen("http://bmcd.ibbr.umd.edu/display/" + pdb_id)
          mystr = fp.read().decode("utf8")
          fp.close()
          lines = mystr.split("\n")
          lines = lines[[idx for idx, s in enumerate(lines) if 'Deposition Date:' in s][0]:[idx for idx, s in enumerate(lines) if '<div class="panel-heading">Polymers</div>' in s][0]]

          reshaped = np.array([item.split("<td>")[1].split("</td>")[0] for item in lines if '<td>' in item], dtype=object).reshape((-1, 6))
          df = pd.DataFrame(reshaped, columns=['Condition', 'Low', 'High', 'Units', 'pH 1', 'pH 2'])
          df[["Low", "High"]] = df[["Low", "High"]].apply(pd.to_numeric)
          with warnings.catch_warnings():
              warnings.simplefilter("ignore")
              df['val'] = np.nanmean((df['Low'], df['High']), axis=0)
          df = df[['Condition', 'val', 'Units']]
          aggregation_functions = {'val': 'sum', 'Units':'first', 'Condition': 'first'}
          df = df.groupby(df['Condition']).aggregate(aggregation_functions).astype(str)

          info[6] = " ".join([" ".join(item) for item in df.values.tolist() if item[0] != '0.0'])

          return info
        except Exception as exc:
            return info

# Run search
with mp.Pool(processes = threads) as p:
      metadata = p.map(f, found_pdbs)

# Save output to csv
with open(output, "w", newline="") as f:
    writer = csv.writer(f, delimiter = "\t")
    writer.writerows([item for item in metadata if item is not None])
