import numpy as np
import pandas as pd
import argparse
import pickle as pkl

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--output", help="output file")
parser.add_argument("--weight-pkl", help="pkl file for weights", default=None)
args = parser.parse_args()

# Initialize weights as new or from file
print("Initializing weights...")
try:
    with open(args.weight_pkl, 'rb') as f:
        weights = pkl.load(f)
except:
    raise ValueError("Unable to load weight PKL")

weights_list = []
for key, value in weights.items():
    try:
        weights_list.append((value.condition, value.w_0_bin, value.w_1_bin, value.w_10_cont, value.w_11_cont, value.w_20_cont, value.w_21_cont))
    except:
        try:
            weights_list.append((value.condition, value.w_0_bin, value.w_1_bin, value.w_0_cont, value.w_1_cont, np.nan, np.nan))
        except:
            try:
                tmp = value.c
                weights_list.append((value.condition, np.nan, np.nan, value.w_0, value.w_1, np.nan, np.nan))
            except:
                weights_list.append((value.condition, value.w_0, value.w_1, np.nan, np.nan, np.nan, np.nan))

pd.DataFrame(weights_list, columns=['condition', 'w_0_bin', 'w_1_bin', 'w_10_cont', 'w_11_cont', 'w_20_cont', 'w_21_cont']).to_csv(args.output, sep="\t", index=False)
