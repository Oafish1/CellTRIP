import argparse
import os
import sys

import numpy as np

import jamie


parser = argparse.ArgumentParser()
parser.add_argument('fname1', type=str)
parser.add_argument('fname2', type=str)
parser.add_argument('-s', '--seed', type=str, default=42)
parser.add_argument('-t', '--target', type=str)
parser.add_argument('-p', type=int, required=True)
parsed = parser.parse_args()

# Load data
X1, X2 = np.loadtxt(parsed.fname1), np.loadtxt(parsed.fname2)
dataset = [X1, X2]

# Project
jm = jamie.JAMIE(manual_seed=parsed.seed)
projection = jm.fit_transform(dataset=dataset)

# Impute if needed
if parsed.target is not None:
    projection = [None for _ in range(2)]
    projection[parsed.target] = jm.modal_predict(dataset[-parsed.target+1], 1-parsed.target+1)

# Write to file
for i, proj in enumerate(projection):
    if proj is not None: np.savetxt(f'P{i+1}.txt', proj)
