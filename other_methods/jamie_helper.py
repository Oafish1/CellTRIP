import argparse
import os
import sys

import numpy as np

import jamie


parser = argparse.ArgumentParser()
parser.add_argument('fname1', type=str)
parser.add_argument('fname2', type=str)
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('-t', '--target', type=int)
parser.add_argument('-p', type=int, required=True)
parser.add_argument('--suffix', type=str)
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
    projection[parsed.target-1] = jm.modal_predict(dataset[2-parsed.target], 2-parsed.target)

# Write to file
for i, proj in enumerate(projection):
    fname = f'P{i+1}' if parsed.target is None else f'I{i+1}'
    if parsed.suffix is not None: fname += f'_{parsed.suffix}'
    fname += '.txt'
    if proj is not None: np.savetxt(fname, proj)
