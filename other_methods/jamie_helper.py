import argparse
import os
import sys

import numpy as np

import jamie


parser = argparse.ArgumentParser()
parser.add_argument('fname1', type=str)
parser.add_argument('fname2', type=str)
parser.add_argument('-m', '--mask', type=str)
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('-t', '--target', type=int)
parser.add_argument('-p', type=int, required=True)
parser.add_argument('--suffix', type=str, required=True)
parsed = parser.parse_args()

# Load data
X1, X2 = np.loadtxt(parsed.fname1), np.loadtxt(parsed.fname2)
dataset = [X1, X2]
input_dataset = dataset

# Seed and random select
if parsed.mask is not None:
    train_mask = np.loadtxt(parsed.mask).astype(bool)
    input_dataset = [d[train_mask] for d in input_dataset]

# Project
jm = jamie.JAMIE(manual_seed=parsed.seed)
projection = jm.fit_transform(dataset=input_dataset)

# Impute if needed and save
if parsed.target is not None:
    projection = jm.modal_predict(dataset[2-parsed.target], 2-parsed.target)
    np.savetxt(f'I_{parsed.suffix}', projection)
else:
    for i, proj in enumerate(projection):
        np.savetxt(f'P{i+1}_{parsed.suffix}.txt', proj)
