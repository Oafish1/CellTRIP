import argparse

import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('fname1', type=str)
parser.add_argument('fname2', type=str)
parser.add_argument('--suffix', type=str)
parsed = parser.parse_args()

# Load data
alpha = np.loadtxt(parsed.fname1)
beta = np.loadtxt(parsed.fname2)
K1 = np.loadtxt('../X1_sim.tsv')  # Hard-coded
K2 = np.loadtxt('../X2_sim.tsv')

# Transform
m1 = np.matmul(K1, alpha)
m2 = np.matmul(K2, beta)

# Save projections
for i, proj in enumerate([m1, m2]):
    fname = f'P{i+1}'
    if parsed.suffix is not None: fname += f'_{parsed.suffix}'
    fname += '.txt'
    np.savetxt(fname, proj)
