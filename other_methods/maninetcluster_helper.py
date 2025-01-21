import argparse
import os
import sys

import numpy as np

# Import
sys.path.insert(0, os.path.abspath('../../../ManiNetCluster/inst/python'))
import alignment
import correspondence
import neighborhood


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('fname1', type=str)
parser.add_argument('fname2', type=str)
parser.add_argument('-a', '--align', type=str, required=True)
parser.add_argument('-p', type=int, required=True)
parsed = parser.parse_args()

# Load data
X1, X2 = np.loadtxt(parsed.fname1), np.loadtxt('../X2.txt')
corr = correspondence.Correspondence(matrix=np.eye(X1.shape[0]))
W1, W2 = neighborhood.neighbor_graph(X1, k=5), neighborhood.neighbor_graph(X2, k=5)

# Project
if parsed.align == 'nlma':
    projection = alignment.manifold_nonlinear(X1, X2, corr, parsed.p, W1, W2)

elif parsed.align == 'lma':
    aln = alignment.ManifoldLinear(X1, X2, corr, parsed.p, W1, W2)
    projection = aln.project(X1, X2)

elif parsed.align == 'cca':
    aln = alignment.CCA(X1, X2, corr, parsed.p)
    projection = aln.project(X1, X2)

else:
    raise ValueError(f'Alignment type {parsed.align} not found.')

# Write to file
for i, proj in enumerate(projection):
    np.savetxt(f'P{i+1}.txt', proj)
