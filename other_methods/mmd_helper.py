import numpy as np


alpha = np.loadtxt('alpha_hat_42_10000.txt')  # Hard-coded
beta = np.loadtxt('beta_hat_42_10000.txt')
K1 = np.loadtxt('../X1_sim.tsv')
K2 = np.loadtxt('../X2_sim.tsv')
m1 = np.matmul(K1, alpha)
m2 = np.matmul(K2, beta)
np.savetxt('P1.txt', m1)
np.savetxt('P2.txt', m2)
