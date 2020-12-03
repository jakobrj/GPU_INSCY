from inscy import *

import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import csv
import matplotlib.pyplot as plt

def load_iris():
    X = normalize(torch.from_numpy(np.loadtxt("data/iris.data", delimiter=',', skiprows=0)).float())
    return X



test = -4

d = 15
c = 4
num_obj = 2
F = 1.
r = 1.
cl = 20
min_size = 1/(8*cl)
std = 5.
dims_pr_cl = 3

ns =  [8*1000, 16*1000, 32*1000, 64*1000, 128*1000, 256*1000, 512*1000, 1024*1000]
N_sizes = [(((num_obj*60)*cl/n)**(1/dims_pr_cl))*(std**(1/2))/200. for n in ns]
print(N_sizes)
n = ns[test]
N_size = N_sizes[test]

X = load_synt_gauss(n=n, d=d, cl=cl, std=std, cl_d=dims_pr_cl, re=0)
# X = load_synt(n=n, d=d, cl=cl, cl_d=dims_pr_cl, re=0)
n = X.shape[0]



t0 = time.time()
rs = GPU_INSCY_memory(X, N_size, F, num_obj, int(n * min_size), r, number_of_cells=c, rectangular=True)
print("GPU_INSCY_memory, took: %.4fs" % (time.time() - t0))


print(rs[0])

count = 0
for i in range(len(rs[1])):
    if len(rs[0][i]) == dims_pr_cl:
        C = rs[1][i]
        print(rs[0][i])
        print(np.sum(np.array(C)>=0))

        count += 1

#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(X[:, rs[0][i][0]], X[:, rs[0][i][1]], X[:, rs[0][i][2]], c=C)
#         plt.show()

print("found:", count, "/", cl)

exit()
