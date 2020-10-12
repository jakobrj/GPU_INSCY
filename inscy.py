from torch.utils.cpp_extension import load
import numpy as np
import torch
import time
import os

t0 = time.time()
print("Compiling our c++/cuda code, this usually takes 1-2 min. ")
inscy = load(name="GPU_INSCY",
             sources=["inscy_map.cpp",
                      "src/utils/util.cu",
                      "src/utils/TmpMalloc.cu",
                      "src/structures/SCY_tree.cpp",
                      "src/structures/GPU_SCY_tree.cu",
                      "src/algorithms/Clustering.cpp",
                      "src/algorithms/GPU_Clustering.cu",
                      "src/algorithms/INSCY.cpp",
                      "src/algorithms/GPU_INSCY.cu"
                      ])
print("Finished compilation, took: %.4fs" % (time.time() - t0))


def normalize(x):
    min_x = x.min(0, keepdim=True)[0]
    max_x = x.max(0, keepdim=True)[0]
    x_normed = (x - min_x) / (max_x - min_x)
    return x_normed

def load_vowel():
    return normalize(torch.from_numpy(np.loadtxt("data/vowel.dat", delimiter=',', skiprows=0)).float())

def load_glass():
    X = normalize(torch.from_numpy(np.loadtxt("data/glass.data", delimiter=',', skiprows=0)).float())
    X = X[:,1:-1].clone()
    return X

def load_pendigits():
    X = normalize(torch.from_numpy(np.loadtxt("data/pendigits.tra", delimiter=',', skiprows=0)).float())
    X = X[:,:-1].clone()
    return X


def INSCY(X, neighborhood_size, F, num_obj, min_size, r=1., number_of_cells=3, rectangular=False):
    return inscy.run_INSCY(X, neighborhood_size, F, num_obj, min_size, r, number_of_cells, rectangular)

def GPU_INSCY(X, neighborhood_size, F, num_obj, min_size, r=1., number_of_cells=3, rectangular=False):
    return inscy.run_GPU_INSCY(X, neighborhood_size, F, num_obj, min_size, r, number_of_cells, rectangular)

def GPU_INSCY_star(X, neighborhood_size, F, num_obj, min_size, r=1., number_of_cells=3, rectangular=False):
    return inscy.run_GPU_INSCY_star(X, neighborhood_size, F, num_obj, min_size, r, number_of_cells, rectangular)

def GPU_INSCY_memory(X, neighborhood_size, F, num_obj, min_size, r=1., number_of_cells=3, rectangular=False):
    return inscy.run_GPU_INSCY_memory(X, neighborhood_size, F, num_obj, min_size, r, number_of_cells, rectangular)

