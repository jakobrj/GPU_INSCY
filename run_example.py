from inscy import *

import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import csv
import matplotlib.pyplot as plt


def run(method, X):
    c = 4
    num_obj = 8
    F = 1.
    r = 1.
    min_size = 0.05
    N_size = 0.01
    n = X.shape[0]
    total = 0.
    for _ in range(3):
        t0 = time.time()
        method(X, N_size, F, num_obj, int(n * min_size), r, number_of_cells=c, rectangular=True)
        t1 = time.time()
        print(t1-t0)
        total += t1-t0
    avg = total/3.
    return avg

def runs(method):
    return [run(method, load_data()) for load_data in (load_glass,load_vowel)]

X = load_vowel()

c = 4
num_obj = 8
F = 1.
r = 1.
min_size = 0.05
N_size = 0.01
n = X.shape[0]

#do one run just to get the GPU started and get more correct measurements
GPU_INSCY(X, N_size, F, num_obj, int(n * min_size), r, number_of_cells=c, rectangular=True)


labels = ["glass", "vowel"]
ra = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8,5))
width = 0.20

rects1 = ax.bar(ra - 3*width/2, runs(INSCY), width=width, label="INSCY")
rects2 = ax.bar(ra - width/2, runs(GPU_INSCY), width=width, label="GPU-INSCY")
rects3 = ax.bar(ra + width/2, runs(GPU_INSCY_star), width=width, label="GPU-INSCY*")
rects4 = ax.bar(ra + 3*width/2, runs(GPU_INSCY_memory), width=width, label="GPU-INSCY-memory")

ax.set_xticks(ra)
ax.set_xticklabels(labels)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = round(rect.get_height(), 3)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1),
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
plt.ylabel('time in seconds')

ax.legend()
plt.rc('font', size=11)
plt.yscale("log")
fig.tight_layout()
plt.show()

