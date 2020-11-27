from inscy import *
import os
import sys
import time
import matplotlib.pyplot as plt

def get_standard_params():
    d = 15
    n = 25000
    c = 4
    num_obj = 2
    F = 1.
    r = 1.
    min_size = 0.01
    N_size = 0.01
    cl = 20
    std = 5.
    dims_pr_cl = 12
    rounds = 3
    return n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, rounds

def get_run_file(experiment, method, n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, round):
    return "experiments_data/"+ experiment +"/" + method + \
              "n" + str(n) + "d" + str(d) + "c" + str(c) + \
              "N_size" + str(N_size) + "F" + str(F) + "r" + str(r) + \
              "num_obj" + str(num_obj) + "min_size" + str(min_size) + \
              "cl" + str(cl) + "std" + str(std) + "dims_pr_cl" + str(dims_pr_cl) + "round" + str(round) + ".npz"

def run(experiment, method, n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, round):

    run_file = get_run_file(experiment, method, n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, round)

    if not os.path.exists(run_file):
        X = load_synt_gauss(n=n, d=d, cl=cl, std=std, re=round, cl_d=dims_pr_cl)

        t0 = time.time()
        subspaces, clusterings = GPU_INSCY_memory(X, N_size, F, num_obj, int(n * min_size), r, number_of_cells=c, rectangular=True)
        t1 = time.time()
        running_time = t1-t0

        np.savez(run_file, running_time=running_time, subspaces=subspaces, clusterings=clusterings)

    else:
        data = np.load(run_file, allow_pickle=True)
        running_time = data["running_time"]
        subspaces = data["subspaces"]
        clusterings = data["clusterings"]

    return running_time, subspaces, clusterings

def plot(avg_running_times, xs, x_label, experiment):
    plt.plot(xs[:len(avg_running_times)], avg_running_times, label="GPU-INSCY", color="orange")
    plt.gcf().subplots_adjust(left=0.14)
    plt.legend(loc='upper left')
    plt.ylabel('time in seconds')
    plt.xlabel(x_label)
    #plt.ylim(0,900)
    plt.tight_layout()
    plt.savefig("plots/"+experiment+".pdf")
    #plt.show()
    plt.clf()


def run_diff_number_of_cl():
    n, d, c, N_size, F, r, num_obj, min_size, _, std, dims_pr_cl, rounds = get_standard_params()
    cls = [2, 4, 8, 16, 32, 64]

    print("running experiment: inc_cl")

    if not os.path.exists('experiments_data/inc_cl/'):
        os.makedirs('experiments_data/inc_cl/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    avg_running_times = []
    for cl in cls:
        print("cl:", cl)
        avg_running_time = 0.
        for round in range(rounds):
            running_time, subspaces, clusterings = run("inc_cl", "GPU_INSCY_memory", n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, round)
            avg_running_time += running_time

        avg_running_time /= rounds
        avg_running_times.append(avg_running_time)

    plot(avg_running_times, cls, "number of clusters", "inc_cl")



def run_diff_std():
    n, d, c, N_size, F, r, num_obj, min_size, cl, _, dims_pr_cl, rounds = get_standard_params()
    stds = [1.*i for i in range(3,10+1)]

    print("running experiment: inc_std")

    if not os.path.exists('experiments_data/inc_std/'):
        os.makedirs('experiments_data/inc_std/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    avg_running_times = []
    for std in stds:
        print("std:", std)
        avg_running_time = 0.
        for round in range(rounds):
            running_time, subspaces, clusterings = run("inc_std", "GPU_INSCY_memory", n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, round)
            avg_running_time += running_time

        avg_running_time /= rounds
        avg_running_times.append(avg_running_time)

    plot(avg_running_times, stds, "standard deviation", "inc_std")




def run_diff_dims_pr_cl():
    n, d, c, N_size, F, r, num_obj, min_size, cl, std, _, rounds = get_standard_params()
    dims_pr_cls = [2, 4, 6, 8, 10, 12]

    print("running experiment: inc_cl_d")

    if not os.path.exists('experiments_data/inc_cl_d/'):
        os.makedirs('experiments_data/inc_cl_d/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    avg_running_times = []
    for dims_pr_cl in dims_pr_cls:
        print("dims_pr_cl:", dims_pr_cl)
        avg_running_time = 0.
        for round in range(rounds):
            running_time, subspaces, clusterings = run("inc_cl_d", "GPU_INSCY_memory", n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, round)
            avg_running_time += running_time

        avg_running_time /= rounds
        avg_running_times.append(avg_running_time)

    plot(avg_running_times, dims_pr_cls, "number of dimensions per clusters", "inc_cl_d")



def run_diff_n():
    _, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, rounds = get_standard_params()
    ns =  [8*1000, 16*1000, 32*1000, 64*1000, 128*1000, 256*1000, 512*1000, 1024*1000]

    print("running experiment: inc_n_large")

    if not os.path.exists('experiments_data/inc_n_large/'):
        os.makedirs('experiments_data/inc_n_large/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    avg_running_times = []
    for n in ns:
        print("n:", n)
        avg_running_time = 0.
        for round in range(rounds):
            running_time, subspaces, clusterings = run("inc_n_large", "GPU_INSCY_memory", n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, round)
            avg_running_time += running_time

        avg_running_time /= rounds
        avg_running_times.append(avg_running_time)

    plot(avg_running_times, ns, "number of points", "inc_n_large")

def run_diff_d():
    n, _, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, rounds = get_standard_params()
    ds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    print("running experiment: inc_d_large")

    if not os.path.exists('experiments_data/inc_d_large/'):
        os.makedirs('experiments_data/inc_d_large/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    avg_running_times = []
    for d in ds:
        print("d:", d)
        avg_running_time = 0.
        for round in range(rounds):
            running_time, subspaces, clusterings = run("inc_d_large", "GPU_INSCY_memory", n, d, c, N_size, F, r, num_obj, min_size, cl, std, min(dims_pr_cl, d), round)
            avg_running_time += running_time

        avg_running_time /= rounds
        avg_running_times.append(avg_running_time)

    plot(avg_running_times, ds, "number of dimensions", "inc_d_large")


def run_diff_d_v2():
    n, _, c, N_size, F, r, num_obj, min_size, cl, std, _, rounds = get_standard_params()
    ds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    print("running experiment: inc_d_large_v2")

    if not os.path.exists('experiments_data/inc_d_large_v2/'):
        os.makedirs('experiments_data/inc_d_large_v2/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    avg_running_times = []
    for d in ds:
        print("d:", d)
        dims_pr_cl = int(d*0.8)
        avg_running_time = 0.
        for round in range(rounds):
            running_time, subspaces, clusterings = run("inc_d_large_v2", "GPU_INSCY_memory", n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, round)
            avg_running_time += running_time

        avg_running_time /= rounds
        avg_running_times.append(avg_running_time)

    plot(avg_running_times, ds, "number of dimensions", "inc_d_large_v2")


experiment = sys.argv[1]
if experiment == "inc_cl":
    run_diff_number_of_cl()
elif experiment == "inc_std":
    run_diff_std()
elif experiment == "inc_cl_d":
    run_diff_dims_pr_cl()
elif experiment == "inc_n_large":
    run_diff_n()
elif experiment == "inc_d_large":
    run_diff_d()
elif experiment == "inc_d_large_v2":
    run_diff_d_v2()
elif experiment == "all":
    run_diff_number_of_cl()
    run_diff_std()
    run_diff_dims_pr_cl()
    run_diff_n()
    run_diff_d()
    run_diff_d_v2()
