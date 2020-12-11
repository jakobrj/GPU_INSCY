from inscy import *
import os
import sys
import time
import matplotlib.pyplot as plt

font_size = 20
dist_lim = 16.

def get_standard_params():

    d = 15
    n = 24000
    c = 4
    num_obj = 1
    F = .1
    r = 1.
    cl = max(1, n//4000)
    min_size = 400
    std = .4
    dims_pr_cl = 3
    N_size = 0.0004 #(((num_obj*10)*cl/n)**(1/dims_pr_cl))*std/200.
    rounds = 3

    return n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, rounds

def get_run_file(experiment, method, n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, generator, round):
    return "experiments_data/"+ experiment +"/" + method + \
              "n" + str(n) + "d" + str(d) + "c" + str(c) + \
              "N_size" + str(N_size) + "F" + str(F) + "r" + str(r) + \
              "num_obj" + str(num_obj) + "min_size" + str(min_size) + \
              "cl" + str(cl) + "std" + str(std) + "dims_pr_cl" + str(dims_pr_cl) + "generator_" + generator + "_round" + str(round) + ".npz"

def run(experiment, method, n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, round, generator="gaussian"):

    def load_synt_wrap(d, n, cl, std, cl_n=None, cl_d=None, noise=0.01, re=0):
        return load_synt(d, n, cl, re, cl_d = cl_d)

    gen = None
    if generator == "gaussian":
        gen = load_synt_gauss
    elif generator == "uniform":
        gen = load_synt_wrap


    run_file = get_run_file(experiment, method, n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, generator, round)

    if not os.path.exists(run_file):
        X = gen(n=n, d=d, cl=cl, std=std, re=round, cl_d=dims_pr_cl)

        t0 = time.time()
        subspaces, clusterings = GPU_INSCY_memory(X, N_size, F, num_obj, min_size, r, number_of_cells=c, rectangular=True)
        t1 = time.time()
        running_time = t1-t0

        np.savez(run_file, running_time=running_time, subspaces=subspaces, clusterings=clusterings)

    else:
        data = np.load(run_file, allow_pickle=True)
        running_time = data["running_time"]
        subspaces = data["subspaces"]
        clusterings = data["clusterings"]

    return running_time, subspaces, clusterings

def plot(avg_running_times, xs, x_label, experiment, y_max=None):

    print(avg_running_times)
    print(xs)

    plt.rcParams.update({'font.size': font_size})
    plt.plot(xs[:len(avg_running_times)], avg_running_times, color="#004488", marker = "x")
    plt.gcf().subplots_adjust(left=0.14)
    plt.ylabel('time in seconds')
    plt.xlabel(x_label)
    if not y_max is None:
        plt.ylim(0, y_max)
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

        #N_size = (((num_obj*10)*cl/n)**(1/dims_pr_cl))*std/200.

        avg_running_time = 0.
        for round in range(rounds):
            running_time, subspaces, clusterings = run("inc_cl", "GPU_INSCY_memory", n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, round)
            avg_running_time += running_time

        avg_running_time /= rounds
        avg_running_times.append(avg_running_time)

    plot(avg_running_times, cls, "number of clusters", "inc_cl", y_max=dist_lim)

def run_diff_number_of_cl_std():
    n, d, c, N_size, F, r, num_obj, min_size, _, std, dims_pr_cl, rounds = get_standard_params()
    cls = [2, 4, 8, 16, 32, 64]

    print("running experiment: inc_cl")

    if not os.path.exists('experiments_data/inc_cl_std/'):
        os.makedirs('experiments_data/inc_cl_std/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    avg_running_times = []
    for cl in cls:
        print("cl:", cl)

        #N_size = (((num_obj*10)*cl/n)**(1/dims_pr_cl))*std/200.
        std = 32*5/cl

        avg_running_time = 0.
        for round in range(rounds):
            running_time, subspaces, clusterings = run("inc_cl_std", "GPU_INSCY_memory", n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, round)
            avg_running_time += running_time

        avg_running_time /= rounds
        avg_running_times.append(avg_running_time)

    plot(avg_running_times, cls, "number of clusters", "inc_cl_std", y_max=dist_lim)



def run_diff_std():
    n, d, c, N_size, F, r, num_obj, min_size, cl, _, dims_pr_cl, rounds = get_standard_params()
    stds = [0.25, 0.5, 1, 2, 4, 8]

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

    plot(avg_running_times, stds, "standard deviation", "inc_std", y_max=dist_lim)




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
    _, d, c, N_size, F, r, num_obj, min_size, _, std, dims_pr_cl, rounds = get_standard_params()
    ns =  [8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000]


    print("running experiment: inc_n_large")

    if not os.path.exists('experiments_data/inc_n_large/'):
        os.makedirs('experiments_data/inc_n_large/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    avg_running_times = []
    for n  in ns:
        cl = max(1, n//4000)
        print("n:", n, "cl:", cl)
        avg_running_time = 0.
        for round in range(rounds):
            running_time, subspaces, clusterings = run("inc_n_large", "GPU_INSCY_memory", n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, round)
            avg_running_time += running_time

        avg_running_time /= rounds
        avg_running_times.append(avg_running_time)

    plot(avg_running_times, ns, "number of points", "inc_n_large", y_max=1100)

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

    plot(avg_running_times, ds, "number of dimensions", "inc_d_large", y_max=1100)


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

def run_diff_distribution():
    n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, rounds = get_standard_params()

    gens = ['gaussian','uniform']

    print("running experiment: diff_distribution")

    if not os.path.exists('experiments_data/diff_dist/'):
        os.makedirs('experiments_data/diff_dist/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    avg_running_times = []
    for gen in gens:
        print("gen:", gen)
        avg_running_time = 0.
        for round in range(rounds):
            running_time, subspaces, clusterings = run("diff_dist", "GPU_INSCY_memory", n, d, c, N_size, F, r, num_obj, min_size, cl, std, dims_pr_cl, round, generator=gen)
            avg_running_time += running_time

        avg_running_time /= rounds
        avg_running_times.append(avg_running_time)


    plt.rcParams.update({'font.size': font_size})
#     plt.figure(figsize=(4,6))
    x = np.arange(2)
    plt.bar(x, height=avg_running_times, color="#004488")
    plt.xticks(x, ['Gaussian','Uniform'])

    plt.gcf().subplots_adjust(left=0.14)
    plt.ylabel('time in seconds')
    plt.xlabel('method')
    plt.ylim((0.,dist_lim))
    plt.tight_layout()
    plt.savefig("plots/diff_dist.pdf")
    #plt.show()
    plt.clf()



experiment = sys.argv[1]
if experiment == "inc_cl":
    run_diff_number_of_cl()
if experiment == "inc_cl_std":
    run_diff_number_of_cl_std()
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
elif experiment == "diff_dist":
    run_diff_distribution()
elif experiment == "all":
    run_diff_n()
    run_diff_d()
    run_diff_number_of_cl()
    run_diff_std()
    run_diff_distribution()
    run_diff_number_of_cl_std()
