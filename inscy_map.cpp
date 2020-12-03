#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "src/utils/util.cuh"
#include "src/utils/TmpMalloc.cuh"
#include "src/structures/SCY_tree.h"
#include "src/structures/Neighborhood_tree.h"
#include "src/structures/GPU_SCY_tree.cuh"
#include "src/algorithms/Clustering.h"
#include "src/algorithms/GPU_Clustering.cuh"
#include "src/algorithms/INSCY.h"
#include "src/algorithms/GPU_INSCY.cuh"

#define BLOCK_SIZE 512

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

using namespace std;

vector<vector<vector<int>>> run_INSCY(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r, int number_of_cells,
               bool rectangular) {

    int n = X.size(0);
    int d = X.size(1);

    float *maxs = new float[d];
    float *mins = new float[d];
    int *subspace = new int[d];
    for (int j = 0; j < d; j++) {
        subspace[j] = j;
        maxs[j] = std::numeric_limits<float>::lowest();
        mins[j] = std::numeric_limits<float>::max();
    }

    for (int i = 0; i < n; i++) {
        float *x_i = X[i].data_ptr<float>();
        for (int j = 0; j < d; j++) {
            if (x_i[j] > maxs[j])
                maxs[j] = x_i[j];
            if (x_i[j] < mins[j])
                mins[j] = x_i[j];
        }
    }


    SCY_tree *scy_tree = new SCY_tree(X, subspace, number_of_cells, d, n, neighborhood_size, mins, maxs);
    //SCY_tree *neighborhood_tree = new SCY_tree(X, subspace, ceil(1. / neighborhood_size), d, n,
    //                                           neighborhood_size);

    Neighborhood_tree *neighborhood_tree = new Neighborhood_tree(X, neighborhood_size, mins);

    map <vector<int>, vector<int>, vec_cmp> result;

    int calls = 0;
    INSCY(scy_tree, neighborhood_tree, X, n, neighborhood_size, F, num_obj, min_size,
          result, 0, d, r, calls, rectangular);
    printf("INSCY(%d): 100%%      \n", calls);

    vector < vector < vector < int>>> tuple;
    vector <vector<int>> subspaces(result.size());
    vector <vector<int>> clusterings(result.size());

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        subspaces[j] = dims;
        vector<int> clustering = p.second;
        clusterings[j] = clustering;
        j++;
    }
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);


    return tuple;
}


vector<vector<vector<int>>> run_GPU_INSCY(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r, int number_of_cells,
              bool rectangular) {

    int n = X.size(0);
    int d = X.size(1);

    float *maxs = new float[d];
    float *mins = new float[d];
    int *subspace = new int[d];
    for (int j = 0; j < d; j++) {
        subspace[j] = j;
        maxs[j] = std::numeric_limits<float>::lowest();
        mins[j] = std::numeric_limits<float>::max();
    }

    for (int i = 0; i < n; i++) {
        float *x_i = X[i].data_ptr<float>();
        for (int j = 0; j < d; j++) {
            if (x_i[j] > maxs[j])
                maxs[j] = x_i[j];
            if (x_i[j] < mins[j])
                mins[j] = x_i[j];
        }
    }

    float *d_X = copy_to_device(X, n, d);
    cudaDeviceSynchronize();
    
    SCY_tree *scy_tree = new SCY_tree(X, subspace, number_of_cells, d, n, neighborhood_size, mins, maxs);

    map<vector<int>, int *, vec_cmp> result;

    int calls = 0;
    GPU_SCY_tree *gpu_scy_tree = scy_tree->convert_to_GPU_SCY_tree();
    gpu_scy_tree->copy_to_device();
    cudaDeviceSynchronize();
    TmpMalloc *tmps = new TmpMalloc();

    tmps->set(gpu_scy_tree->number_of_points, gpu_scy_tree->number_of_nodes, gpu_scy_tree->number_of_dims);

    int *d_neighborhoods;
    int *d_neighborhood_end;

    GPU_INSCY(d_neighborhoods, d_neighborhood_end, tmps, gpu_scy_tree, d_X, n, d,
              neighborhood_size, F, num_obj, min_size,
              result, 0, d, r, calls, rectangular);
    cudaDeviceSynchronize();
    printf("GPU_INSCY(%d)\n", calls);


    vector < vector < vector < int>>> tuple;
    vector <vector<int>> subspaces(result.size());
    vector <vector<int>> clusterings(result.size());

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        subspaces[j] = dims;
        int *d_clustering = p.second;
        vector<int> clustering(n);
        cudaMemcpy(clustering.data(), d_clustering, n * sizeof(int), cudaMemcpyDeviceToHost);
        tmps->free_points(d_clustering);
        clusterings[j] = clustering;
        j++;
    }
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);

    cudaDeviceSynchronize();

    cudaFree(d_X);
    delete gpu_scy_tree;
    tmps->free_all();
    delete tmps;
    cudaDeviceSynchronize();

    return tuple;
}


vector<vector<vector<int>>> run_GPU_INSCY_star(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r,
                        int number_of_cells, bool rectangular) {

    int n = X.size(0);
    int d = X.size(1);

    float *maxs = new float[d];
    float *mins = new float[d];
    int *subspace = new int[d];
    for (int j = 0; j < d; j++) {
        subspace[j] = j;
        maxs[j] = std::numeric_limits<float>::lowest();
        mins[j] = std::numeric_limits<float>::max();
    }

    for (int i = 0; i < n; i++) {
        float *x_i = X[i].data_ptr<float>();
        for (int j = 0; j < d; j++) {
            if (x_i[j] > maxs[j])
                maxs[j] = x_i[j];
            if (x_i[j] < mins[j])
                mins[j] = x_i[j];
        }
    }

    float *d_X = copy_to_device(X, n, d);
    cudaDeviceSynchronize();

    SCY_tree *scy_tree = new SCY_tree(X, subspace, number_of_cells, d, n, neighborhood_size, mins, maxs);

    map<vector<int>, int *, vec_cmp> result;

    int calls = 0;
    GPU_SCY_tree *scy_tree_gpu = scy_tree->convert_to_GPU_SCY_tree();
    scy_tree_gpu->copy_to_device();
    cudaDeviceSynchronize();

    TmpMalloc *tmps = new TmpMalloc();

    tmps->set(scy_tree_gpu->number_of_points, scy_tree_gpu->number_of_nodes, scy_tree_gpu->number_of_dims);

    int *d_neighborhoods;
    int *d_neighborhood_sizes;
    int *d_neighborhood_end;

    GPU_INSCY_star(d_neighborhoods, d_neighborhood_end, tmps, scy_tree_gpu, d_X, n, d,
                   neighborhood_size, F, num_obj, min_size,
                   result, 0, d, r, calls, rectangular);
    cudaDeviceSynchronize();
    printf("GPU_INSCY_star(%d)\n", calls);

    vector < vector < vector < int>>> tuple;
    vector <vector<int>> subspaces(result.size());
    vector <vector<int>> clusterings(result.size());

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        subspaces[j] = dims;
        int *d_clustering = p.second;
        vector<int> clustering(n);
        cudaMemcpy(clustering.data(), d_clustering, n * sizeof(int), cudaMemcpyDeviceToHost);
        tmps->free_points(d_clustering);
        clusterings[j] = clustering;
        j++;
    }
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);

    cudaDeviceSynchronize();

    cudaFree(d_X);
    delete scy_tree_gpu;
    tmps->free_all();
    delete tmps;
    cudaDeviceSynchronize();

    return tuple;
}


vector<vector<vector<int>>> run_GPU_INSCY_memory(at::Tensor X, float neighborhood_size, float F, int num_obj, int min_size, float r,
                          int number_of_cells, bool rectangular) {

    int n = X.size(0);
    int d = X.size(1);

    float *maxs = new float[d];
    float *mins = new float[d];
    int *subspace = new int[d];
    for (int j = 0; j < d; j++) {
        subspace[j] = j;
        maxs[j] = std::numeric_limits<float>::lowest();
        mins[j] = std::numeric_limits<float>::max();
    }

    for (int i = 0; i < n; i++) {
        float *x_i = X[i].data_ptr<float>();
        for (int j = 0; j < d; j++) {
            if (x_i[j] > maxs[j])
                maxs[j] = x_i[j];
            if (x_i[j] < mins[j])
                mins[j] = x_i[j];
        }
    }

    float *d_X = copy_to_device(X, n, d);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    SCY_tree *scy_tree = new SCY_tree(X, subspace, number_of_cells, d, n, neighborhood_size, mins, maxs);

    map<vector<int>, int *, vec_cmp> result;


    int calls = 0;
    GPU_SCY_tree *scy_tree_gpu = scy_tree->convert_to_GPU_SCY_tree();
    scy_tree_gpu->copy_to_device();
    cudaDeviceSynchronize();
    TmpMalloc *tmps = new TmpMalloc();

    tmps->set(scy_tree_gpu->number_of_points, scy_tree_gpu->number_of_nodes, scy_tree_gpu->number_of_dims);

    int *d_neighborhoods;
    int *d_neighborhood_sizes;
    int *d_neighborhood_end;


    GPU_INSCY_memory(d_neighborhoods, d_neighborhood_end, tmps, scy_tree_gpu, d_X, n, d,
                     neighborhood_size, F, num_obj, min_size,
                     result, 0, d, r, calls, rectangular);
    cudaDeviceSynchronize();
    printf("GPU_INSCY_memory(%d)\n", calls);


    vector < vector < vector < int>>> tuple;
    vector <vector<int>> subspaces(result.size());
    vector <vector<int>> clusterings(result.size());

    int j = 0;
    for (auto p : result) {
        vector<int> dims = p.first;
        subspaces[j] = dims;
        int *d_clustering = p.second;
        vector<int> clustering(n);
        cudaMemcpy(clustering.data(), d_clustering, n * sizeof(int), cudaMemcpyDeviceToHost);
        tmps->free_points(d_clustering);
        clusterings[j] = clustering;
        j++;
    }
    tuple.push_back(subspaces);
    tuple.push_back(clusterings);

    cudaDeviceSynchronize();

    cudaFree(d_X);
    delete scy_tree_gpu;
    tmps->free_all();
    delete tmps;
    cudaDeviceSynchronize();

    return tuple;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("run_INSCY",    &run_INSCY,    "");
m.def("run_GPU_INSCY",    &run_GPU_INSCY,    "");
m.def("run_GPU_INSCY_star",    &run_GPU_INSCY_star,    "");
m.def("run_GPU_INSCY_memory",    &run_GPU_INSCY_memory,    "");
}