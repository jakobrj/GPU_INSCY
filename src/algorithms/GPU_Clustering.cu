#include "GPU_Clustering.cuh"
#include "../utils/util.cuh"
#include "../utils/TmpMalloc.cuh"
#include "../structures/GPU_SCY_tree.cuh"


#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512

#define PI 3.14

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


using namespace std;

__device__
float dist_gpu(int p_id, int q_id, float *X, int *subspace, int subspace_size, int d) {
    float *p = &X[p_id * d];
    float *q = &X[q_id * d];
    float distance = 0;
    for (int i = 0; i < subspace_size; i++) {
        int d_i = subspace[i];
        float diff = p[d_i] - q[d_i];
        distance += diff * diff;
    }
    return sqrt(distance);
}

__device__
float phi_gpu(int p_id, int *d_neighborhood, float neighborhood_size, int number_of_neighbors,
              float *X, int *d_points, int *subspace, int subspace_size, int d) {
    float sum = 0;
    for (int j = 0; j < number_of_neighbors; j++) {
        int q_id = d_points[d_neighborhood[j]];
        if (q_id >= 0) {
            float distance = dist_gpu(p_id, q_id, X, subspace, subspace_size, d) / neighborhood_size;
            float sq = distance * distance;
            sum += (1. - sq);
        }
    }
    return sum;
}

__device__
float gamma_gpu(int n) {
    if (n == 2) {
        return 1.;
    } else if (n == 1) {
        return sqrt(PI);
    }
    return (n / 2. - 1.) * gamma_gpu(n - 2);
}

__device__
float c_gpu(int subspace_size) {
    float r = pow(PI, subspace_size / 2.);
    r = r / gamma_gpu(subspace_size + 2);
    return r;
}

__device__
float alpha_gpu(int subspace_size, float neighborhood_size, int n) {
    float v = 1.;
    float r = 2 * n * pow(neighborhood_size, subspace_size) * c_gpu(subspace_size);
    r = r / (pow(v, subspace_size) * (subspace_size + 2));
    return r;
}

__device__
float expDen_gpu(int subspace_size, float neighborhood_size, int n) {
    float v = 1.;
    float r = n * c_gpu(subspace_size) * pow(neighborhood_size, subspace_size);
    r = r / pow(v, subspace_size);
    return r;
}

__device__
float omega_gpu(int subspace_size) {
    return 2.0 / (subspace_size + 2.0);
}

__global__
void
kernel_find_neighborhood_sizes_1(int **d_new_neighborhood_sizes_list, float *d_X, int n, int d,
                                      float neighborhood_size,
                                      int **subspace_list, int *d_subspace_size) {
    int k = blockIdx.y;

    int *d_new_neighborhood_sizes = d_new_neighborhood_sizes_list[k];
    int *subspace = subspace_list[k];
    int subspace_size = d_subspace_size[k];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int number_of_neighbors = 0;
    for (int j = 0; j < n; j++) {
        if (i != j) {
            float distance = dist_gpu(i, j, d_X, subspace, subspace_size, d);
            if (neighborhood_size >= distance) {
                number_of_neighbors++;
            }
        }
    }
    d_new_neighborhood_sizes[i] = number_of_neighbors;
}

__global__
void
kernel_find_neighborhoods_1(int **d_new_neighborhoods_list, int **d_new_neighborhood_end_list, float *d_X, int n,
                                 int d,
                                 float neighborhood_size, int **subspace_list, int *d_subspace_size) {
    int k = blockIdx.y;

    int *d_new_neighborhoods = d_new_neighborhoods_list[k];
    int *d_new_neighborhood_end = d_new_neighborhood_end_list[k];
    int *subspace = subspace_list[k];
    int subspace_size = d_subspace_size[k];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int new_offset = i > 0 ? d_new_neighborhood_end[i - 1] : 0;
    int *d_new_neighborhood = d_new_neighborhoods + new_offset;

    int number_of_neighbors = 0;
    for (int j = 0; j < n; j++) {
        if (i != j) {
            float distance = dist_gpu(i, j, d_X, subspace, subspace_size, d);
            if (neighborhood_size >= distance) {
                d_new_neighborhood[number_of_neighbors] = j;
                number_of_neighbors++;
            }
        }
    }
}

__global__
void
kernel_find_neighborhood_sizes_2(int *d_neighborhoods, int *d_neighborhood_end,
                                      int **d_new_neighborhood_sizes_list,
                                      float *d_X, int n, int d, float neighborhood_size,
                                      int **points_list, int *d_number_of_points,
                                      int **subspace_list, int *d_subspace_size) {
    int k = blockIdx.y;

    int *d_new_neighborhood_sizes = d_new_neighborhood_sizes_list[k];
    int *subspace = subspace_list[k];
    int subspace_size = d_subspace_size[k];
    int *points = points_list[k];
    int number_of_points = d_number_of_points[k];


    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < number_of_points; i += blockDim.x * gridDim.x) {

        int p_id = points[i];

        int number_of_neighbors = 0;
        int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
        for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
            int q_id = d_neighborhoods[j];

            if (p_id != q_id) {
                float distance = dist_gpu(p_id, q_id, d_X, subspace, subspace_size, d);
                if (neighborhood_size >= distance) {
                    number_of_neighbors++;
                }
            }
        }
        d_new_neighborhood_sizes[p_id] = number_of_neighbors;
    }
}

__global__
void kernel_find_neighborhoods_2(int *d_neighborhoods, int *d_neighborhood_end,
                                      int **d_new_neighborhoods_list, int **d_new_neighborhood_end_list,
                                      float *d_X, int n, int d, float neighborhood_size,
                                      int **points_list, int *d_number_of_points,
                                      int **subspace_list, int *d_subspace_size) {
    int k = blockIdx.y;

    int *d_new_neighborhoods = d_new_neighborhoods_list[k];
    int *d_new_neighborhood_end = d_new_neighborhood_end_list[k];
    int *subspace = subspace_list[k];
    int subspace_size = d_subspace_size[k];
    int *points = points_list[k];
    int number_of_points = d_number_of_points[k];


    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < number_of_points; i += blockDim.x * gridDim.x) {

        int p_id = points[i];

        int new_offset = p_id > 0 ? d_new_neighborhood_end[p_id - 1] : 0;
        int *d_new_neighborhood = d_new_neighborhoods + new_offset;

        int number_of_neighbors = 0;

        int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
        for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
            int q_id = d_neighborhoods[j];
            if (p_id != q_id) {
                float distance = dist_gpu(p_id, q_id, d_X, subspace, subspace_size, d);
                if (neighborhood_size >= distance) {
                    d_new_neighborhood[number_of_neighbors] = q_id;
                    number_of_neighbors++;
                }
            }
        }
    }
}

pair<int **, int **> find_neighborhoods(TmpMalloc *tmps, int *d_neighborhoods, int *d_neighborhood_end,
                                            float *d_X, int n, int d, GPU_SCY_tree *scy_tree,
                                            vector <vector<GPU_SCY_tree *>> L_merged,
                                            float neighborhood_size) {
    int size = 0;

    for (vector < GPU_SCY_tree * > list: L_merged) {
        for (GPU_SCY_tree *restricted_scy_tree: list) {
            size++;
        }
    }

    if (size == 0)
        return pair<int **, int **>();

    int *h_restricted_dims_list[size];
    int h_number_of_restricted_dims[size];
    int *h_points_list[size];
    int h_number_of_points[size];

    int **h_new_neighborhoods_list = new int *[size];
    int *h_new_neighborhood_sizes_list[size];
    int **h_new_neighborhood_end_list = new int *[size];

    int j = 0;
    int avg_number_of_points = 0;
    for (vector < GPU_SCY_tree * > list: L_merged) {
        for (GPU_SCY_tree *restricted_scy_tree: list) {
            h_restricted_dims_list[j] = restricted_scy_tree->d_restricted_dims;
            h_number_of_restricted_dims[j] = restricted_scy_tree->number_of_restricted_dims;
            h_points_list[j] = restricted_scy_tree->d_points;
            h_number_of_points[j] = restricted_scy_tree->number_of_points;

            avg_number_of_points += restricted_scy_tree->number_of_points;

            h_new_neighborhood_sizes_list[j] = tmps->malloc_points();
            h_new_neighborhood_end_list[j] = tmps->malloc_points();

            cudaMemset(h_new_neighborhood_end_list[j], 0, n * sizeof(int));
            cudaMemset(h_new_neighborhood_sizes_list[j], 0, n * sizeof(int));

            j++;
        }
    }
    if (size > 0) {
        avg_number_of_points /= size;
        avg_number_of_points = max(1, avg_number_of_points);
    } else {
        avg_number_of_points = 1;
    }

    int **d_restricted_dims_list;
    cudaMalloc(&d_restricted_dims_list, size * sizeof(int *));
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(d_restricted_dims_list, h_restricted_dims_list, size * sizeof(int *), cudaMemcpyHostToDevice);
    int *d_number_of_restricted_dims;
    cudaMalloc(&d_number_of_restricted_dims, size * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(d_number_of_restricted_dims, h_number_of_restricted_dims, size * sizeof(int), cudaMemcpyHostToDevice);
    int **d_points_list;
    cudaMalloc(&d_points_list, size * sizeof(int *));
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(d_points_list, h_points_list, size * sizeof(int *), cudaMemcpyHostToDevice);
    int *d_number_of_points;
    cudaMalloc(&d_number_of_points, size * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(d_number_of_points, h_number_of_points, size * sizeof(int), cudaMemcpyHostToDevice);

    int **d_new_neighborhoods_list;
    cudaMalloc(&d_new_neighborhoods_list, size * sizeof(int *));
    int **d_new_neighborhood_sizes_list;
    cudaMalloc(&d_new_neighborhood_sizes_list, size * sizeof(int *));
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(d_new_neighborhood_sizes_list, h_new_neighborhood_sizes_list, size * sizeof(int *),
               cudaMemcpyHostToDevice);
    int **d_new_neighborhood_end_list;
    cudaMalloc(&d_new_neighborhood_end_list, size * sizeof(int *));
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(d_new_neighborhood_end_list, h_new_neighborhood_end_list, size * sizeof(int *), cudaMemcpyHostToDevice);

    gpuErrchk(cudaPeekAtLastError());

    if (scy_tree->number_of_restricted_dims == 0) {

        int number_of_blocks = n / BLOCK_SIZE;
        if (n % BLOCK_SIZE) number_of_blocks++;
        int number_of_threads = min(n, BLOCK_SIZE);
        dim3 block(number_of_threads);
        dim3 grid(number_of_blocks, size);

        gpuErrchk(cudaPeekAtLastError());

        kernel_find_neighborhood_sizes_1<<<grid, block >> >(d_new_neighborhood_sizes_list,
                                                                 d_X, n, d, neighborhood_size,
                                                                 d_restricted_dims_list,
                                                                 d_number_of_restricted_dims);

        gpuErrchk(cudaPeekAtLastError());

        for (int j = 0; j < size; j++) {
            int total_size;

            gpuErrchk(cudaPeekAtLastError());
            inclusive_scan_any(h_new_neighborhood_sizes_list[j], h_new_neighborhood_end_list[j], n, tmps);
            gpuErrchk(cudaPeekAtLastError());

            cudaMemcpy(&total_size, h_new_neighborhood_end_list[j] + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            gpuErrchk(cudaPeekAtLastError());

            int *tmp;
            cudaMalloc(&tmp, total_size * sizeof(int));
            h_new_neighborhoods_list[j] = tmp;
        }

        cudaMemcpy(d_new_neighborhoods_list, h_new_neighborhoods_list, size * sizeof(int *), cudaMemcpyHostToDevice);

        kernel_find_neighborhoods_1<<<grid, block >> >(d_new_neighborhoods_list, d_new_neighborhood_end_list,
                                                            d_X, n, d, neighborhood_size,
                                                            d_restricted_dims_list,
                                                            d_number_of_restricted_dims);


        gpuErrchk(cudaPeekAtLastError());
    } else {
        int number_of_blocks = avg_number_of_points / BLOCK_SIZE;
        if (avg_number_of_points % BLOCK_SIZE) number_of_blocks++;
        int number_of_threads = min(avg_number_of_points, BLOCK_SIZE);
        dim3 block(64);
        dim3 grid(16, size);

        gpuErrchk(cudaPeekAtLastError());

        kernel_find_neighborhood_sizes_2<<<grid, block >> >(d_neighborhoods, d_neighborhood_end,
                                                                 d_new_neighborhood_sizes_list, d_X, n, d,
                                                                 neighborhood_size,
                                                                 d_points_list,
                                                                 d_number_of_points,
                                                                 d_restricted_dims_list,
                                                                 d_number_of_restricted_dims);
        gpuErrchk(cudaPeekAtLastError());

        for (int j = 0; j < size; j++) {
            int total_size;

            gpuErrchk(cudaPeekAtLastError());
            inclusive_scan_any(h_new_neighborhood_sizes_list[j], h_new_neighborhood_end_list[j], n, tmps);
            gpuErrchk(cudaPeekAtLastError());

            cudaMemcpy(&total_size, h_new_neighborhood_end_list[j] + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            gpuErrchk(cudaPeekAtLastError());

            int *tmp;
            cudaMalloc(&tmp, total_size * sizeof(int));
            h_new_neighborhoods_list[j] = tmp;
        }
        cudaMemcpy(d_new_neighborhoods_list, h_new_neighborhoods_list, size * sizeof(int *), cudaMemcpyHostToDevice);

        kernel_find_neighborhoods_2<<<grid, block >> >(d_neighborhoods, d_neighborhood_end,
                                                            d_new_neighborhoods_list, d_new_neighborhood_end_list,
                                                            d_X, n, d, neighborhood_size,
                                                            d_points_list,
                                                            d_number_of_points,
                                                            d_restricted_dims_list,
                                                            d_number_of_restricted_dims);

    }

    for (int j = 0; j < size; j++) {
        tmps->free_points(h_new_neighborhood_sizes_list[j]);
    }

    cudaFree(d_new_neighborhoods_list);
    cudaFree(d_new_neighborhood_sizes_list);
    cudaFree(d_new_neighborhood_end_list);

    cudaFree(d_restricted_dims_list);
    cudaFree(d_number_of_restricted_dims);
    cudaFree(d_points_list);
    cudaFree(d_number_of_points);

    return pair<int **, int **>(h_new_neighborhoods_list, h_new_neighborhood_end_list);
}


pair<int **, int **>
find_neighborhoods_star(TmpMalloc *tmps, int *d_neighborhoods, int *d_neighborhood_end, float *d_X, int n, int d,
                           GPU_SCY_tree *scy_tree, vector <vector<GPU_SCY_tree *>> L_merged, float neighborhood_size) {
    int size = 0;

    for (vector < GPU_SCY_tree * > list: L_merged) {
        for (GPU_SCY_tree *restricted_scy_tree: list) {
            size++;
        }
    }

    if (size == 0)
        return pair<int **, int **>();

    int *h_restricted_dims_list[size];
    int h_number_of_restricted_dims[size];
    int *h_points_list[size];
    int h_number_of_points[size];

    int **h_new_neighborhoods_list = new int *[size];
    int *h_new_neighborhood_sizes_list[size];
    int **h_new_neighborhood_end_list = new int *[size];

    int j = 0;
    int avg_number_of_points = 0;
    for (vector < GPU_SCY_tree * > list: L_merged) {
        for (GPU_SCY_tree *restricted_scy_tree: list) {
            h_restricted_dims_list[j] = restricted_scy_tree->d_restricted_dims;
            h_number_of_restricted_dims[j] = restricted_scy_tree->number_of_restricted_dims;
            h_points_list[j] = restricted_scy_tree->d_points;
            h_number_of_points[j] = restricted_scy_tree->number_of_points;

            avg_number_of_points += restricted_scy_tree->number_of_points;

            h_new_neighborhood_sizes_list[j] = tmps->malloc_points();
            h_new_neighborhood_end_list[j] = tmps->malloc_points();

            cudaMemset(h_new_neighborhood_end_list[j], 0, n * sizeof(int));
            cudaMemset(h_new_neighborhood_sizes_list[j], 0, n * sizeof(int));

            j++;
        }
    }
    if (size > 0) {
        avg_number_of_points /= size;
        avg_number_of_points = max(1, avg_number_of_points);
    } else {
        avg_number_of_points = 1;
    }

    int **d_restricted_dims_list;
    cudaMalloc(&d_restricted_dims_list, size * sizeof(int *));
    cudaMemcpy(d_restricted_dims_list, h_restricted_dims_list, size * sizeof(int *), cudaMemcpyHostToDevice);
    int *d_number_of_restricted_dims;
    cudaMalloc(&d_number_of_restricted_dims, size * sizeof(int));
    cudaMemcpy(d_number_of_restricted_dims, h_number_of_restricted_dims, size * sizeof(int), cudaMemcpyHostToDevice);
    int **d_points_list;
    cudaMalloc(&d_points_list, size * sizeof(int *));
    cudaMemcpy(d_points_list, h_points_list, size * sizeof(int *), cudaMemcpyHostToDevice);
    int *d_number_of_points;
    cudaMalloc(&d_number_of_points, size * sizeof(int));
    cudaMemcpy(d_number_of_points, h_number_of_points, size * sizeof(int), cudaMemcpyHostToDevice);

    int **d_new_neighborhoods_list;
    cudaMalloc(&d_new_neighborhoods_list, size * sizeof(int *));
    int **d_new_neighborhood_sizes_list;
    cudaMalloc(&d_new_neighborhood_sizes_list, size * sizeof(int *));
    cudaMemcpy(d_new_neighborhood_sizes_list, h_new_neighborhood_sizes_list, size * sizeof(int *),
               cudaMemcpyHostToDevice);
    int **d_new_neighborhood_end_list;
    cudaMalloc(&d_new_neighborhood_end_list, size * sizeof(int *));
    cudaMemcpy(d_new_neighborhood_end_list, h_new_neighborhood_end_list, size * sizeof(int *), cudaMemcpyHostToDevice);

    gpuErrchk(cudaPeekAtLastError());

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = min(n, BLOCK_SIZE);
    dim3 block(number_of_threads);
    dim3 grid(number_of_blocks, size);

    gpuErrchk(cudaPeekAtLastError());

    kernel_find_neighborhood_sizes_1<<<grid, block >> >(d_new_neighborhood_sizes_list,
                                                             d_X, n, d, neighborhood_size,
                                                             d_restricted_dims_list,
                                                             d_number_of_restricted_dims);

    gpuErrchk(cudaPeekAtLastError());

    for (int j = 0; j < size; j++) {
        int total_size;

        gpuErrchk(cudaPeekAtLastError());
        inclusive_scan_any(h_new_neighborhood_sizes_list[j], h_new_neighborhood_end_list[j], n, tmps);
        gpuErrchk(cudaPeekAtLastError());

        cudaMemcpy(&total_size, h_new_neighborhood_end_list[j] + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
        gpuErrchk(cudaPeekAtLastError());

        int *tmp;
        cudaMalloc(&tmp, total_size * sizeof(int));
        h_new_neighborhoods_list[j] = tmp;
    }

    cudaMemcpy(d_new_neighborhoods_list, h_new_neighborhoods_list, size * sizeof(int *), cudaMemcpyHostToDevice);

    kernel_find_neighborhoods_1<<<grid, block >> >(d_new_neighborhoods_list, d_new_neighborhood_end_list,
                                                        d_X, n, d, neighborhood_size,
                                                        d_restricted_dims_list,
                                                        d_number_of_restricted_dims);

    gpuErrchk(cudaPeekAtLastError());


    for (int j = 0; j < size; j++) {
        tmps->free_points(h_new_neighborhood_sizes_list[j]);
    }

    cudaFree(d_new_neighborhoods_list);
    cudaFree(d_new_neighborhood_sizes_list);
    cudaFree(d_new_neighborhood_end_list);

    cudaFree(d_restricted_dims_list);
    cudaFree(d_number_of_restricted_dims);
    cudaFree(d_points_list);
    cudaFree(d_number_of_points);

    return pair<int **, int **>(h_new_neighborhoods_list, h_new_neighborhood_end_list);
}

__global__
void
disjoint_set_clustering(int **d_clustering_list,
                                 int **d_neighborhoods_list, int **d_neighborhood_end_list,
                                 bool *d_is_dense_list, int **d_points_list, int *d_number_of_points, int n) {
    int j = blockIdx.x;
    int number_of_points = d_number_of_points[j];
    int *d_clustering = d_clustering_list[j];
    int *d_points = d_points_list[j];
    int *d_neighborhoods = d_neighborhoods_list[j];
    int *d_neighborhood_end = d_neighborhood_end_list[j];
    bool *d_is_dense = &d_is_dense_list[j * n];


    __shared__ int changed;
    changed = 1;
    __syncthreads();
    for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
        int p_id = d_points[i];
        if (d_is_dense[p_id]) {
            d_clustering[p_id] = p_id;
        }
    }

    __syncthreads();

    while (changed) {
        __syncthreads();
        changed = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
            int p_id = d_points[i];
            if (!d_is_dense[p_id]) continue;

            int root = d_clustering[p_id];

            int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
            for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
                int q_id = d_neighborhoods[j];
                if (d_is_dense[q_id]) {
                    if (d_clustering[q_id] < root) {
                        root = d_clustering[q_id];
                        changed = 1;
                    }
                }
            }
            d_clustering[p_id] = root;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
            int p_id = d_points[i];
            int root = d_clustering[p_id];
            while (root >= 0 && root != d_clustering[root]) {
                root = d_clustering[root];
            }
            d_clustering[p_id] = root;
        }
    }
}

__global__
void compute_is_dense_rectangular(bool *d_is_dense_list, int **d_points_list, int *d_number_of_points,
                                           int **d_neighborhoods_list, float neighborhood_size,
                                           int **d_neighborhood_end_list,
                                           float *X, int **subspace_list, int subspace_size, float F, int n,
                                           int num_obj, int d) {
    int j = blockIdx.y;

    int number_of_points = d_number_of_points[j];
    int *d_points = d_points_list[j];
    int *subspace = subspace_list[j];
    int *d_neighborhoods = d_neighborhoods_list[j];
    int *d_neighborhood_end = d_neighborhood_end_list[j];
    bool *d_is_dense = &d_is_dense_list[j * n];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < number_of_points; i += blockDim.x * gridDim.x) {

        int p_id = d_points[i];

        int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
        int neighbor_count = d_neighborhood_end[p_id] - offset;
        float a = expDen_gpu(subspace_size, neighborhood_size, n);
        d_is_dense[p_id] = neighbor_count >= max(F * a, (float) num_obj);
    }
}

__global__
void compute_is_dense(bool *d_is_dense_list, int **d_points_list, int *d_number_of_points,
                               int **d_neighborhoods_list, float neighborhood_size,
                               int **d_neighborhood_end_list,
                               float *X, int **subspace_list, int subspace_size, float F, int n,
                               int num_obj, int d) {

    int j = blockIdx.y;

    int number_of_points = d_number_of_points[j];
    int *d_points = d_points_list[j];
    int *subspace = subspace_list[j];
    int *d_neighborhoods = d_neighborhoods_list[j];
    int *d_neighborhood_end = d_neighborhood_end_list[j];
    bool *d_is_dense = &d_is_dense_list[j * n];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < number_of_points; i += blockDim.x * gridDim.x) {

        int p_id = d_points[i];

        float p = 0;
        int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
        for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
            int q_id = d_neighborhoods[j];
            if (q_id >= 0) {
                float distance = dist_gpu(p_id, q_id, X, subspace, subspace_size, d) / neighborhood_size;
                float sq = distance * distance;
                p += (1. - sq);
            }
        }
        float a = alpha_gpu(subspace_size, neighborhood_size, n);
        float w = omega_gpu(subspace_size);
        d_is_dense[p_id] = p >= max(F * a, num_obj * w);
    }
}

void GPU_Clustering(vector<int *> new_neighborhoods_list, vector<int *> new_neighborhood_end_list, TmpMalloc *tmps,
                    vector<int *> clustering_list,
                    vector<GPU_SCY_tree *> restricted_scy_tree_list, float *d_X, int n, int d,
                    float neighborhood_size, float F,
                    int num_obj, bool rectangular) {

    int size = restricted_scy_tree_list.size();
    if (size == 0) return;


    int **d_clustering_list;
    cudaMalloc(&d_clustering_list, size * sizeof(int *));
    cudaMemcpy(d_clustering_list, clustering_list.data(), size * sizeof(int *), cudaMemcpyHostToDevice);
    int **d_neighborhoods_list;
    cudaMalloc(&d_neighborhoods_list, size * sizeof(int *));
    cudaMemcpy(d_neighborhoods_list, new_neighborhoods_list.data(), size * sizeof(int *), cudaMemcpyHostToDevice);
    int **d_neighborhood_end_list;
    cudaMalloc(&d_neighborhood_end_list, size * sizeof(int *));
    cudaMemcpy(d_neighborhood_end_list, new_neighborhood_end_list.data(), size * sizeof(int *), cudaMemcpyHostToDevice);

    gpuErrchk(cudaPeekAtLastError());
    int *h_points_list[size];
    int *h_restricted_dims_list[size];
    int h_number_of_points[size];

    int number_of_points = 0;
    for (int i = 0; i < size; i++) {
        GPU_SCY_tree *restricted_scy_tree = restricted_scy_tree_list[i];
        h_points_list[i] = restricted_scy_tree->d_points;
        h_restricted_dims_list[i] = restricted_scy_tree->d_restricted_dims;
        h_number_of_points[i] = restricted_scy_tree->number_of_points;
        number_of_points += restricted_scy_tree->number_of_points;
    }
    number_of_points /= size;

    int **d_points_list;
    cudaMalloc(&d_points_list, size * sizeof(int *));
    cudaMemcpy(d_points_list, h_points_list, size * sizeof(int *), cudaMemcpyHostToDevice);

    int **d_restricted_dims_list;
    cudaMalloc(&d_restricted_dims_list, size * sizeof(int *));
    cudaMemcpy(d_restricted_dims_list, h_restricted_dims_list, size * sizeof(int *), cudaMemcpyHostToDevice);

    int *d_number_of_points;
    cudaMalloc(&d_number_of_points, size * sizeof(int));
    cudaMemcpy(d_number_of_points, h_number_of_points, size * sizeof(int), cudaMemcpyHostToDevice);

    int number_of_restricted_dims = restricted_scy_tree_list[0]->number_of_restricted_dims;

    bool *d_is_dense = tmps->get_bool_array(tmps->bool_array_counter++, size * n);

    cudaMemset(d_is_dense, 0, size * n * sizeof(bool));

    int number_of_blocks = number_of_points / BLOCK_SIZE;
    if (number_of_points % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = max(64, min(number_of_points, BLOCK_SIZE));

    gpuErrchk(cudaPeekAtLastError());

    dim3 grid(number_of_blocks, size);

    if (rectangular) {
        compute_is_dense_rectangular<<< grid, number_of_threads >> >
                (d_is_dense, d_points_list, d_number_of_points, d_neighborhoods_list, neighborhood_size,
                 d_neighborhood_end_list, d_X, d_restricted_dims_list,
                 number_of_restricted_dims, F, n, num_obj, d);
    } else {
        compute_is_dense<<< grid, number_of_threads >> >
                (d_is_dense, d_points_list, d_number_of_points, d_neighborhoods_list, neighborhood_size,
                 d_neighborhood_end_list, d_X, d_restricted_dims_list,
                 number_of_restricted_dims, F, n, num_obj, d);
    }

    gpuErrchk(cudaPeekAtLastError());

    disjoint_set_clustering<<< size, number_of_threads >> >
            (d_clustering_list, d_neighborhoods_list, d_neighborhood_end_list,
             d_is_dense, d_points_list, d_number_of_points, n);


    cudaFree(d_points_list);
    cudaFree(d_restricted_dims_list);
    cudaFree(d_number_of_points);

    cudaFree(d_clustering_list);
    cudaFree(d_neighborhoods_list);
    cudaFree(d_neighborhood_end_list);
    gpuErrchk(cudaPeekAtLastError());
}