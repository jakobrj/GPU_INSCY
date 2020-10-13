#include "util.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <numeric>

#include <ATen/ATen.h>
#include <torch/extension.h>
#include "nvToolsExt.h"
#include "TmpMalloc.cuh"

#define SECTION_SIZE 64
#define SECTION_SIZE_LARGE 512
#define BLOCK_WIDTH 64

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


using namespace std;

float *copy_to_device(vector <vector<float>> X, int number_of_points, int number_of_dims) {
    float *d_X;
    cudaMalloc(&d_X, sizeof(float) * number_of_points * number_of_dims);
    for (int i = 0; i < number_of_points; i++) {
        float *h_x_i = X[i].data();
        cudaMemcpy(&d_X[i * number_of_dims], h_x_i, sizeof(float) * number_of_dims, cudaMemcpyHostToDevice);
    }
    return d_X;
}


float *copy_to_device(at::Tensor X, int number_of_points, int number_of_dims) {
    float *d_X;
    cudaMalloc(&d_X, sizeof(float) * number_of_points * number_of_dims);
    cudaMemcpy(d_X, X.data_ptr<float>(), sizeof(float) * number_of_points * number_of_dims, cudaMemcpyHostToDevice);
    return d_X;
}

__global__
void print_array_gpu(int *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 10 && x[i] > -1)
            printf(" ");
        if (x[i] < 100 && x[i] > -10)
            printf(" ");
        printf("%d ", x[i]);
    }
    printf("\n");
}

__global__
void print_array_gpu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f ", x[i]);
    }
    printf("\n");
}

__global__
void print_array_gpu(bool *x, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", (int) x[i]);
    }
    printf("\n");
}

__global__
void scan_kernel_eff(int *x, int *y, int n) {
/**
 * from the cuda book
 */
    __shared__ int XY[SECTION_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        XY[threadIdx.x] = x[i];
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            XY[index] += XY[index - stride];
        }
    }

    for (int stride = SECTION_SIZE; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    if (i < n) {
        y[i] = XY[threadIdx.x];
    }
}


__global__
void scan_kernel_eff_large1(int *x, int *y, int *S, int n) {
/**
 * from the cuda book
 */
    __shared__ int XY[SECTION_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        XY[threadIdx.x] = x[i];
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            XY[index] += XY[index - stride];
        }
    }

    for (int stride = SECTION_SIZE; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    if (i < n) {
        y[i] = XY[threadIdx.x];
    }

    if (threadIdx.x == 0) {
        S[blockIdx.x] = XY[SECTION_SIZE - 1];
    }

}

__global__
void scan_kernel_eff_large3(int *y, int *S, int n) {
/**
 * from the cuda book
 */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && i < n) {
        y[i] += S[blockIdx.x - 1];
    }
}

void inclusive_scan(int *source, int *result, int n) {
    int numBlocks = n / SECTION_SIZE;
    if (n % SECTION_SIZE) numBlocks++;

    if (n > SECTION_SIZE) {
        int *S;
        cudaMalloc((void **) &S, numBlocks * sizeof(int));
        scan_kernel_eff_large1 << < numBlocks, SECTION_SIZE >> > (source, result, S, n);
        inclusive_scan(S, S, numBlocks);
        scan_kernel_eff_large3 << < numBlocks, SECTION_SIZE >> > (result, S, n);
        cudaFree(S);
    } else {
        scan_kernel_eff << < numBlocks, SECTION_SIZE >> > (source, result, n);
    }
}

void inclusive_scan_points(int *source, int *result, int n, TmpMalloc *tmps) {
    int numBlocks = n / SECTION_SIZE;
    if (n % SECTION_SIZE) numBlocks++;

    if (n > SECTION_SIZE) {
        int *S = tmps->malloc_points();
        scan_kernel_eff_large1 << < numBlocks, SECTION_SIZE >> > (source, result, S, n);
        inclusive_scan(S, S, numBlocks);
        scan_kernel_eff_large3 << < numBlocks, SECTION_SIZE >> > (result, S, n);
        tmps->free_points(S);
    } else {
        scan_kernel_eff << < numBlocks, SECTION_SIZE >> > (source, result, n);
    }
}


void inclusive_scan_nodes(int *source, int *result, int n, TmpMalloc *tmps) {
    int numBlocks = n / SECTION_SIZE;
    if (n % SECTION_SIZE) numBlocks++;

    if (n > SECTION_SIZE) {
        int *S = tmps->malloc_nodes();
        scan_kernel_eff_large1 << < numBlocks, SECTION_SIZE >> > (source, result, S, n);
        inclusive_scan(S, S, numBlocks);
        scan_kernel_eff_large3 << < numBlocks, SECTION_SIZE >> > (result, S, n);
        tmps->free_nodes(S);

    } else {
        scan_kernel_eff << < numBlocks, SECTION_SIZE >> > (source, result, n);
        gpuErrchk(cudaPeekAtLastError());
    }
}

void inclusive_scan_any(int *source, int *result, int n, TmpMalloc *tmps) {
    int numBlocks = n / SECTION_SIZE;
    if (n % SECTION_SIZE) numBlocks++;

    if (n > SECTION_SIZE) {
        int *S = tmps->malloc_any(numBlocks);
        scan_kernel_eff_large1 << < numBlocks, SECTION_SIZE >> > (source, result, S, n);
        inclusive_scan_any(S, S, numBlocks, tmps);
        scan_kernel_eff_large3 << < numBlocks, SECTION_SIZE >> > (result, S, n);
        tmps->free_any(S, numBlocks);

    } else {
        scan_kernel_eff << < numBlocks, SECTION_SIZE >> > (source, result, n);
        gpuErrchk(cudaPeekAtLastError());
    }
}

void inclusive_scan_cpu(int *d_x, int *d_y, int n) {
    int *h_x = new int[n];
    int *h_y = new int[n];
    cudaMemcpy(h_y, d_y, n * sizeof(int), cudaMemcpyDeviceToHost);

    int tmp = 0;
    for (int i = 0; i < n; i++) {
        tmp += h_y[i];
        h_x[i] = tmp;
    }

    cudaMemcpy(d_x, h_x, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    delete h_x;
    delete h_y;
}


void inclusive_scan_async(int *x, int *y, int n, cudaStream_t stream) {
    int numBlocks = n / BLOCK_WIDTH;
    if (n % BLOCK_WIDTH) numBlocks++;

    if (n > SECTION_SIZE) {
        int *S;
        cudaMalloc((void **) &S, (n / SECTION_SIZE) * sizeof(int));
        scan_kernel_eff_large1 << < numBlocks, BLOCK_WIDTH, 0, stream >> > (x, y, S, n);
        inclusive_scan_async(S, S, n / SECTION_SIZE, stream);
        scan_kernel_eff_large3 << < numBlocks, BLOCK_WIDTH, 0, stream >> > (y, S, n);
        cudaFree(S);
    } else {
        scan_kernel_eff << < numBlocks, BLOCK_WIDTH, 0, stream >> > (x, y, n);
    }
}

void populate(int *parents, int *cells, int *counts, int *dim_start, int *dims, int c, int d, int n) {
    int lvl_size = c - c * 1 / 3;
    int prev_lvl_size = 0;
    int prev_count = 0;

    for (int i = 0; i < d; i++) {
        dims[i] = d - i;
        dim_start[i] = prev_count;
        int p = -1;
        for (int j = 0; j < lvl_size; j++) {
            p += j % 3 == 2 ? 0 : 1;

            if (i == 0) {
                parents[j + prev_count] = -1;
            } else {
                parents[j + prev_count] = prev_count - prev_lvl_size + p;
            }
        }
        prev_count += lvl_size;
        prev_lvl_size = lvl_size;
        lvl_size *= 1.5;
    }

    for (int i = 0; i < d; i++) {
        int r_count = 0;
        int c_no = 0;
        for (int j = 0; j < ((i < d - 1 ? dim_start[i + 1] : n) - dim_start[i]); j++) {
            int m = (i == 0 ? c * 1 / 3 : c - 2);
            if (i != 0 && j % 3 != 2) {
                r_count = 0;
                c_no = 0;
            }
            while (r_count < m && rand() % c < m) {
                r_count++;
                c_no++;
            }

            cells[dim_start[i] + j] = c_no + 1;

            c_no++;
        }
    }

    for (int j = 0; j < dim_start[d - 1]; j++) {
        counts[j] = 0;
    }

    for (int j = dim_start[d - 1]; j < n; j++) {
        int count = rand() % 10 * rand() % 10 + 1;
        counts[j] = count;
        int p = parents[j];
        while (p != -1) {
            counts[p] += count;
            p = parents[p];
        }
    }
}

void print_scy_tree(int *parents, int *cells, int *counts, int *dim_start, int *dims, int d, int n) {

    printf("r:  %d/%d\n", cells[0], counts[0]);
    if (d == 0)
        return;

    int *leaf_count = new int[n];

    for (int i = 0; i < n; i++)
        leaf_count[i] = 0;

    for (int i = dim_start[d - 1]; i < n; i++) {
        leaf_count[i] = 0;
        int p = i;
        while (p > 0) {
            leaf_count[p]++;
            p = parents[p];
        }
    }
    for (int i = 0; i < d; i++) {
        printf("%d: ", dims[i]);
        for (int j = dim_start[i]; j < ((i < (d - 1)) ? dim_start[i + 1] : n); j++) {

            if (cells[j] < 100) printf(" ");
            if (cells[j] < 10) printf(" ");
            printf("%d/%d ", cells[j], counts[j]);
            if (counts[j] < 100 && counts[j] > -10) printf(" ");
            if (counts[j] < 10 && counts[j] > -1) printf(" ");

            for (int k = 0; k < leaf_count[j] - 1; k++) {
                printf("        ", cells[j], counts[j]);
            }
        }
        printf("\n");
    }
}

int get_size(int c, int d) {
    int lvl_size = c - c * 1 / 3;
    int prev_count = 0;

    for (int i = 0; i < d; i++) {
        prev_count += lvl_size;
        lvl_size *= 1.5;
    }
    return prev_count;
}

void print_array_range(int *x, int start, int end) {
    for (int i = start; i < end; i++) {
        printf("%d ", (int) x[i]);
    }
    printf("\n\n");
}

void print_array(int *x, int n) {
    int left = 400;
    int right = 400;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            if (x[i] < 10 && x[i] > -1)
                printf(" ");
            if (x[i] < 100 && x[i] > -10)
                printf(" ");
            printf("%d ", (int) x[i]);
        }
    } else {
        for (int i = 0; i < left; i++) {
            printf("%d ", (int) x[i]);
        }
        printf(" ... ");
        for (int i = n - right; i < n; i++) {
            printf("%d ", (int) x[i]);
        }
    }
    printf("\n\n");
}

void print_array(vector<int> x, int n) {
    int left = 400;
    int right = 400;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            printf("%d ", (int) x[i]);
        }
    } else {
        for (int i = 0; i < left; i++) {
            printf("%d ", (int) x[i]);
        }
        printf(" ... ");
        for (int i = n - right; i < n; i++) {
            printf("%d ", (int) x[i]);
        }
    }
    printf("\n\n");
}

void print_array(float *x, int n) {
    int left = 30;
    int right = 10;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            printf("%f ", (float) x[i]);
        }
    } else {
        for (int i = 0; i < left; i++) {
            printf("%f ", (float) x[i]);
        }
        printf(" ... ");
        for (int i = n - right; i < n; i++) {
            printf("%f ", (float) x[i]);
        }
    }
    printf("\n\n");
}

void print_array(thrust::device_vector<int> x, int n) {
    int left = 30;
    int right = 10;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            printf("%d ", (int) x[i]);
        }
    } else {
        for (int i = 0; i < left; i++) {
            printf("%d ", x[i]);
        }
        printf(" ... ");
        for (int i = n - right; i < n; i++) {
            printf("%d ", x[i]);
        }
    }
    printf("\n\n");
}


int get_incorrect(int *array_1, int *array_2, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (array_1[i] != array_2[i]) {
            count++;
        }
    }
    return count;
}


float v_mean(std::vector<float> v) {
    //https://stackoverflow.com/questions/28574346/find-average-of-input-to-vector-c
    return accumulate(v.begin(), v.end(), 0.0) / v.size();
}


vector<float> m_get_col(vector <vector<float>> m, int i) {
    vector<float> col;
    for (int j = 0; j < m.size(); j++) {
        col.push_back(m[j][i]);
    }
    return col;
}

float v_min(std::vector<float> v) {
    float min = std::numeric_limits<float>::infinity();
    for (int i = 0; i < v.size(); i++) {
        if (v[i] < min) {
            min = v[i];
        }
    }
    return min;
}

float v_max(std::vector<float> v) {
    float max = -100000.;
    for (int i = 0; i < v.size(); i++) {
        if (v[i] > max) {
            max = v[i];
        }
    }
    return max;
}

int v_max(std::vector<int> v) {
    int max = -100000;
    for (int i = 0; i < v.size(); i++) {
        if (v[i] > max) {
            max = v[i];
        }
    }
    return max;
}


void m_normalize(std::vector <std::vector<float>> &m) {

    float *min = new float[m[0].size()];
    float *max = new float[m[0].size()];

    for (int j = 0; j < m[0].size(); j++) {
        min[j] = 100000.;
        max[j] = -100000.;
    }

    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[0].size(); j++) {
            min[j] = min[j] < m[i][j] ? min[j] : m[i][j];
            max[j] = max[j] > m[i][j] ? max[j] : m[i][j];
        }
        printf("finding min/max: %d%%\r", int(((i + 1) * 100) / m.size()));
    }
    printf("finding min/max: 100%%\n");

    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[0].size(); j++) {
            m[i][j] = max[j] != min[j] ? (m[i][j] - min[j]) / (max[j] - min[j]) : 0;
        }
        printf("normalizing: %d%%\r", int(((i + 1) * 100) / m.size()));
    }
    printf("normalizing: 100%%\n");
}

template<class T>
vector <T> clone(vector <T> v_old) {
    vector <T> v_clone(v_old);
    return v_clone;
}

void zero(int *array, int n) {
    for (int i = 0; i < n; i++)
        array[i] = 0;
}


bool subspace_of(vector<int> subspace, vector<int> subspace_mark) {
    int i = 0;
    int j = 0;
    while (j < subspace_mark.size() && i < subspace.size()) {
        if (subspace[i] == subspace_mark[j]) {
            i++;
            j++;
        } else {
            j++;
        }
    }
    return i == subspace.size();
}

bool vec_cmp::operator()(const vector<int> &a, const vector<int> &b) const {
    int i = a.size() - 1;
    int j = b.size() - 1;
    while (a[i] == b[j]) {
        i--;
        j--;
        if (i < 0 || j < 0) {
            return i < j;
        }
    }

    return a[i] < b[j];
}

void join(map <vector<int>, vector<int>, vec_cmp> &result, vector<int> &clustering, vector<int> subspace, int min_size,
          float r) {

    int clustering_max = v_max(clustering);
    if (clustering_max < 0) {
        return;
    }

    int n = clustering.size();

    map<int, int> sizes;

    for (int i = 0; i < n; i++) {
        int cluster = clustering[i];
        if (cluster >= 0) {
            if (sizes.count(cluster)) {
                sizes[cluster]++;
            } else {
                sizes.insert(pair<int, int>(cluster, 1));
            }
        }
    }

    for (int i = 0; i < n; i++) {
        int cluster = clustering[i];
        if (cluster >= 0 && sizes[cluster] < min_size) {
            clustering[i] = -1;
        }
    }

    for (pair <vector<int>, vector<int>> subspace_clustering : result) {

        vector<int> subspace_H = subspace_clustering.first;
        vector<int> clustering_H = subspace_clustering.second;

        if (subspace_of(subspace, subspace_H)) {

            map<int, int> sizes_H;
            set<int> to_be_removed;
            for (int cluster_id: clustering_H) {
                if (cluster_id >= 0) {
                    if (sizes_H.count(cluster_id)) {
                        sizes_H[cluster_id]++;
                    } else {
                        sizes_H.insert(pair<int, int>(cluster_id, 1));
                    }
                }
            }

            for (int i = 0; i < n; i++) {
                int cluster = clustering[i];
                int cluster_H = clustering_H[i];
                if (cluster >= 0 && cluster_H >= 0 && sizes[cluster] * r < sizes_H[cluster_H]) {
                    to_be_removed.insert(cluster);
                }
            }

            for (int i = 0; i < n; i++) {
                int cluster = clustering[i];
                if (cluster >= 0 &&
                    to_be_removed.find(cluster) != to_be_removed.end()) {
                    clustering[i] = -1;
                }
            }
        }
    }

    clustering_max = v_max(clustering);
    if (clustering_max >= 0) {
        if (result.count(subspace)) {
            vector<int> clustering_old = result[subspace];
            int m = v_max(clustering_old);
            for (int i = 0; i < n; i++) {
                if (clustering[i] == -2) {
                    clustering_old[i] = clustering[i];
                } else if (clustering[i] >= 0) {
                    clustering_old[i] = m + 1 + clustering[i];
                }
            }
            result[subspace] = clustering_old;
        } else {
            result.insert(pair < vector < int > , vector < int >> (subspace, clustering));
        }
    }
}


__global__
void join_count_kernel(int *d_sizes, int *d_clustering, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int cluster = d_clustering[i];
        if (cluster >= 0) {
            atomicAdd(&d_sizes[cluster], 1);
        }
    }
}

__global__
void join_erease_kernel(int *d_sizes, int *d_clustering, int n, int min_size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int cluster = d_clustering[i];
        if (cluster >= 0 && d_sizes[cluster] < min_size) {
            d_clustering[i] = -1;
        }
    }
}

__global__
void
join_marke_remove_kernel(int *d_to_be_removed, int *d_sizes, int *d_clustering, int *d_sizes_H, int *d_clustering_H,
                         int n, float r) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int cluster = d_clustering[i];
        int cluster_H = d_clustering_H[i];
        if (cluster >= 0 && cluster_H >= 0 && d_sizes[cluster] * r < d_sizes_H[cluster_H]) {
            d_to_be_removed[cluster] = 1;
        }
    }
}

__global__
void join_remove_kernel(int *d_to_be_removed, int *d_sizes, int *d_clustering, int n) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int cluster = d_clustering[i];
        if (cluster >= 0 && d_to_be_removed[cluster]) {
            d_clustering[i] = -1;
        }
    }
}

__global__
void copy_if_positive(int *d_clustering_old, int *d_clustering, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (d_clustering[i] >= 0) {
            d_clustering_old[i] = d_clustering[i];
        }
    }
}

__global__
void v_max_gpu_kernel(int *d_max, int *d_array, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        atomicMax(&d_max[0], d_array[i]);
    }
}

int v_max_gpu(int *d_array, int n) {
    int BLOCK_SIZE = 512;
    int number_of_threads = min(n, BLOCK_SIZE);

    int *d_max;
    cudaMalloc(&d_max, sizeof(int));
    cudaMemset(d_max, -1, sizeof(int));

    v_max_gpu_kernel << < 1, number_of_threads >> > (d_max, d_array, n);

    int h_max;
    cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_max);
    return h_max;
}

int v_max_gpu(int *d_array, int n, TmpMalloc *tmps) {
    int BLOCK_SIZE = 512;
    int number_of_threads = min(n, BLOCK_SIZE);

    int *d_max = tmps->malloc_one();
    cudaMemset(d_max, -1, sizeof(int));

    v_max_gpu_kernel << < 1, number_of_threads >> > (d_max, d_array, n);

    int h_max;
    cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    tmps->free_one(d_max);
    return h_max;
}

void
join_gpu(map<vector<int>, int *, vec_cmp> &result, int *d_clustering,
         vector<int> subspace, int min_size, float r, int n, TmpMalloc *tmps) {
    tmps->reset_counters();

    int BLOCK_SIZE = 512;
    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = min(n, BLOCK_SIZE);

    int *d_sizes = tmps->get_int_array(tmps->int_array_counter++, n);
    cudaMemset(d_sizes, 0, n * sizeof(int));

    join_count_kernel << < 1, number_of_threads >> > (d_sizes, d_clustering, n);

    join_erease_kernel << < number_of_blocks, number_of_threads >> > (d_sizes, d_clustering, n, min_size);


    int *d_clustering_H;
    int *d_sizes_H = tmps->get_int_array(tmps->int_array_counter++, n);
    int *d_to_be_removed = tmps->get_int_array(tmps->int_array_counter++, n);

    for (pair<vector<int>, int *> subspace_clustering : result) {

        vector<int> subspace_H = subspace_clustering.first;
        d_clustering_H = subspace_clustering.second;


        nvtxRangePushA("subspace_of");
        bool sub_of = subspace_of(subspace, subspace_H);
        nvtxRangePop();
        if (sub_of) {

            cudaMemset(d_sizes_H, 0, n * sizeof(int));
            cudaMemset(d_to_be_removed, 0, n * sizeof(int));

            join_count_kernel << < 1, number_of_threads >> > (d_sizes_H, d_clustering_H, n);

            join_marke_remove_kernel << < number_of_blocks, number_of_threads >> >
            (d_to_be_removed, d_sizes, d_clustering, d_sizes_H, d_clustering_H, n, r);

            join_remove_kernel << < number_of_blocks, number_of_threads >> >
            (d_to_be_removed, d_sizes, d_clustering, n);
        }
    }


    int clustering_max = v_max_gpu(d_clustering, n);
    if (clustering_max >= 0) {
        if (result.count(subspace)) {
            int *d_clustering_old = result[subspace];

            copy_if_positive << < number_of_blocks, number_of_threads >> > (d_clustering_old, d_clustering, n);
            tmps->free_points(d_clustering);
            result[subspace] = d_clustering_old;
        } else {
            result.insert(pair < vector < int > , int * > (subspace, d_clustering));
        }
    } else {
        tmps->free_points(d_clustering);
    }
}

void
join_gpu1(map <vector<int>, vector<int>, vec_cmp> &result, vector<int> &clustering, int *d_clustering,
          vector<int> subspace, int min_size,
          float r, int n) {

    int BLOCK_SIZE = 512;
    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = min(n, BLOCK_SIZE);

    int *d_sizes;
    cudaMalloc(&d_sizes, n * sizeof(int));
    cudaMemset(d_sizes, 0, n * sizeof(int));

    join_count_kernel << < 1, number_of_threads >> > (d_sizes, d_clustering, n);

    join_erease_kernel << < number_of_blocks, number_of_threads >> > (d_sizes, d_clustering, n, min_size);

    int *d_subspace_H;
    int *d_clustering_H;
    cudaMalloc(&d_clustering_H, n * sizeof(int));

    int *d_sizes_H;
    cudaMalloc(&d_sizes_H, n * sizeof(int));

    int *d_to_be_removed;
    cudaMalloc(&d_to_be_removed, n * sizeof(int));

    for (pair <vector<int>, vector<int>> subspace_clustering : result) {

        vector<int> subspace_H = subspace_clustering.first;
        vector<int> clustering_H = subspace_clustering.second;
        cudaMemcpy(d_clustering_H, clustering_H.data(), n * sizeof(int), cudaMemcpyHostToDevice);

        if (subspace_of(subspace, subspace_H)) {

            cudaMemset(d_sizes_H, 0, n * sizeof(int));
            cudaMemset(d_to_be_removed, 0, n * sizeof(int));

            join_count_kernel << < 1, number_of_threads >> > (d_sizes_H, d_clustering_H, n);

            join_marke_remove_kernel << < number_of_blocks, number_of_threads >> >
            (d_to_be_removed, d_sizes, d_clustering, d_sizes_H, d_clustering_H, n, r);

            join_remove_kernel << < number_of_blocks, number_of_threads >> >
            (d_to_be_removed, d_sizes, d_clustering, n);
        }
    }
    cudaMemcpy(clustering.data(), d_clustering, n * sizeof(int), cudaMemcpyDeviceToHost);

    int clustering_max = v_max(clustering);
    int clustering_max_other = v_max_gpu(d_clustering, n);
    if (clustering_max != clustering_max_other)
        printf("%d!=%d\n", clustering_max, clustering_max_other);
    if (clustering_max >= 0) {
        if (result.count(subspace)) {
            vector<int> clustering_old = result[subspace];
            int *d_clustering_old;
            cudaMalloc(&d_clustering_old, n * sizeof(int));
            cudaMemcpy(d_clustering_old, clustering_old.data(), n * sizeof(int), cudaMemcpyHostToDevice);

            copy_if_positive << < number_of_blocks, number_of_threads >> > (d_clustering_old, d_clustering, n);
            cudaMemcpy(clustering_old.data(), d_clustering_old, n * sizeof(int), cudaMemcpyDeviceToHost);

            result[subspace] = clustering_old;
        } else {
            result.insert(pair < vector < int > , vector < int >> (subspace, clustering));
        }
    }
}