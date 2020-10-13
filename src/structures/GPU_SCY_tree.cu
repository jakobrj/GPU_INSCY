#include "GPU_SCY_tree.cuh"
#include "../utils/TmpMalloc.cuh"
#include "../utils/util.cuh"

#define BLOCK_WIDTH 64
#define BLOCK_SIZE 512

#define PI 3.141592654f

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__
void memset(int *a, int i, int val) {
    a[i] = val;
}

__global__
void find_dim_i(int *d_dim_i, int *d_dims, int dim_no, int d) {
    for (int i = 0; i < d; i++) {
        if (d_dims[i] == dim_no) {
            d_dim_i[0] = i;
        }
    }
}

__device__
int get_lvl_size_gpu(int *d_dim_start, int dim_i, int number_of_dims, int number_of_nodes) {
    return (dim_i == number_of_dims - 1 ? number_of_nodes : d_dim_start[dim_i + 1]) -
           d_dim_start[dim_i];
}


__device__
float dist_prune_gpu(int p_id, int q_id, float *X, int d, int *subspace, int subsapce_size) {
    float *p = &X[p_id * d];
    float *q = &X[q_id * d];
    float distance = 0;
    for (int i = 0; i < subsapce_size; i++) {
        int d_i = subspace[i];
        float diff = p[d_i] - q[d_i];
        distance += diff * diff;
    }
    return sqrt(distance);
}

__device__
float gamma_prune_gpu(int n) {
    if (n == 2) {
        return 1.;
    } else if (n == 1) {
        return sqrt(PI);
    }
    return (n / 2. - 1.) * gamma_prune_gpu(n - 2);
}

__device__
float c_prune_gpu(int subspace_size) {
    float r = pow(PI, subspace_size / 2.);
    r = r / gamma_prune_gpu(subspace_size + 2);
    return r;
}

__device__
float alpha_prune_gpu(int subspace_size, float neighborhood_size, int n, float v) {
    float r = 2 * n * pow(neighborhood_size, subspace_size) * c_prune_gpu(subspace_size);
    r = r / (pow(v, subspace_size) * (subspace_size + 2));
    return r;
}

__device__
float expDen_prune_gpu(int subspace_size, float neighborhood_size, int n, float v) {
    float r = n * c_prune_gpu(subspace_size) * pow(neighborhood_size, subspace_size);
    r = r / pow(v, subspace_size);
    return r;
}

__device__
float omega_prune_gpu(int subspace_size) {
    return 2.0 / (subspace_size + 2.0);
}


__global__
void check_is_s_connected(int *d_parents, int *d_cells, int *d_counts, int *d_dim_start,
                          int *d_is_included_full, int *d_new_counts_full, int *d_is_s_connected_full,
                          int *d_dim_i_full,
                          int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points) {

    int i = blockIdx.x;

    int dim_i = d_dim_i_full[i];
    int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i, number_of_dims, number_of_nodes);
    int lvl_start = d_dim_start[dim_i];

    for (int j = threadIdx.x; j < lvl_size; j += blockDim.x) {
        int cell_no = d_cells[lvl_start + j];

        int one_offset = i * number_of_cells + cell_no;

        if (d_counts[lvl_start + j] < 0 &&
            (d_parents[lvl_start + j] == 0 || d_counts[d_parents[lvl_start + j]] >= 0)) {
            d_is_s_connected_full[one_offset] = 1;
        }
    }
}

__global__
void compute_merge_map(int *d_is_s_connected_full, int *d_merge_map_full, int number_of_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int one_offset = i * number_of_cells;

    int *d_is_s_connected = d_is_s_connected_full + one_offset;
    int *d_merge_map = d_merge_map_full + one_offset;

    int prev_s_connected = false;
    int prev_cell_no = 0;
    for (int cell_no = 0; cell_no < number_of_cells; cell_no++) {
        if (prev_s_connected) {
            d_merge_map[cell_no] = prev_cell_no;
        } else {
            d_merge_map[cell_no] = cell_no;
        }

        prev_s_connected = d_is_s_connected[cell_no];
        prev_cell_no = d_merge_map[cell_no];
    }
}


__global__
void restrict_merge_dim(int *d_new_parents_full, int *d_parents, int *d_cells, int *d_counts, int *d_dim_start,
                        int *d_is_included_full, int *d_new_counts_full, int *d_is_s_connected_full,
                        int *d_dim_i_full, int *d_merge_map_full,
                        int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points) {

    int i = blockIdx.x;

    int *d_merge_map = d_merge_map_full + i * number_of_cells;

    int dim_i = d_dim_i_full[i];
    int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i, number_of_dims, number_of_nodes);
    int lvl_start = d_dim_start[dim_i];

    for (int j = threadIdx.x; j < lvl_size; j += blockDim.x) {

        int cell_no = d_merge_map[d_cells[lvl_start + j]];
        int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
        int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
        int one_offset = i * number_of_cells + cell_no;

        int *d_is_included = d_is_included_full + node_offset;
        int *d_new_counts = d_new_counts_full + node_offset;
        int *d_new_parents = d_new_parents_full + node_offset;
        int *d_is_s_connected = d_is_s_connected_full + one_offset;

        int count = d_counts[lvl_start + j] > 0 ? d_counts[lvl_start + j] : 0;
        d_is_included[d_parents[lvl_start + j]] = 1;
        d_new_parents[d_parents[lvl_start + j]] = d_parents[d_parents[lvl_start + j]];
        atomicAdd(&d_new_counts[d_parents[lvl_start + j]], count);
    }
}


__global__
void
restrict_dim_prop_up(int *d_new_parents_full, int *d_children_full, int *d_parents, int *d_counts, int *d_cells,
                     int *d_dim_start,
                     int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full,
                     int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points) {

    int i = blockIdx.x;
    int cell_no = blockIdx.y;

    int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;

    int *d_is_included = d_is_included_full + node_offset;
    int *d_new_counts = d_new_counts_full + node_offset;
    int *d_new_parents = d_new_parents_full + node_offset;
    int *d_children = d_children_full
                      + 2 * i * number_of_cells * number_of_cells * number_of_nodes
                      + 2 * cell_no * number_of_nodes * number_of_cells;

    int dim_i = d_dim_i_full[i];

    d_new_parents[0] = 0;

    for (int d_j = dim_i - 1; d_j >= 0; d_j--) {

        int lvl_size = get_lvl_size_gpu(d_dim_start, d_j, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[d_j];

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            atomicMax(&d_is_included[d_parents[n_i]], d_is_included[n_i]);
            atomicAdd(&d_new_counts[d_parents[n_i]],
                      d_new_counts[n_i] > 0 ? d_new_counts[n_i] : 0);
            if (d_counts[n_i] < 0) {
                d_new_counts[n_i] = -1;
            }

            int s_connection = d_counts[n_i] >= 0 ? 0 : 1;
            if (d_is_included[n_i]) {
                d_new_parents[d_parents[n_i]] = d_parents[d_parents[n_i]];
                int cell = d_cells[d_parents[n_i]] >= 0 ? d_cells[d_parents[n_i]] : 0;
                d_children[d_parents[d_parents[n_i]] * number_of_cells * 2 + 2 * cell +
                           s_connection] = n_i;
            }
        }
        __syncthreads();
    }
}


__global__
void
restrict_merge_dim_prop_down_first(int *d_new_parents_full, int *d_children_full, int *d_parents, int *d_counts,
                                   int *d_cells,
                                   int *d_dim_start,
                                   int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full,
                                   int *d_merge_map_full,
                                   int number_of_dims, int number_of_nodes, int number_of_cells,
                                   int number_of_points) {
    int i = blockIdx.x;
    int cell_no = blockIdx.y;


    int *d_merge_map = d_merge_map_full + i * number_of_cells;

    if (cell_no > 0 && d_merge_map[cell_no] == d_merge_map[cell_no - 1]) {
        return;
    }

    int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;

    int *d_is_included = d_is_included_full + node_offset;
    int *d_new_counts = d_new_counts_full + node_offset;
    int *d_new_parents = d_new_parents_full + node_offset;
    int *d_children = d_children_full
                      + 2 * i * number_of_cells * number_of_cells * number_of_nodes
                      + 2 * cell_no * number_of_nodes * number_of_cells;

    int dim_i = d_dim_i_full[i];


    if (dim_i + 1 < number_of_dims) {
        int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i + 1, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[dim_i + 1];

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int new_parent = d_parents[d_parents[n_i]];
            int s_connection = d_counts[n_i] >= 0 ? 0 : 1;

            int is_cell_no = ((d_merge_map[d_cells[d_parents[n_i]]] == cell_no) ? 1 : 0);
            if (is_cell_no && !(d_counts[d_parents[n_i]] < 0 && d_counts[d_parents[d_parents[n_i]]] >= 0)) {
                atomicMax(&d_children[new_parent * number_of_cells * 2 + 2 * d_cells[n_i] + s_connection], n_i);
                d_new_parents[n_i] = new_parent;
            }
        }

        __syncthreads();

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int new_parent = d_new_parents[n_i];
            if (new_parent >= 0) {
                int s_connection = d_counts[n_i] >= 0 ? 0 : 1;
                int n_new = d_children[new_parent * number_of_cells * 2 + 2 * d_cells[n_i] + s_connection];

                if (n_i == n_new) {
                    atomicMax(&d_is_included[n_new], 1);
                }

                if (d_counts[n_i] >= 0) {
                    atomicAdd(&d_new_counts[n_new], d_counts[n_i]);
                } else {
                    d_new_counts[n_new] = -1;
                }
            }
        }
    }
}


__global__
void restrict_dim_prop_down(int *d_new_parents_full, int *d_children_full,
                            int *d_parents, int *d_counts, int *d_cells,
                            int *d_dim_start,
                            int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full,
                            int number_of_dims, int number_of_nodes, int number_of_cells,
                            int number_of_points) {
    int i = blockIdx.x;
    int cell_no = blockIdx.y;

    int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;

    int *d_is_included = d_is_included_full + node_offset;
    int *d_new_counts = d_new_counts_full + node_offset;
    int *d_new_parents = d_new_parents_full + node_offset;
    int *d_children = d_children_full
                      + 2 * i * number_of_cells * number_of_cells * number_of_nodes
                      + 2 * cell_no * number_of_nodes * number_of_cells;


    int dim_i = d_dim_i_full[i];


    for (int d_j = dim_i + 2; d_j < number_of_dims; d_j++) {
        int lvl_size = get_lvl_size_gpu(d_dim_start, d_j, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[d_j];

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int s_connection = d_counts[n_i] >= 0 ? 0 : 1;
            int old_parent = d_parents[n_i];
            int parent_s_connection = d_counts[old_parent] >= 0 ? 0 : 1;
            int new_parent_parent = d_new_parents[old_parent];
            if (new_parent_parent >= 0) {
                int new_parent = d_children[new_parent_parent * number_of_cells * 2 +
                                            2 * d_cells[old_parent] + parent_s_connection];

                if (new_parent >= 0) {
                    d_children[new_parent * number_of_cells * 2 + 2 * d_cells[n_i] + s_connection] = n_i;
                    d_new_parents[n_i] = new_parent;
                }
            }
        }

        __syncthreads();

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int new_parent = d_new_parents[n_i];
            int s_connection = d_counts[n_i] >= 0 ? 0 : 1;
            if (new_parent >= 0) {
                int n_new = d_children[new_parent * number_of_cells * 2 + 2 * d_cells[n_i] + s_connection];
                if (n_i == n_new) {
                    atomicMax(&d_is_included[n_new], d_is_included[new_parent]);
                }

                if (d_counts[n_i] >= 0) {
                    atomicAdd(&d_new_counts[n_new], d_counts[n_i]);
                } else {
                    d_new_counts[n_new] = -1;
                }
            }
        }
        __syncthreads();
    }
}

__global__
void
restrict_move(int *d_new_parents, int *d_cells_1, int *d_cells_2, int *d_parents_1, int *d_parents_2,
              int *d_new_counts, int *d_counts_2,
              int *d_new_indecies, int *d_is_included, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d_is_included[i]) {
        int new_idx = d_new_indecies[i] - 1;
        d_cells_2[new_idx] = d_cells_1[i];
        int new_parent = d_new_parents[i];
        d_parents_2[new_idx] = d_new_indecies[new_parent] - 1;
        d_counts_2[new_idx] = d_new_counts[i];
    }
}

__global__
void restrict_update_dim(int *dim_start_1, int *dims_1, int *dim_start_2, int *dims_2, int *new_indecies,
                         int *d_dim_i,
                         int d_2) {
    int d_i_start = d_dim_i[0];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = j + (d_i_start <= j ? 1 : 0);
    if (j < d_2) {
        int idx = dim_start_1[i] - 1;
        dim_start_2[j] = idx >= 0 ? new_indecies[idx] : 0;
        dims_2[j] = dims_1[i];
    }
}


__global__
void
restrict_update_restricted_dim(int restrict_dim, int *d_restricted_dims_1, int *d_restricted_dims_2,
                               int number_of_restricted_dims_1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_restricted_dims_1)
        d_restricted_dims_2[i] = d_restricted_dims_1[i];
    if (i == number_of_restricted_dims_1)
        d_restricted_dims_2[i] = restrict_dim;
}


__global__
void
restrict_merge_is_points_included(int *d_new_parents, int *d_points_placement, int *d_cells, int *d_is_included,
                                  int *d_is_point_included, int *d_dim_i, int *d_merge_map,
                                  int number_of_dims, int number_of_points, int c_i) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int dim_i = d_dim_i[0];
    bool restricted_dim_is_leaf = (dim_i == number_of_dims - 1);

    if (i >= number_of_points) return;

    int is_included = 0;
    int new_parent = d_new_parents[d_points_placement[i]];
    if (new_parent >= 0)
        is_included = 1;

    if (restricted_dim_is_leaf && d_merge_map[d_cells[d_points_placement[i]]] == c_i) {
        is_included = 1;
    }

    d_is_point_included[i] = is_included;
}


__global__
void
move_points(int *d_new_parents, int *d_children,
            int *d_parents, int *d_cells, int *d_points_1, int *d_points_placement_1,
            int *d_points_2, int *d_points_placement_2,
            int *d_point_new_indecies, int *d_new_indecies,
            int *d_is_point_included, int *d_dim_i,
            int number_of_points, int number_of_dims, int number_of_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int dim_i = d_dim_i[0];
    bool restricted_dim_is_leaf = (dim_i == number_of_dims - 1);

    if (i >= number_of_points) return;

    if (d_is_point_included[i]) {
        int new_parent = d_new_parents[d_points_placement_1[i]];
        int old_parent = d_parents[d_points_placement_1[i]];
        d_points_2[d_point_new_indecies[i] - 1] = d_points_1[i];
        if (restricted_dim_is_leaf) {
            d_points_placement_2[d_point_new_indecies[i] - 1] =
                    d_new_indecies[old_parent] - 1;
        } else {
            int n_i = d_points_placement_1[i];
            int n_new = d_children[new_parent * number_of_cells * 2 + 2 * d_cells[n_i]];
            if (n_new < 0) {
            }
            d_points_placement_2[d_point_new_indecies[i] - 1] = d_new_indecies[n_new] - 1;
        }
    }
}


__global__
void compute_is_weak_dense_prune(int *d_is_dense, int *d_neighborhoods, int *d_neighborhood_end,
                                 int *d_points, int number_of_points,
                                 int *subspace, int subspace_size,
                                 float *X, int n, int d, float F, int num_obj,
                                 float neighborhood_size, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_points) {

        int p_id = d_points[i];

        float p = 0;
        int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
        for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
            int q_id = d_neighborhoods[j];
            if (q_id >= 0) {
                float distance = dist_prune_gpu(p_id, q_id, X, d, subspace, subspace_size) / neighborhood_size;
                float sq = distance * distance;
                p += (1. - sq);
            }
        }
        float a = alpha_prune_gpu(d, neighborhood_size, n, v);
        float w = omega_prune_gpu(d);
        d_is_dense[i] = p >= max(F * a, num_obj * w) ? 1 : 0;
    }
}


__global__
void compute_is_weak_dense_rectangular_prune(int *d_is_dense, int *d_neighborhoods, int *d_neighborhood_end,
                                             int *d_points, int number_of_points,
                                             int *subspace, int subspace_size,
                                             float *X, int n, int d, float F, int num_obj,
                                             float neighborhood_size, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_points) {

        int p_id = d_points[i];
        int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
        int neighbor_count = d_neighborhood_end[p_id] - offset;
        float a = expDen_prune_gpu(d, neighborhood_size, n, v);
        d_is_dense[i] = neighbor_count >= max(F * a, (float) num_obj);
    }
}

__global__
void reset_counts_prune(int *d_counts, int number_of_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_nodes) {
        if (d_counts[i] > 0) {
            d_counts[i] = 0;
        }
    }
}

__global__
void remove_pruned_points_prune(int *d_is_dense, int *d_new_indices,
                                int *d_new_points, int *d_new_point_placement,
                                int *d_points, int *d_point_placement, int number_of_points,
                                int *d_counts, int *d_parents, int number_of_nodes) {
    for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
        if (d_is_dense[i]) {
            int new_i = d_new_indices[i] - 1;
            d_new_points[new_i] = d_points[i];
            d_new_point_placement[new_i] = d_point_placement[i];
            int node = d_point_placement[i];
            atomicAdd(&d_counts[node], 1);
            int count = 0;
            while (d_parents[node] != node) {
                if (node <= d_parents[node]) {
                    break;
                }
                count++;
                node = d_parents[node];
                atomicAdd(&d_counts[node], 1);
            }
        }
    }
}


__global__
void compute_has_child_prune(int *d_has_child, int *d_parents, int *d_cells, int *d_counts, int number_of_nodes,
                             int number_of_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_nodes) {
        if (d_counts[i] > 0) {
            int cell = d_cells[i];
            int parent = d_parents[i];
            if (parent != i) {
                d_has_child[parent * number_of_cells + cell] = 1;
            }
        }
    }
}

__global__
void compute_is_included_prune(int *d_is_included, int *d_has_child,
                               int *d_parents, int *d_cells, int *d_counts, int number_of_nodes, int number_of_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_nodes) {
        int cell = d_cells[i];
        int parent = d_parents[i];
        if (parent == i || d_has_child[parent * number_of_cells + cell]) {
            d_is_included[i] = 1;
        }
    }
}

__global__
void update_point_placement(int *d_new_indices, int *d_points_placement, int number_of_points) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_points) {
        int placement = d_points_placement[i];
        d_points_placement[i] = d_new_indices[placement] - 1;
    }
}

__global__
void remove_nodes(int *d_new_indices, int *d_is_included, int *d_new_parents, int *d_new_cells, int *d_new_counts,
                  int *d_parents, int *d_cells, int *d_counts, int number_of_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_nodes) {
        if (d_is_included[i]) {
            int i_new = d_new_indices[i] - 1;
            int parent = d_parents[i];
            d_new_parents[i_new] = d_new_indices[parent] - 1;
            d_new_cells[i_new] = d_cells[i];
            d_new_counts[i_new] = d_counts[i];
        }
    }
}

__global__
void update_dim_start(int *d_new_indices, int *d_dim_start, int number_of_dims) {
    for (int i = threadIdx.x; i < number_of_dims; i += blockDim.x) {
        int idx = d_dim_start[i] - 1;
        d_dim_start[i] = idx >= 0 ? d_new_indices[idx] : 0;
    }
}


__global__
void prune_count_kernel(int *d_sizes, int *d_clustering, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int cluster = d_clustering[i];
        if (cluster >= 0) {
            atomicAdd(&d_sizes[cluster], 1);
        }
    }
}

__global__
void prune_to_use(int *d_cluster_to_use, int *d_clustering_H, int *d_points, int number_of_points) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < number_of_points; i += blockDim.x * gridDim.x) {
        int p_id = d_points[i];
        int cluster_id = d_clustering_H[p_id];
        if (cluster_id >= 0) {
            d_cluster_to_use[cluster_id] = 1;
        }
    }
}

__global__
void prune_min_cluster(int *d_min_size, int *d_cluster_to_use, int *d_sizes, int *d_clustering, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int size = d_sizes[i];
        if (d_cluster_to_use[i]) {
            atomicCAS(&d_min_size[0], -1, size);
            atomicMin(&d_min_size[0], size);
        }
    }
}


void GPU_SCY_tree::copy_to_device() {
    cudaMemcpy(d_parents, h_parents, sizeof(int) * this->number_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cells, h_cells, sizeof(int) * this->number_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, h_counts, sizeof(int) * this->number_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dim_start, h_dim_start, sizeof(int) * this->number_of_dims, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims, h_dims, sizeof(int) * this->number_of_dims, cudaMemcpyHostToDevice);
    cudaMemcpy(d_points, h_points, sizeof(int) * this->number_of_points, cudaMemcpyHostToDevice);
    cudaMemcpy(d_points_placement, h_points_placement, sizeof(int) * this->number_of_points,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_restricted_dims, h_restricted_dims, sizeof(int) * this->number_of_restricted_dims,
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}


GPU_SCY_tree::GPU_SCY_tree(TmpMalloc *tmps, int number_of_nodes, int number_of_dims, int number_of_restricted_dims,
                           int number_of_points, int number_of_cells, float *mins, float *maxs, float v) {

    this->mins = mins;
    this->maxs = maxs;
    this->v = v;

    this->tmps = tmps;
    this->number_of_nodes = number_of_nodes;
    this->number_of_dims = number_of_dims;
    this->number_of_restricted_dims = number_of_restricted_dims;
    this->number_of_points = number_of_points;
    this->number_of_cells = number_of_cells;
    gpuErrchk(cudaPeekAtLastError());

    this->h_parents = new int[number_of_nodes];
    zero(this->h_parents, number_of_nodes);

    this->h_cells = new int[number_of_nodes];
    zero(this->h_cells, number_of_nodes);

    this->h_counts = new int[number_of_nodes];
    zero(this->h_counts, number_of_nodes);

    this->h_dim_start = new int[number_of_dims];
    zero(this->h_dim_start, number_of_dims);

    this->h_dims = new int[number_of_dims];
    zero(this->h_dims, number_of_dims);

    this->h_points = new int[number_of_points];
    zero(this->h_points, number_of_points);

    this->h_points_placement = new int[number_of_points];
    zero(this->h_points_placement, number_of_points);

    this->h_restricted_dims = new int[number_of_restricted_dims];
    zero(this->h_restricted_dims, number_of_restricted_dims);

    gpuErrchk(cudaPeekAtLastError());

    if (number_of_nodes > 0) {
        this->d_parents = tmps->malloc_nodes();
        gpuErrchk(cudaPeekAtLastError());

        this->d_cells = tmps->malloc_nodes();
        gpuErrchk(cudaPeekAtLastError());

        this->d_counts = tmps->malloc_nodes();
        gpuErrchk(cudaPeekAtLastError());
    }

    if (number_of_dims > 0) {
        this->d_dim_start = tmps->malloc_dims();
        this->d_dims = tmps->malloc_dims();
        gpuErrchk(cudaPeekAtLastError());
    }

    if (number_of_restricted_dims > 0) {
        this->d_restricted_dims = tmps->malloc_dims();
        gpuErrchk(cudaPeekAtLastError());
    }

    if (number_of_points > 0) {
        this->d_points = tmps->malloc_points();
        gpuErrchk(cudaPeekAtLastError());

        this->d_points_placement = tmps->malloc_points();
        gpuErrchk(cudaPeekAtLastError());
    }
}


GPU_SCY_tree::GPU_SCY_tree(int number_of_nodes, int number_of_dims, int number_of_restricted_dims, int number_of_points,
                           int number_of_cells, float *mins, float *maxs, float v) {

    this->mins = mins;
    this->maxs = maxs;
    this->v = v;

    this->number_of_nodes = number_of_nodes;
    this->number_of_dims = number_of_dims;
    this->number_of_restricted_dims = number_of_restricted_dims;
    this->number_of_points = number_of_points;
    this->number_of_cells = number_of_cells;

    this->h_parents = new int[number_of_nodes];
    zero(this->h_parents, number_of_nodes);

    this->h_cells = new int[number_of_nodes];
    zero(this->h_cells, number_of_nodes);

    this->h_counts = new int[number_of_nodes];
    zero(this->h_counts, number_of_nodes);

    this->h_dim_start = new int[number_of_dims];
    zero(this->h_dim_start, number_of_dims);

    this->h_dims = new int[number_of_dims];
    zero(this->h_dims, number_of_dims);

    this->h_points = new int[number_of_points];
    zero(this->h_points, number_of_points);

    this->h_points_placement = new int[number_of_points];
    zero(this->h_points_placement, number_of_points);

    this->h_restricted_dims = new int[number_of_restricted_dims];
    zero(this->h_restricted_dims, number_of_restricted_dims);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    if (number_of_nodes > 0) {
        cudaMalloc(&this->d_parents, number_of_nodes * sizeof(int));
        cudaMemset(this->d_parents, 0, number_of_nodes * sizeof(int));

        cudaMalloc(&this->d_cells, number_of_nodes * sizeof(int));
        cudaMemset(this->d_cells, 0, number_of_nodes * sizeof(int));

        cudaMalloc(&this->d_counts, number_of_nodes * sizeof(int));
        cudaMemset(this->d_counts, 0, number_of_nodes * sizeof(int));

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }

    if (number_of_dims > 0) {
        cudaMalloc(&this->d_dim_start, number_of_dims * sizeof(int));
        cudaMemset(this->d_dim_start, 0, number_of_dims * sizeof(int));

        cudaMalloc(&this->d_dims, number_of_dims * sizeof(int));
        cudaMemset(this->d_dims, 0, number_of_dims * sizeof(int));

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }

    if (number_of_restricted_dims > 0) {
        cudaMalloc(&this->d_restricted_dims, number_of_restricted_dims * sizeof(int));
        cudaMemset(this->d_restricted_dims, 0, number_of_restricted_dims * sizeof(int));

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }

    if (number_of_points > 0) {
        cudaMalloc(&this->d_points, number_of_points * sizeof(int));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemset(this->d_points, 0, number_of_points * sizeof(int));
        gpuErrchk(cudaPeekAtLastError());

        cudaMalloc(&this->d_points_placement, number_of_points * sizeof(int));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemset(this->d_points_placement, 0, number_of_points * sizeof(int));

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }
}

vector <vector<GPU_SCY_tree *>>
GPU_SCY_tree::restrict_merge(TmpMalloc *tmps, int first_dim_no, int number_of_dims, int number_of_cells) {
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaPeekAtLastError());

    GPU_SCY_tree *scy_tree = this;

    tmps->reset_counters();


    int number_of_blocks;
    dim3 block(128);
    dim3 grid(number_of_dims, number_of_cells);

    int c = scy_tree->number_of_cells;
    int d = scy_tree->number_of_dims;

    int total_number_of_dim = first_dim_no + number_of_dims;
    int number_of_restrictions = number_of_dims * number_of_cells;

    vector <vector<GPU_SCY_tree *>> L(number_of_dims);

    vector <vector<GPU_SCY_tree *>> L_merged(number_of_dims);

    if (scy_tree->number_of_nodes * number_of_restrictions == 0)
        return L_merged;
    gpuErrchk(cudaPeekAtLastError());

    int *d_new_indecies = tmps->get_int_array(tmps->int_array_counter++, scy_tree->number_of_nodes *
                                                                         number_of_restrictions);
    gpuErrchk(cudaPeekAtLastError());
    int *d_new_counts = tmps->get_int_array(tmps->int_array_counter++,
                                            scy_tree->number_of_nodes * number_of_restrictions);
    gpuErrchk(cudaPeekAtLastError());
    int *d_is_included = tmps->get_int_array(tmps->int_array_counter++,
                                             scy_tree->number_of_nodes * number_of_restrictions);
    gpuErrchk(cudaPeekAtLastError());
    int *d_children_full = tmps->get_int_array(tmps->int_array_counter++,
                                               2 * scy_tree->number_of_nodes * number_of_restrictions *
                                               scy_tree->number_of_cells);
    gpuErrchk(cudaPeekAtLastError());

    int *d_parents_full = tmps->get_int_array(tmps->int_array_counter++,
                                              scy_tree->number_of_nodes * number_of_restrictions);
    gpuErrchk(cudaPeekAtLastError());

    cudaMemset(d_new_indecies, 0, scy_tree->number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMemset(d_new_counts, 0, scy_tree->number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMemset(d_is_included, 0, scy_tree->number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMemset(d_parents_full, -1, scy_tree->number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMemset(d_children_full, -1,
               2 * scy_tree->number_of_nodes * number_of_restrictions * scy_tree->number_of_cells * sizeof(int));
    for (int i = 0; i < number_of_dims; i++) {
        for (int cell_no = 0; cell_no < number_of_cells; cell_no++) {
            int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
            memset << < 1, 1 >> > (d_is_included + node_offset, 0, 1);
        }
    }
    gpuErrchk(cudaPeekAtLastError());

    int *d_is_point_included = tmps->get_int_array(tmps->int_array_counter++, this->number_of_points *
                                                                              number_of_restrictions);
    gpuErrchk(cudaPeekAtLastError());
    int *d_point_new_indecies = tmps->get_int_array(tmps->int_array_counter++, this->number_of_points *
                                                                               number_of_restrictions);
    gpuErrchk(cudaPeekAtLastError());

    cudaMemset(d_is_point_included, 0, this->number_of_points * number_of_restrictions * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());

    int *d_is_s_connected = tmps->get_int_array(tmps->int_array_counter++,
                                                number_of_restrictions);
    cudaMemset(d_is_s_connected, 0, number_of_restrictions * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());

    int *d_dim_i = tmps->get_int_array(tmps->int_array_counter++, number_of_dims);

    gpuErrchk(cudaPeekAtLastError());

    int *h_new_number_of_points = new int[number_of_restrictions];
    int *h_new_number_of_nodes = new int[number_of_restrictions];

    int *d_merge_map = tmps->get_int_array(tmps->int_array_counter++, number_of_restrictions);
    int *h_merge_map = new int[number_of_restrictions];

    int dim_no = first_dim_no;
    while (dim_no < total_number_of_dim) {
        int i = dim_no - first_dim_no;
        L[i] = vector<GPU_SCY_tree *>(number_of_cells);

        find_dim_i << < 1, 1 >> >
        (d_dim_i + i, scy_tree->d_dims, dim_no, scy_tree->number_of_dims);
        dim_no++;
    }

    if (number_of_dims > 0) {

        check_is_s_connected << < number_of_dims, block >> >
        (scy_tree->d_parents, scy_tree->d_cells, scy_tree->d_counts, scy_tree->d_dim_start,
                d_is_included, d_new_counts, d_is_s_connected, d_dim_i,
                scy_tree->number_of_dims, scy_tree->number_of_nodes,
                scy_tree->number_of_cells, scy_tree->number_of_points);
        gpuErrchk(cudaPeekAtLastError());

        compute_merge_map << < 1, number_of_dims >> >
        (d_is_s_connected, d_merge_map, scy_tree->number_of_cells);
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpy(h_merge_map, d_merge_map, number_of_restrictions * sizeof(int), cudaMemcpyDeviceToHost);
        gpuErrchk(cudaPeekAtLastError());

        restrict_merge_dim << < number_of_dims, block >> >
        (d_parents_full, scy_tree->d_parents, scy_tree->d_cells, scy_tree->d_counts, scy_tree->d_dim_start,
                d_is_included, d_new_counts, d_is_s_connected, d_dim_i, d_merge_map,
                scy_tree->number_of_dims, scy_tree->number_of_nodes,
                scy_tree->number_of_cells, scy_tree->number_of_points);
        gpuErrchk(cudaPeekAtLastError());

        restrict_dim_prop_up << < grid, block >> >
        (d_parents_full, d_children_full, scy_tree->d_parents, scy_tree->d_counts, scy_tree->d_cells, scy_tree->d_dim_start,
                d_is_included, d_new_counts, d_dim_i,
                scy_tree->number_of_dims, scy_tree->number_of_nodes,
                scy_tree->number_of_cells, scy_tree->number_of_points);

        gpuErrchk(cudaPeekAtLastError());

        restrict_merge_dim_prop_down_first << < grid, block >> >
        (d_parents_full, d_children_full, scy_tree->d_parents, scy_tree->d_counts, scy_tree->d_cells, scy_tree->d_dim_start,
                d_is_included, d_new_counts, d_dim_i, d_merge_map,
                scy_tree->number_of_dims, scy_tree->number_of_nodes,
                scy_tree->number_of_cells, scy_tree->number_of_points);

        gpuErrchk(cudaPeekAtLastError());

        restrict_dim_prop_down << < grid, block >> >
        (d_parents_full, d_children_full, scy_tree->d_parents, scy_tree->d_counts, scy_tree->d_cells, scy_tree->d_dim_start,
                d_is_included, d_new_counts, d_dim_i,
                scy_tree->number_of_dims, scy_tree->number_of_nodes,
                scy_tree->number_of_cells, scy_tree->number_of_points);

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        dim_no = first_dim_no;
        while (dim_no < total_number_of_dim) {
            int i = dim_no - first_dim_no;
            int cell_no = 0;
            while (cell_no < number_of_cells) {
                int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
                int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
                int one_offset = i * number_of_cells + cell_no;

                if (cell_no == 0 || h_merge_map[one_offset - 1] != h_merge_map[one_offset]) {
                    // 2. do a scan to find the new indices for the nodes in the restricted tree
                    cudaDeviceSynchronize();
                    gpuErrchk(cudaPeekAtLastError());
                    inclusive_scan_nodes(d_is_included + node_offset, d_new_indecies + node_offset,
                                         scy_tree->number_of_nodes, tmps);

                    // 3. construct restricted tree
                    gpuErrchk(cudaPeekAtLastError());

                    int new_number_of_points = 0;
                    cudaMemcpy(&new_number_of_points, d_new_counts + node_offset, sizeof(int),
                               cudaMemcpyDeviceToHost);
                    gpuErrchk(cudaPeekAtLastError());

                    int new_number_of_nodes = 0;
                    cudaMemcpy(&new_number_of_nodes, d_new_indecies + node_offset + scy_tree->number_of_nodes - 1,
                               sizeof(int), cudaMemcpyDeviceToHost);
                    gpuErrchk(cudaPeekAtLastError());

                    cudaDeviceSynchronize();
                    gpuErrchk(cudaPeekAtLastError());

                    float ra = this->maxs[dim_no] - this->mins[dim_no];
                    GPU_SCY_tree *restricted_scy_tree = new GPU_SCY_tree(tmps, new_number_of_nodes,
                                                                         scy_tree->number_of_dims - 1,
                                                                         scy_tree->number_of_restricted_dims + 1,
                                                                         new_number_of_points,
                                                                         scy_tree->number_of_cells, mins, maxs,
                                                                         scy_tree->v * ra);
                    gpuErrchk(cudaPeekAtLastError());

                    L[i][cell_no] = restricted_scy_tree;
                    L_merged[i].push_back(restricted_scy_tree);

                    restricted_scy_tree->is_s_connected = false;

                    cudaDeviceSynchronize();
                    gpuErrchk(cudaPeekAtLastError());
                }
                cell_no++;
            }
            dim_no++;
        }

        dim_no = first_dim_no;
        while (dim_no < total_number_of_dim) {
            int i = dim_no - first_dim_no;
            int cell_no = 0;
            while (cell_no < number_of_cells) {
                int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
                int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
                int one_offset = i * number_of_cells + cell_no;
                if (cell_no == 0 || h_merge_map[one_offset - 1] != h_merge_map[one_offset]) {
                    GPU_SCY_tree *restricted_scy_tree = L[i][cell_no];

                    number_of_blocks = scy_tree->number_of_nodes / BLOCK_WIDTH;
                    if (scy_tree->number_of_nodes % BLOCK_WIDTH) number_of_blocks++;
                    restrict_move<<< number_of_blocks, BLOCK_WIDTH, 0 >>>
                            (d_parents_full + node_offset, scy_tree->d_cells, restricted_scy_tree->d_cells,
                             scy_tree->d_parents, restricted_scy_tree->d_parents, d_new_counts + node_offset,
                             restricted_scy_tree->d_counts, d_new_indecies + node_offset,
                             d_is_included + node_offset, scy_tree->number_of_nodes);
                }
                cell_no++;
            }
            dim_no++;
        }
        gpuErrchk(cudaPeekAtLastError());

        dim_no = first_dim_no;
        while (dim_no < total_number_of_dim) {
            int i = dim_no - first_dim_no;
            int cell_no = 0;
            while (cell_no < number_of_cells) {
                int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
                int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
                int one_offset = i * number_of_cells + cell_no;
                if (cell_no == 0 || h_merge_map[one_offset - 1] != h_merge_map[one_offset]) {
                    GPU_SCY_tree *restricted_scy_tree = L[i][cell_no];

                    if (scy_tree->number_of_dims > 1) {

                        number_of_blocks = restricted_scy_tree->number_of_dims / BLOCK_WIDTH;
                        if (restricted_scy_tree->number_of_dims % BLOCK_WIDTH) number_of_blocks++;

                        restrict_update_dim << < number_of_blocks, BLOCK_WIDTH, 0 >> >
                        (scy_tree->d_dim_start, scy_tree->d_dims,
                                restricted_scy_tree->d_dim_start,
                                restricted_scy_tree->d_dims,
                                d_new_indecies +
                                node_offset,
                                d_dim_i +
                                i, restricted_scy_tree->number_of_dims);

                    }
                }
                cell_no++;
            }
            dim_no++;
        }
        gpuErrchk(cudaPeekAtLastError());

        dim_no = first_dim_no;
        while (dim_no < total_number_of_dim) {
            int i = dim_no - first_dim_no;
            int cell_no = 0;
            while (cell_no < number_of_cells) {
                int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
                int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
                int one_offset = i * number_of_cells + cell_no;
                if (cell_no == 0 || h_merge_map[one_offset - 1] != h_merge_map[one_offset]) {
                    GPU_SCY_tree *restricted_scy_tree = L[i][cell_no];

                    number_of_blocks = restricted_scy_tree->number_of_restricted_dims / BLOCK_WIDTH;
                    if (restricted_scy_tree->number_of_restricted_dims % BLOCK_WIDTH) number_of_blocks++;
                    restrict_update_restricted_dim << < number_of_blocks, BLOCK_WIDTH, 0 >> >
                    (dim_no, scy_tree->d_restricted_dims, restricted_scy_tree->d_restricted_dims, scy_tree->number_of_restricted_dims);

                    for (int k = 0; k < scy_tree->number_of_restricted_dims; k++) {
                        restricted_scy_tree->h_restricted_dims[k] = scy_tree->h_restricted_dims[k];
                    }
                    restricted_scy_tree->h_restricted_dims[scy_tree->number_of_restricted_dims] = dim_no;
                }
                cell_no++;
            }
            dim_no++;
        }
        gpuErrchk(cudaPeekAtLastError());

        dim_no = first_dim_no;
        while (dim_no < total_number_of_dim) {
            int i = dim_no - first_dim_no;
            int cell_no = 0;
            while (cell_no < number_of_cells) {
                int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
                int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
                int one_offset = i * number_of_cells + cell_no;

                if (cell_no == 0 || h_merge_map[one_offset - 1] != h_merge_map[one_offset]) {
                    GPU_SCY_tree *restricted_scy_tree = L[i][cell_no];

                    number_of_blocks = number_of_points / BLOCK_WIDTH;
                    if (number_of_points % BLOCK_WIDTH) number_of_blocks++;
                    restrict_merge_is_points_included
                    <<< number_of_blocks, BLOCK_WIDTH, 0 >>>
                            (d_parents_full + node_offset, scy_tree->d_points_placement, scy_tree->d_cells,
                             d_is_included + node_offset,
                             d_is_point_included + point_offset,
                             d_dim_i + i,
                             d_merge_map + i * number_of_cells,
                             scy_tree->number_of_dims, scy_tree->number_of_points, cell_no);

                }
                cell_no++;
            }
            dim_no++;
        }
        gpuErrchk(cudaPeekAtLastError());

        dim_no = first_dim_no;
        while (dim_no < total_number_of_dim) {
            int i = dim_no - first_dim_no;
            int cell_no = 0;
            while (cell_no < number_of_cells) {
                int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
                int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
                int one_offset = i * number_of_cells + cell_no;

                if (cell_no == 0 || h_merge_map[one_offset - 1] != h_merge_map[one_offset]) {
                    GPU_SCY_tree *restricted_scy_tree = L[i][cell_no];

                    inclusive_scan_points(d_is_point_included + point_offset,
                                          d_point_new_indecies + point_offset,
                                          number_of_points, tmps);
                }
                cell_no++;
            }
            dim_no++;
        }
        gpuErrchk(cudaPeekAtLastError());

        dim_no = first_dim_no;
        while (dim_no < total_number_of_dim) {
            int i = dim_no - first_dim_no;
            int cell_no = 0;
            while (cell_no < number_of_cells) {
                int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
                int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
                int one_offset = i * number_of_cells + cell_no;
                if (cell_no == 0 || h_merge_map[one_offset - 1] != h_merge_map[one_offset]) {
                    GPU_SCY_tree *restricted_scy_tree = L[i][cell_no];

                    if (restricted_scy_tree->number_of_points > 0) {


                        move_points <<< number_of_blocks, BLOCK_WIDTH, 0 >>>
                                (d_parents_full + node_offset, d_children_full
                                                               + 2 * i * number_of_cells * number_of_cells *
                                                                 number_of_nodes
                                                               + 2 * cell_no * number_of_nodes * number_of_cells,
                                 scy_tree->d_parents, scy_tree->d_cells,
                                 scy_tree->d_points, scy_tree->d_points_placement, restricted_scy_tree->d_points,
                                 restricted_scy_tree->d_points_placement,
                                 d_point_new_indecies +
                                 point_offset,
                                 d_new_indecies + node_offset,
                                 d_is_point_included + point_offset,
                                 d_dim_i + i,
                                 number_of_points, scy_tree->number_of_dims, scy_tree->number_of_cells);
                        gpuErrchk(cudaPeekAtLastError());

                    }
                }
                cell_no++;
            }
            dim_no++;
        }
        gpuErrchk(cudaPeekAtLastError());
    }

    delete[] h_new_number_of_points;
    delete[] h_new_number_of_nodes;
    delete[] h_merge_map;

    return L_merged;
}

bool
GPU_SCY_tree::pruneRecursion(TmpMalloc *tmps, int min_size, float *d_X, int n, int d,
                             float neighborhood_size, float F,
                             int num_obj, int *d_neighborhoods, int *d_neighborhood_end,
                             bool rectangular) {


    if (this->number_of_points < min_size) {
        return false;
    }
    int blocks_points = this->number_of_points / 512;
    if (this->number_of_points % 512) blocks_points++;
    int blocks_nodes = this->number_of_nodes / 512;
    if (this->number_of_nodes % 512) blocks_nodes++;

    int *d_is_dense = tmps->malloc_points();
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    int *d_new_indices = tmps->malloc_points();
    gpuErrchk(cudaPeekAtLastError());
    cudaMemset(d_new_indices, 0, sizeof(int) * this->number_of_points);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    if (rectangular) {
        compute_is_weak_dense_rectangular_prune <<< blocks_points, min(512, this->number_of_points) >>>(d_is_dense,
                                                                                                        d_neighborhoods,
                                                                                                        d_neighborhood_end,
                                                                                                        this->d_points,
                                                                                                        this->number_of_points,
                                                                                                        this->d_restricted_dims,
                                                                                                        this->number_of_restricted_dims,
                                                                                                        d_X, n, d,
                                                                                                        F, num_obj,
                                                                                                        neighborhood_size,
                                                                                                        this->v);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    } else {
        compute_is_weak_dense_prune <<< blocks_points, min(512, this->number_of_points) >>>(d_is_dense, d_neighborhoods,
                                                                                            d_neighborhood_end,
                                                                                            this->d_points,
                                                                                            this->number_of_points,
                                                                                            this->d_restricted_dims,
                                                                                            this->number_of_restricted_dims,
                                                                                            d_X, n, d,
                                                                                            F, num_obj,
                                                                                            neighborhood_size, this->v);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }

    inclusive_scan_points(d_is_dense, d_new_indices, this->number_of_points, tmps);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    int new_number_of_points;
    cudaMemcpy(&new_number_of_points, d_new_indices + this->number_of_points - 1, sizeof(int), cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());

    if (new_number_of_points == 0) {
        tmps->free_points(d_is_dense);
        tmps->free_points(d_new_indices);
        return false;
    }

    int *d_new_points = tmps->malloc_points();
    gpuErrchk(cudaPeekAtLastError());

    int *d_new_point_placement = tmps->malloc_points();
    gpuErrchk(cudaPeekAtLastError());

    reset_counts_prune<<<blocks_nodes, min(512, this->number_of_nodes)>>>(this->d_counts, this->number_of_nodes);
    gpuErrchk(cudaPeekAtLastError());

    remove_pruned_points_prune <<< 1, min(512, this->number_of_points) >>>(d_is_dense, d_new_indices,
                                                                           d_new_points, d_new_point_placement,
                                                                           this->d_points, this->d_points_placement,
                                                                           this->number_of_points,
                                                                           this->d_counts, this->d_parents,
                                                                           this->number_of_nodes);
    gpuErrchk(cudaPeekAtLastError());


    tmps->free_points(this->d_points);
    tmps->free_points(this->d_points_placement);
    tmps->free_points(d_is_dense);
    tmps->free_points(d_new_indices);
    gpuErrchk(cudaPeekAtLastError());

    this->d_points = d_new_points;
    this->d_points_placement = d_new_point_placement;
    this->number_of_points = new_number_of_points;

    gpuErrchk(cudaPeekAtLastError());


    int *d_is_included = tmps->malloc_nodes();
    gpuErrchk(cudaPeekAtLastError());
    cudaMemset(d_is_included, 0, sizeof(int) * this->number_of_nodes);
    gpuErrchk(cudaPeekAtLastError());

    d_new_indices = tmps->malloc_nodes();
    gpuErrchk(cudaPeekAtLastError());
    cudaMemset(d_new_indices, 0, sizeof(int) * this->number_of_nodes);
    gpuErrchk(cudaPeekAtLastError());

    int *d_has_child = tmps->get_int_array(tmps->int_array_counter++, this->number_of_nodes * this->number_of_cells);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemset(d_has_child, 0, sizeof(int) * this->number_of_nodes * this->number_of_cells);
    gpuErrchk(cudaPeekAtLastError());

    compute_has_child_prune << < blocks_nodes, min(512, this->number_of_nodes) >> > (d_has_child,
            this->d_parents, this->d_cells, this->d_counts, this->number_of_nodes, this->number_of_cells);

    gpuErrchk(cudaPeekAtLastError());

    compute_is_included_prune << < blocks_nodes, min(512, this->number_of_nodes) >> > (d_is_included, d_has_child,
            this->d_parents, this->d_cells, this->d_counts, this->number_of_nodes, this->number_of_cells);

    gpuErrchk(cudaPeekAtLastError());

    inclusive_scan_nodes(d_is_included, d_new_indices, this->number_of_nodes, tmps);

    gpuErrchk(cudaPeekAtLastError());

    int new_number_of_nodes;
    cudaMemcpy(&new_number_of_nodes, d_new_indices + this->number_of_nodes - 1, sizeof(int), cudaMemcpyDeviceToHost);

    gpuErrchk(cudaPeekAtLastError());
    if (new_number_of_nodes <= 0) {
        tmps->free_nodes(d_is_included);
        tmps->free_nodes(d_new_indices);
        return false;
    }

    int *d_new_parents = tmps->malloc_nodes();
    int *d_new_cells = tmps->malloc_nodes();
    int *d_new_counts = tmps->malloc_nodes();
    gpuErrchk(cudaPeekAtLastError());

    blocks_points = this->number_of_points / 512;
    if (this->number_of_points % 512) blocks_points++;
    update_point_placement << < blocks_points, min(512, this->number_of_points) >> >
    (d_new_indices, this->d_points_placement, this->number_of_points);

    gpuErrchk(cudaPeekAtLastError());

    remove_nodes << < blocks_nodes, min(512, this->number_of_nodes) >> >
    (d_new_indices, d_is_included, d_new_parents, d_new_cells, d_new_counts,
            this->d_parents, this->d_cells, this->d_counts, this->number_of_nodes);

    gpuErrchk(cudaPeekAtLastError());
    tmps->free_nodes(this->d_parents);
    tmps->free_nodes(this->d_cells);
    tmps->free_nodes(this->d_counts);
    gpuErrchk(cudaPeekAtLastError());


    this->d_parents = d_new_parents;
    this->d_cells = d_new_cells;
    this->d_counts = d_new_counts;
    this->number_of_nodes = new_number_of_nodes;

    if (this->number_of_dims > 0) {

        update_dim_start << < 1, min(512, this->number_of_dims) >> >
        (d_new_indices, this->d_dim_start, this->number_of_dims);

        gpuErrchk(cudaPeekAtLastError());
    }

    tmps->free_nodes(d_is_included);
    tmps->free_nodes(d_new_indices);

    return this->number_of_points >= min_size;
}


bool GPU_SCY_tree::pruneRedundancy(float r, map<vector<int>, int *, vec_cmp> result, int n, TmpMalloc *tmps) {

    tmps->reset_counters();

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = min(n, BLOCK_SIZE);

    int max_min_size = 0;

    vector<int> subspace(this->h_restricted_dims, this->h_restricted_dims +
                                                  this->number_of_restricted_dims);
    vector<int> max_min_subspace;

    int *d_clustering_H;

    int *d_sizes_H = tmps->get_int_array(tmps->int_array_counter++, n);

    int *d_cluster_to_use = tmps->get_int_array(tmps->int_array_counter++, n);
    int *d_min_size = tmps->malloc_one();

    for (std::pair<vector<int>, int *> subspace_clustering : result) {

        // find sizes of clusters
        vector<int> subspace_mark = subspace_clustering.first;

        if (subspace_of(subspace, subspace_mark)) {

            d_clustering_H = subspace_clustering.second;
            cudaMemset(d_sizes_H, 0, n * sizeof(int));
            cudaMemset(d_cluster_to_use, 0, n * sizeof(int));
            prune_count_kernel << < 1, number_of_threads >> > (d_sizes_H, d_clustering_H, n);

            prune_to_use << < number_of_blocks, number_of_threads >> >
            (d_cluster_to_use, d_clustering_H, d_points, this->number_of_points);

            // find the minimum size for each subspace
            cudaMemset(d_min_size, -1, sizeof(int));
            int min_size;
            prune_min_cluster << < 1, number_of_threads >> >
            (d_min_size, d_cluster_to_use, d_sizes_H, d_clustering_H, n);
            cudaMemcpy(&min_size, d_min_size, sizeof(int), cudaMemcpyDeviceToHost);

            // find the maximum minimum size for each subspace
            if (min_size > max_min_size) {
                max_min_size = min_size;
                max_min_subspace = subspace_mark;
            }
        }
    }

    tmps->free_one(d_min_size);

    if (max_min_size == 0) {
        return true;
    }

    return this->number_of_points * r > max_min_size * 1.;
}


GPU_SCY_tree::~GPU_SCY_tree() {
    if (!freed_partial) {
        if (number_of_nodes > 0) {
            if (tmps == nullptr) {
                cudaFree(d_parents);
                cudaFree(d_cells);
                cudaFree(d_counts);
            } else {
                tmps->free_nodes(d_parents);
                tmps->free_nodes(d_cells);
                tmps->free_nodes(d_counts);
            }
            delete[] h_parents;
            delete[] h_cells;
            delete[] h_counts;
        }
        if (number_of_dims > 0) {
            if (tmps == nullptr) {
                cudaFree(d_dim_start);
                cudaFree(d_dims);
            } else {
                tmps->free_dims(d_dim_start);
                tmps->free_dims(d_dims);
            }
            delete[] h_dim_start;
            delete[] h_dims;
        }
        if (number_of_points > 0) {
            if (tmps == nullptr) {
                cudaFree(d_points_placement);
            } else {
                tmps->free_points(d_points_placement);
            }
            delete[] h_points_placement;
        }
    }
    if (number_of_restricted_dims > 0) {
        if (tmps == nullptr) {
            cudaFree(d_restricted_dims);
        } else {
            tmps->free_dims(d_restricted_dims);
        }
        delete[] h_restricted_dims;
    }
    if (number_of_points > 0) {
        if (tmps == nullptr) {
            cudaFree(d_points);
        } else {
            tmps->free_points(d_points);
        }
        delete[] h_points;
    }
}