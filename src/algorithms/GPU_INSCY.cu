#include "GPU_INSCY.cuh"
#include "GPU_Clustering.cuh"
#include "../structures/GPU_SCY_tree.cuh"
#include "../utils/util.cuh"
#include "../utils/TmpMalloc.cuh"

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void free_tree(TmpMalloc *tmps, GPU_SCY_tree *&restricted_scy_tree, int *&d_new_neighborhoods,
               int *&d_new_neighborhood_end) {
    cudaFree(d_new_neighborhoods);
    tmps->free_points(d_new_neighborhood_end);
    delete restricted_scy_tree;
}

void GPU_INSCY(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, GPU_SCY_tree *scy_tree,
               float *d_X, int n, int d, float neighborhood_size, float F, int num_obj,
               int min_size, map<vector<int>, int *, vec_cmp> &result, int first_dim_no,
               int total_number_of_dim, float r, int &calls, bool rectangular) {
    calls++;
    printf("GPU_INSCY(%d)\r", calls);

    int number_of_dims = total_number_of_dim - first_dim_no;
    int number_of_cells = scy_tree->number_of_cells;

    vector <vector<GPU_SCY_tree *>> L_merged = scy_tree->restrict_merge(tmps, first_dim_no, number_of_dims,
                                                                                   number_of_cells);

    vector < GPU_SCY_tree * > restricted_scy_tree_list;
    vector<int *> clustering_list;
    vector<int *> new_neighborhoods_list;
    vector<int *> new_neighborhood_end_list;

    pair<int **, int **> p = find_neighborhoods(tmps, d_neighborhoods, d_neighborhood_end,
                                                    d_X, n, d, scy_tree, L_merged, neighborhood_size);


    int **hd_new_neighborhoods_list = p.first;
    int **hd_new_neighborhood_end_list = p.second;


    int j = 0;
    int dim_no = first_dim_no;
    while (dim_no < total_number_of_dim) {

        int *d_clustering = tmps->malloc_points();
        cudaMemset(d_clustering, -1, sizeof(int) * n);

        bool clustering_used = false;

        int i = dim_no - first_dim_no;
        for (GPU_SCY_tree *restricted_scy_tree : L_merged[i]) {

            int *d_new_neighborhoods = hd_new_neighborhoods_list[j];
            int *d_new_neighborhood_end = hd_new_neighborhood_end_list[j];
            j++;

            bool pruneRecursion = restricted_scy_tree->pruneRecursion(tmps, min_size, d_X, n, d,
                                                                                    neighborhood_size, F, num_obj,
                                                                                    d_new_neighborhoods,
                                                                                    d_new_neighborhood_end,
                                                                                    rectangular);

            if (pruneRecursion) {
                GPU_INSCY(d_new_neighborhoods, d_new_neighborhood_end, tmps, restricted_scy_tree,
                          d_X, n, d, neighborhood_size, F, num_obj, min_size,
                          result, dim_no + 1, total_number_of_dim, r, calls, rectangular);

                bool pruneRedundancy = restricted_scy_tree->pruneRedundancy(r, result, n, tmps);
                if (pruneRedundancy) {

                    restricted_scy_tree_list.push_back(restricted_scy_tree);
                    clustering_list.push_back(d_clustering);
                    clustering_used = true;
                    new_neighborhoods_list.push_back(d_new_neighborhoods);
                    new_neighborhood_end_list.push_back(d_new_neighborhood_end);

                } else {
                    free_tree(tmps, restricted_scy_tree, d_new_neighborhoods, d_new_neighborhood_end);
                }
            } else {
                free_tree(tmps, restricted_scy_tree, d_new_neighborhoods, d_new_neighborhood_end);
            }
        }

        dim_no++;

        if (!clustering_used)
            tmps->free_points(d_clustering);
    }

    GPU_Clustering(new_neighborhoods_list, new_neighborhood_end_list, tmps,
                   clustering_list, restricted_scy_tree_list,
                   d_X, n, d, neighborhood_size,
                   F, num_obj, rectangular);

    for (int i = 0; i < restricted_scy_tree_list.size(); i++) {
        GPU_SCY_tree *restricted_scy_tree = restricted_scy_tree_list[i];
        int *d_clustering = clustering_list[i];
        int *d_new_neighborhoods = new_neighborhoods_list[i];
        int *d_new_neighborhood_end = new_neighborhood_end_list[i];

        if (i == restricted_scy_tree_list.size() - 1 ||
            (i < restricted_scy_tree_list.size() - 1 && d_clustering != clustering_list[i + 1])) {

            vector<int> subspace = vector<int>(restricted_scy_tree->h_restricted_dims,
                                               restricted_scy_tree->h_restricted_dims +
                                               restricted_scy_tree->number_of_restricted_dims);

            join_gpu(result, d_clustering, subspace, min_size, r, n, tmps);
        }
        free_tree(tmps, restricted_scy_tree, d_new_neighborhoods, d_new_neighborhood_end);
    }
}


void GPU_INSCY_star(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, GPU_SCY_tree *scy_tree,
                    float *d_X, int n, int d, float neighborhood_size, float F, int num_obj,
                    int min_size, map<vector<int>, int *, vec_cmp> &result, int first_dim_no,
                    int total_number_of_dim, float r, int &calls, bool rectangular) {
    calls++;
    printf("GPU_INSCY_star(%d)\r", calls);

    int number_of_dims = total_number_of_dim - first_dim_no;
    int number_of_cells = scy_tree->number_of_cells;

    vector <vector<GPU_SCY_tree *>> L_merged = scy_tree->restrict_merge(tmps, first_dim_no, number_of_dims,
                                                                                   number_of_cells);

    vector < GPU_SCY_tree * > restricted_scy_tree_list;
    vector<int *> clustering_list;
    vector<int *> new_neighborhoods_list;
    vector<int *> new_neighborhood_end_list;

    pair<int **, int **> p = find_neighborhoods_star(tmps, d_neighborhoods, d_neighborhood_end,
                                                        d_X, n, d, scy_tree, L_merged, neighborhood_size);

    int **hd_new_neighborhoods_list = p.first;
    int **hd_new_neighborhood_end_list = p.second;


    int j = 0;
    int dim_no = first_dim_no;
    while (dim_no < total_number_of_dim) {

        int *d_clustering = tmps->malloc_points();
        cudaMemset(d_clustering, -1, sizeof(int) * n);

        bool clustering_used = false;

        int i = dim_no - first_dim_no;
        for (GPU_SCY_tree *restricted_scy_tree : L_merged[i]) {

            int *d_new_neighborhoods = hd_new_neighborhoods_list[j];
            int *d_new_neighborhood_end = hd_new_neighborhood_end_list[j];
            j++;

            bool pruneRecursion = restricted_scy_tree->pruneRecursion(tmps, min_size, d_X, n, d,
                                                                                    neighborhood_size, F, num_obj,
                                                                                    d_new_neighborhoods,
                                                                                    d_new_neighborhood_end,
                                                                                    rectangular);

            if (pruneRecursion) {

                GPU_INSCY_star(d_new_neighborhoods, d_new_neighborhood_end, tmps, restricted_scy_tree,
                               d_X, n, d, neighborhood_size, F, num_obj, min_size,
                               result, dim_no + 1, total_number_of_dim, r, calls, rectangular);

                bool pruneRedundancy = restricted_scy_tree->pruneRedundancy(r, result, n, tmps);
                if (pruneRedundancy) {

                    restricted_scy_tree_list.push_back(restricted_scy_tree);
                    clustering_list.push_back(d_clustering);
                    clustering_used = true;
                    new_neighborhoods_list.push_back(d_new_neighborhoods);
                    new_neighborhood_end_list.push_back(d_new_neighborhood_end);

                } else {
                    free_tree(tmps, restricted_scy_tree, d_new_neighborhoods, d_new_neighborhood_end);
                }
            } else {
                free_tree(tmps, restricted_scy_tree, d_new_neighborhoods, d_new_neighborhood_end);
            }
        }

        dim_no++;

        if (!clustering_used)
            tmps->free_points(d_clustering);
    }

    GPU_Clustering(new_neighborhoods_list, new_neighborhood_end_list, tmps,
                   clustering_list, restricted_scy_tree_list,
                   d_X, n, d, neighborhood_size,
                   F, num_obj, rectangular);

    for (int i = 0; i < restricted_scy_tree_list.size(); i++) {
        GPU_SCY_tree *restricted_scy_tree = restricted_scy_tree_list[i];
        int *d_clustering = clustering_list[i];
        int *d_new_neighborhoods = new_neighborhoods_list[i];
        int *d_new_neighborhood_end = new_neighborhood_end_list[i];

        if (i == restricted_scy_tree_list.size() - 1 ||
            (i < restricted_scy_tree_list.size() - 1 && d_clustering != clustering_list[i + 1])) {

            vector<int> subspace = vector<int>(restricted_scy_tree->h_restricted_dims,
                                               restricted_scy_tree->h_restricted_dims +
                                               restricted_scy_tree->number_of_restricted_dims);

            join_gpu(result, d_clustering, subspace, min_size, r, n, tmps);
        }
        free_tree(tmps, restricted_scy_tree, d_new_neighborhoods, d_new_neighborhood_end);
    }
}

void GPU_INSCY_memory(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, GPU_SCY_tree *scy_tree,
                      float *d_X, int n, int d, float neighborhood_size, float F, int num_obj,
                      int min_size, map<vector<int>, int *, vec_cmp> &result, int first_dim_no,
                      int total_number_of_dim, float r, int &calls, bool rectangular) {
    calls++;
    printf("GPU_INSCY_memory(%d)\r", calls);

    int number_of_dims = total_number_of_dim - first_dim_no;
    int number_of_cells = scy_tree->number_of_cells;

    int start_dim_no = first_dim_no;
    int dim_step_size = 2;
    while (start_dim_no < total_number_of_dim) {
        int end_dim_size = start_dim_no + dim_step_size;
        if (end_dim_size > total_number_of_dim) {
            end_dim_size = total_number_of_dim;
        }

        vector <vector<GPU_SCY_tree *>> L_merged = scy_tree->restrict_merge(tmps, start_dim_no,
                                                                                       end_dim_size - start_dim_no,
                                                                                       number_of_cells);

        vector < GPU_SCY_tree * > restricted_scy_tree_list;
        vector<int *> clustering_list;
        vector<int *> new_neighborhoods_list;
        vector<int *> new_neighborhood_end_list;

        pair<int **, int **> p = find_neighborhoods(tmps, d_neighborhoods, d_neighborhood_end,
                                                        d_X, n, d, scy_tree, L_merged, neighborhood_size);

        int **hd_new_neighborhoods_list = p.first;
        int **hd_new_neighborhood_end_list = p.second;


        int j = 0;
        int dim_no = start_dim_no;
        while (dim_no < end_dim_size) {

            int *d_clustering = tmps->malloc_points();
            cudaMemset(d_clustering, -1, sizeof(int) * n);

            bool clustering_used = false;

            int i = dim_no - start_dim_no;
            for (GPU_SCY_tree *restricted_scy_tree : L_merged[i]) {

                int *d_new_neighborhoods = hd_new_neighborhoods_list[j];
                int *d_new_neighborhood_end = hd_new_neighborhood_end_list[j];
                j++;

                bool pruneRecursion = restricted_scy_tree->pruneRecursion(tmps, min_size, d_X, n, d,
                                                                                        neighborhood_size, F, num_obj,
                                                                                        d_new_neighborhoods,
                                                                                        d_new_neighborhood_end,
                                                                                        rectangular);

                if (pruneRecursion) {
                    GPU_INSCY_memory(d_new_neighborhoods, d_new_neighborhood_end, tmps, restricted_scy_tree,
                                     d_X, n, d, neighborhood_size, F, num_obj, min_size,
                                     result, dim_no + 1, total_number_of_dim, r, calls, rectangular);

                    bool pruneRedundancy = restricted_scy_tree->pruneRedundancy(r, result, n, tmps);
                    if (pruneRedundancy) {

                        restricted_scy_tree_list.push_back(restricted_scy_tree);
                        clustering_list.push_back(d_clustering);
                        clustering_used = true;
                        new_neighborhoods_list.push_back(d_new_neighborhoods);
                        new_neighborhood_end_list.push_back(d_new_neighborhood_end);

                    } else {
                        free_tree(tmps, restricted_scy_tree, d_new_neighborhoods, d_new_neighborhood_end);
                    }
                } else {
                    free_tree(tmps, restricted_scy_tree, d_new_neighborhoods, d_new_neighborhood_end);
                }
            }

            dim_no++;

            if (!clustering_used)
                tmps->free_points(d_clustering);
        }

        GPU_Clustering(new_neighborhoods_list, new_neighborhood_end_list, tmps,
                       clustering_list, restricted_scy_tree_list,
                       d_X, n, d, neighborhood_size,
                       F, num_obj, rectangular);

        for (int i = 0; i < restricted_scy_tree_list.size(); i++) {
            GPU_SCY_tree *restricted_scy_tree = restricted_scy_tree_list[i];
            int *d_clustering = clustering_list[i];
            int *d_new_neighborhoods = new_neighborhoods_list[i];
            int *d_new_neighborhood_end = new_neighborhood_end_list[i];

            if (i == restricted_scy_tree_list.size() - 1 ||
                (i < restricted_scy_tree_list.size() - 1 && d_clustering != clustering_list[i + 1])) {

                vector<int> subspace = vector<int>(restricted_scy_tree->h_restricted_dims,
                                                   restricted_scy_tree->h_restricted_dims +
                                                   restricted_scy_tree->number_of_restricted_dims);

                join_gpu(result, d_clustering, subspace, min_size, r, n, tmps);
            }
            free_tree(tmps, restricted_scy_tree, d_new_neighborhoods, d_new_neighborhood_end);
        }
        start_dim_no += dim_step_size;
    }
}
