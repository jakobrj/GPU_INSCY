#include <ATen/ATen.h>
#include <torch/extension.h>

#include "INSCY.h"
#include "../algorithms/Clustering.h"
#include "../structures/SCY_tree.h"
#include "../structures/Neighborhood_tree.h"
#include "../utils/util.cuh"


void
INSCY(SCY_tree *scy_tree, Neighborhood_tree *neighborhood_tree, at::Tensor X, int n, float neighborhood_size, float F,
      int num_obj, int min_size, map <vector<int>, vector<int>, vec_cmp> &result, int first_dim_no, int d, float r,
      int &calls, bool rectangular) {

    calls++;
    printf("INSCY(%d)\r", calls);


    int dim_no = first_dim_no;
    while (dim_no < d) {
        int cell_no = 0;

        vector<int> subspace_clustering(n, -1);
        vector<int> subspace;

        int count = 0;
        while (cell_no < scy_tree->number_of_cells) {
            SCY_tree *restricted_scy_tree = scy_tree->restrict(dim_no, cell_no);
            subspace = vector<int>(restricted_scy_tree->restricted_dims, restricted_scy_tree->restricted_dims +
                                                                         restricted_scy_tree->number_of_restricted_dims);
            restricted_scy_tree->mergeWithNeighbors(scy_tree, dim_no, cell_no);

            int before = restricted_scy_tree->number_of_points;
            if (restricted_scy_tree->pruneRecursionAndRemove(min_size, neighborhood_tree, X, neighborhood_size,
                                                              restricted_scy_tree->restricted_dims,
                                                              restricted_scy_tree->number_of_restricted_dims, F,
                                                              num_obj, n, d, rectangular)) {
                INSCY(restricted_scy_tree, neighborhood_tree, X, n, neighborhood_size,
                      F, num_obj, min_size, result,
                      dim_no + 1, d, r, calls, rectangular);

                if (restricted_scy_tree->pruneRedundancy(r, result)) {
                    Clustering(restricted_scy_tree, neighborhood_tree, X, n, neighborhood_size, F, num_obj,
                               subspace_clustering, min_size, r, result, rectangular);
                }
            }
            delete restricted_scy_tree;
            cell_no++;
        }


        join(result, subspace_clustering, subspace, min_size, r);

        dim_no++;
    }
}