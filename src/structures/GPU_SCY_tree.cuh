#ifndef GPU_INSCY_GPU_SCY_TREE_CUH
#define GPU_INSCY_GPU_SCY_TREE_CUH

#include <map>
#include <vector>

using namespace std;

class TmpMalloc;

struct vec_cmp;

class GPU_SCY_tree {

public:
    TmpMalloc *tmps = nullptr;

    bool freed_partial = false;

    float v = 1.;

    //host variables
    int number_of_cells;
    int number_of_dims;
    int number_of_restricted_dims;
    int number_of_nodes;
    int number_of_points;
    bool is_s_connected;

    //device variables
    int d_number_of_cells;
    int d_number_of_dims;

    int d_number_of_restricted_dims;
    int d_number_of_nodes;
    int d_number_of_points;
    float d_cell_size;
    bool d_is_s_connected;

    //host node representation
    int *h_parents;
    int *h_cells;
    int *h_counts;
    int *h_dim_start;
    int *h_dims;
    int *h_restricted_dims;

    int *h_points;
    int *h_points_placement;

    //device node representation
    int *d_parents;
    int *d_cells;
    int *d_counts;
    int *d_dim_start;
    int *d_dims;
    int *d_restricted_dims;

    int *d_points;
    int *d_points_placement;

    float * mins;
    float * maxs;

    GPU_SCY_tree(TmpMalloc *tmps, int number_of_nodes, int number_of_dims, int number_of_restricted_dims,
                 int number_of_points, int number_of_cells, float *mins, float *maxs, float v);

    GPU_SCY_tree(int number_of_nodes, int number_of_dims, int number_of_restricted_dims, int number_of_points,
                 int number_of_cells, float *mins, float *maxs, float v);

    ~GPU_SCY_tree();

    vector <vector<GPU_SCY_tree *>> restrict_merge(TmpMalloc *tmps, int first_dim_no, int number_of_dims,
                                                   int number_of_cells);

    bool pruneRedundancy(float r, map<vector<int>, int *, vec_cmp> result, int n, TmpMalloc *tmps);

    bool pruneRecursion(TmpMalloc *tmps, int min_size, float *d_X, int n, int d,
                        float neighborhood_size, float F, int num_obj, int *d_neighborhoods,
                        int *d_neighborhood_end, bool rectangular);

    void copy_to_device();
};


#endif //GPU_INSCY_GPU_SCY_TREE_CUH
