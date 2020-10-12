#ifndef GPU_INSCY_GPU_INSCY_CUH
#define GPU_INSCY_GPU_INSCY_CUH

#include <map>
#include <vector>

using namespace std;

class TmpMalloc;

class GPU_SCY_tree;

struct vec_cmp;

void GPU_INSCY(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, GPU_SCY_tree *scy_tree,
               float *d_X, int n, int d, float neighborhood_size, float F, int num_obj,
               int min_size, map<vector<int>, int *, vec_cmp> &result, int first_dim_no,
               int total_number_of_dim, float r, int &calls, bool rectangular);

void GPU_INSCY_star(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, GPU_SCY_tree *scy_tree,
                    float *d_X, int n, int d, float neighborhood_size, float F, int num_obj,
                    int min_size, map<vector<int>, int *, vec_cmp> &result, int first_dim_no,
                    int total_number_of_dim, float r, int &calls, bool rectangular);

void GPU_INSCY_memory(int *d_neighborhoods, int *d_neighborhood_end, TmpMalloc *tmps, GPU_SCY_tree *scy_tree,
                      float *d_X, int n, int d, float neighborhood_size, float F, int num_obj,
                      int min_size, map<vector<int>, int *, vec_cmp> &result, int first_dim_no,
                      int total_number_of_dim, float r, int &calls, bool rectangular);

#endif //GPU_INSCY_GPU_INSCY_CUH
