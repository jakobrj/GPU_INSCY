#ifndef GPU_INSCY_GPU_CLUSTERING_CUH
#define GPU_INSCY_GPU_CLUSTERING_CUH
#include <vector>
#include <map>

using namespace std;

class TmpMalloc;

class GPU_SCY_tree;

void GPU_Clustering(vector<int *> new_neighborhoods_list, vector<int *> new_neighborhood_end_list, TmpMalloc *tmps,
                    vector<int *> clustering_list, vector<GPU_SCY_tree *> restricted_scy_tree_list, float *d_X, int n,
                    int d, float neighborhood_size, float F, int num_obj, bool rectangular);

pair<int **, int **> find_neighborhoods(TmpMalloc *tmps, int *d_neighborhoods, int *d_neighborhood_end,
                                            float *d_X, int n, int d, GPU_SCY_tree *scy_tree,
                                            vector <vector<GPU_SCY_tree *>> L_merged,
                                            float neighborhood_size);

pair<int **, int **>
find_neighborhoods_star(TmpMalloc *tmps, int *d_neighborhoods, int *d_neighborhood_end, float *d_X, int n, int d,
                           GPU_SCY_tree *scy_tree, vector <vector<GPU_SCY_tree *>> L_merged, float neighborhood_size);

#endif //GPU_INSCY_GPU_CLUSTERING_CUH
