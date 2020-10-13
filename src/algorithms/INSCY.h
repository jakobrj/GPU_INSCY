#ifndef GPU_INSCY_INSCY_H
#define GPU_INSCY_INSCY_H

#include <map>
#include <vector>

using namespace std;

class SCY_tree;

class Neighborhood_tree;

struct vec_cmp;

void INSCY(SCY_tree *scy_tree, Neighborhood_tree *neighborhood_tree, at::Tensor X, int n, float neighborhood_size, float F,
           int num_obj, int min_size, map <vector<int>, vector<int>, vec_cmp> &result, int first_dim_no, int d, float r,
           int &calls, bool rectangular);


#endif //GPU_INSCY_INSCY_H
