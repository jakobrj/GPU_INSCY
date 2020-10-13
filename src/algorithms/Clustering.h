#ifndef GPU_INSCY_CLUSTERING_H
#define GPU_INSCY_CLUSTERING_H

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <vector>
#include <map>

using namespace std;

class SCY_tree;

class Neighborhood_tree;

struct vec_cmp;

void Clustering(SCY_tree *scy_tree, Neighborhood_tree *neighborhood_tree, at::Tensor X, int n, float neighborhood_size, float F,
                int num_obj, vector<int> &clustering, int min_size, float r,
                map <vector<int>, vector<int>, vec_cmp> result, bool rectangular);

float alpha(int subspace_size, float neighborhood_size, int n, float v);

float omega(int subspace_size);

float c(int subspace_size);

float phi(int point_id, vector<int> neighbors, float neighborhood_size, at::Tensor X, int *subspace, int subspace_size);

vector<int> neighborhood(Neighborhood_tree *neighborhood_tree, int p_id, at::Tensor X, float neighborhood_size, int *subspace,
                         int subspace_size);

#endif //GPU_INSCY_CLUSTERING_H
