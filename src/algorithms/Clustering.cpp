#include <ATen/ATen.h>
#include <torch/extension.h>

#include <set>
#include <map>

#include "Clustering.h"
#include "../structures/SCY_tree.h"
#include "../utils/util.cuh"

#define BLOCK_WIDTH 64

#define PI 3.14


using namespace std;

float dist(int p_id, int q_id, at::Tensor X, int *subspace, int subspace_size) {
    float *p = X[p_id].data_ptr<float>();
    float *q = X[q_id].data_ptr<float>();
    float distance = 0.;
    for (int i = 0; i < subspace_size; i++) {
        int d_i = subspace[i];
        float diff = p[d_i] - q[d_i];
        distance += diff * diff;
    }
    return sqrt(distance);
}


vector<int> neighborhood(SCY_tree *neighborhood_tree, int p_id, at::Tensor X,
                         float neighborhood_size, int *subspace, int subspace_size) {
    vector<int> neighbors;

    float *p = X[p_id].data_ptr<float>();
    vector<int> possible_neighbors = neighborhood_tree->get_possible_neighbors(p, subspace, subspace_size,
                                                                               neighborhood_size);

    int count = 0;

    for (int q_id: possible_neighbors) {
        count++;
        if (p_id == q_id) {
            continue;
        }
        float distance = dist(p_id, q_id, X, subspace, subspace_size);

        if (neighborhood_size >= distance) {
            neighbors.push_back(q_id);
        }
    }
    return neighbors;
}

float gamma(int n) {
    if (n == 2) {
        return 1.;
    } else if (n == 1) {
        return sqrt(PI);
    }
    return (n / 2. - 1.) * gamma(n - 2);
}

float c(int subspace_size) {
    float r = pow(PI, subspace_size / 2.);
    r = r / gamma(subspace_size + 2);
    return r;
}

float phi(int point_id, vector<int> neighbors, float neighborhood_size, at::Tensor X, int *subspace,
          int subspace_size) {

    float sum = 0;
    for (int q_id : neighbors) {
        float d = dist(point_id, q_id, X, subspace, subspace_size) / neighborhood_size;
        float sq = d * d;
        sum += (1. - sq);
    }

    return sum;
}

float alpha(int subspace_size, float neighborhood_size, int n) {
    float v = 1.;
    float r = 2 * n * pow(neighborhood_size, subspace_size) * c(subspace_size);
    r = r / (pow(v, subspace_size) * (subspace_size + 2));
    return r;
}

float omega(int subspace_size) {
    return 2.0 / (subspace_size + 2.0);
}

float expDen(int subspace_size, float neighborhood_size, int n) {
    float v = 1.;
    float r = n * c(subspace_size) * pow(neighborhood_size, subspace_size);
    r = r / pow(v, subspace_size);
    return r;
}

bool dense(int point_id, vector<int> neighbors, float neighborhood_size, at::Tensor X, int *subspace,
           int subspace_size,
           float F, int n, int num_obj) {
    float p = phi(point_id, neighbors, neighborhood_size, X, subspace, subspace_size);
    float a = alpha(subspace_size, neighborhood_size, n);
    float w = omega(subspace_size);

    return p >= max(F * a, num_obj * w);
}

bool dense_rectangular(int point_id, vector<int> neighbors, float neighborhood_size, at::Tensor X, int *subspace,
                       int subspace_size,
                       float F, int n, int num_obj) {
    float a = expDen(subspace_size, neighborhood_size, n);

    return neighbors.size() >= max(F * a, (float) num_obj);
}

void
Clustering(SCY_tree *scy_tree, SCY_tree *neighborhood_tree, at::Tensor X, int n, float neighborhood_size, float F,
           int num_obj, vector<int> &clustering, int min_size, float r, map <vector<int>, vector<int>, vec_cmp> result,
           bool rectangular) {

    int *subspace = scy_tree->restricted_dims;
    int subspace_size = scy_tree->number_of_restricted_dims;

    int clustered_count = 0;
    int prev_clustered_count = 0;
    int next_cluster_label = max(0, v_max(clustering)) + 1;
    vector<int> points = scy_tree->get_points();

    int d = X.size(1);

    queue<int> q;
    for (int i : points) {

        if (clustering[i] != -1) {
            continue;
        }

        int label = next_cluster_label;
        prev_clustered_count = clustered_count;
        q.push(i);

        int c = 0;
        while (!q.empty()) {
            c++;
            int p_id = q.front();
            q.pop();
            vector<int> neighbors = neighborhood(neighborhood_tree, p_id, X, neighborhood_size, subspace,
                                                 subspace_size);

            bool is_dense = rectangular ?
                            dense_rectangular(p_id, neighbors, neighborhood_size, X, subspace, subspace_size, F, n,
                                              num_obj) :
                            dense(p_id, neighbors, neighborhood_size, X, subspace, subspace_size, F, n, num_obj);

            if (is_dense) {
                clustering[p_id] = label;
                clustered_count++;
                for (int q_id : neighbors) {
                    if (clustering[q_id] == -1) {
                        clustering[q_id] = -2;
                        q.push(q_id);
                    }
                }
            }
        }

        if (clustered_count > prev_clustered_count) {
            next_cluster_label++;
        }
    }
}