#ifndef GPU_INSCY_SCY_TREE_H
#define GPU_INSCY_SCY_TREE_H

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <math.h>
#include <vector>
#include <map>
#include <memory>

using namespace std;

class Node;

class Neighborhood_tree;

class GPU_SCY_tree;

struct vec_cmp;

class SCY_tree {

public:
    int number_of_dims;
    int number_of_restricted_dims;
    int number_of_cells;
    int number_of_points;

    float v = 1.;

    float * maxs;
    float * mins;
    int *dims;
    int *restricted_dims;
    bool is_s_connected;

    shared_ptr <Node> root;

    SCY_tree(at::Tensor X, int *subspace, int number_of_cells, int subspace_size, int n, float neighborhood_size, float *mins, float *maxs);

    SCY_tree(float *mins, float *maxs, float v);

    SCY_tree *restrict(int dim_no, int cell_no);

    int get_cell_no(float x_ij, int j);

    shared_ptr <Node> set_node(shared_ptr <Node> node, int &cell_no, int &node_counter);

    void construct_s_connection(float neighborhood_size, int &node_counter, shared_ptr <Node> node, float *x_i, int j,
                                float x_ij, int cell_no);

    vector<int> get_possible_neighbors(float *p, int *subspace, int subspace_size, float neighborhood_size);

    void merge(SCY_tree *other_scy_tree);

    SCY_tree *mergeWithNeighbors(SCY_tree *parent_SCYTree, int dim_no, int &cell_no);

    bool pruneRecursionAndRemove2(int min_size, Neighborhood_tree *neighborhood_tree, at::Tensor X, float neighborhood_size,
                                  int *subspace, int subspace_size, float F, int num_obj, int n, int d,
                                  bool rectangular);

    bool pruneRedundancy(float r, map <vector<int>, vector<int>, vec_cmp> result);

    vector<int> get_points();

    void get_points_node(shared_ptr <Node> node, vector<int> &result);

    bool restrict_node(shared_ptr <Node> old_node, shared_ptr <Node> new_parent, int dim_no, int cell_no, int depth,
                       bool &s_connection_found);

    shared_ptr <Node> set_s_connection(shared_ptr <Node> node, int cell_no, int &node_counter);

    void get_possible_neighbors_from(vector<int> &list, float *p, shared_ptr <Node> node, int depth, int subspace_index,
                                     int *subspace, int subspace_size, float neighborhood_size);

    GPU_SCY_tree *convert_to_GPU_SCY_tree();

    int get_number_of_nodes();

    int get_number_of_nodes_in_subtree(shared_ptr <Node> node);

    void propergate_count(shared_ptr <Node> node);

    void get_leafs(shared_ptr <Node> &node, vector <shared_ptr<Node>> &leafs);
};


#endif //GPU_INSCY_SCY_TREE_H
