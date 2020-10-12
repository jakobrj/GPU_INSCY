#include <ATen/ATen.h>
#include <torch/extension.h>

#include "Node.h"
#include "SCY_tree.h"
#include "GPU_SCY_tree.cuh"
#include "../algorithms/Clustering.h"
#include "../utils/util.cuh"



int SCY_tree::get_cell_no(float x_ij) {
    return min(int(x_ij / this->cell_size), this->number_of_cells - 1);
}

SCY_tree::SCY_tree() {
    this->root = make_shared<Node>(-1);
    this->is_s_connected = false;
}

SCY_tree::SCY_tree(at::Tensor X, int *subspace, int number_of_cells, int subspace_size,
                   int n, float neighborhood_size) {
    float v = 1.;
    this->number_of_cells = number_of_cells;
    this->cell_size = v / this->number_of_cells;
    this->dims = subspace;
    this->number_of_dims = subspace_size;
    this->number_of_restricted_dims = 0;

    shared_ptr <Node> root(new Node(-1));
    int node_counter = 0;
    for (int i = 0; i < n; i++) {
        root->count += 1;
        shared_ptr <Node> node = root;
        float *x_i = X[i].data_ptr<float>();

        for (int j = 0; j < number_of_dims; j++) {

            //computing cell no
            float x_ij = x_i[this->dims[j]];
            int cell_no = this->get_cell_no(x_ij);

            //update cell
            shared_ptr <Node> child = set_node(node, cell_no, node_counter);
            child->count += 1;

            this->construct_s_connection(neighborhood_size, node_counter, node, x_i, j, x_ij, cell_no);
            node = child;
        }
        node->points.push_back(i);
        node->is_leaf = true;
    }
    this->number_of_points = root->count;
    this->root = root;
}

int *add_restricted_dim(int *restricted_dims, int number_of_restricted_dims, int dim_no) {
    int *new_restricted_dims = new int[number_of_restricted_dims + 1];
    for (int i = 0; i < number_of_restricted_dims; i++) {
        new_restricted_dims[i] = restricted_dims[i];
    }
    new_restricted_dims[number_of_restricted_dims] = dim_no;
    return new_restricted_dims;
}

SCY_tree *SCY_tree::restrict(int dim_no, int cell_no) {
    auto *restricted_scy_tree = new SCY_tree();
    restricted_scy_tree->number_of_cells = this->number_of_cells;
    restricted_scy_tree->number_of_dims = this->number_of_dims - 1;
    restricted_scy_tree->restricted_dims = add_restricted_dim(this->restricted_dims, this->number_of_restricted_dims,
                                                              dim_no);
    restricted_scy_tree->number_of_restricted_dims = this->number_of_restricted_dims + 1;
    restricted_scy_tree->cell_size = this->cell_size;
    restricted_scy_tree->dims = new int[restricted_scy_tree->number_of_dims];
    int j = 0;
    for (int i = 0; i < this->number_of_dims; i++) {
        if (this->dims[i] != dim_no) {
            restricted_scy_tree->dims[j] = this->dims[i];
            j++;
        }
    }

    int depth = 0;
    restricted_scy_tree->is_s_connected = false;
    for (pair<int, shared_ptr<Node> > child_pair: this->root->children) {
        shared_ptr <Node> old_child = child_pair.second;
        this->restrict_node(old_child, restricted_scy_tree->root, dim_no, cell_no, depth,
                            restricted_scy_tree->is_s_connected);
    }

    for (pair<int, shared_ptr<Node> > child_pair: this->root->s_connections) {
        shared_ptr <Node> old_child = child_pair.second;
        this->restrict_node(old_child, restricted_scy_tree->root, dim_no, cell_no, depth,
                            restricted_scy_tree->is_s_connected);
    }

    restricted_scy_tree->number_of_points = restricted_scy_tree->root->count;
    return restricted_scy_tree;
}

vector<int> SCY_tree::get_possible_neighbors(float *p,
                                                int *subspace, int subspace_size,
                                                float neighborhood_size) {
    vector<int> list;
    vector <shared_ptr<Node>> nodes;
    int depth = -1;
    int subspace_index = 0;
    get_possible_neighbors_from(list, p, root, depth, subspace_index, subspace, subspace_size, neighborhood_size);
    return list;
}

shared_ptr <Node> SCY_tree::set_s_connection(shared_ptr <Node> node, int cell_no, int &node_counter) {
    if (node->s_connections.find(cell_no) == node->s_connections.end()) {
        shared_ptr <Node> s_connection(new Node(cell_no));
        s_connection->count = -1;
        node->s_connections.insert(pair < int, shared_ptr < Node >> (cell_no, s_connection));
        node_counter++;
        return s_connection;
    } else {
        return node->s_connections[cell_no];
    }
}

void SCY_tree::construct_s_connection(float neighborhood_size, int &node_counter, shared_ptr <Node> node,
                                         float *x_i, int j, float x_ij, int cell_no) {
    if (x_ij >= ((cell_no + 1) * cell_size - neighborhood_size)) {
        //todo maybe change neighborhood_size to something else

        shared_ptr <Node> s_connection = set_s_connection(node, cell_no, node_counter);
        shared_ptr <Node> pre_s_connection = s_connection;
        for (int k = j + 1; k < number_of_dims; k++) {
            float x_ik = x_i[dims[k]];
            int cell_no_k = get_cell_no(x_ik);
            s_connection = set_s_connection(pre_s_connection, cell_no_k, node_counter);
            pre_s_connection = s_connection;
        }
        pre_s_connection->is_leaf = true;
    }
}

shared_ptr <Node> SCY_tree::set_node(shared_ptr <Node> node, int &cell_no, int &node_counter) {

    shared_ptr <Node> child(nullptr);
    if (node->children.find(cell_no) == node->children.end()) {
        child = make_shared<Node>(cell_no);
        node->children.insert(pair < int, shared_ptr < Node >> (cell_no, child));
        node_counter++;

    } else {
        child = node->children[cell_no];
    }
    return child;
}


bool
SCY_tree::restrict_node(shared_ptr <Node> old_node, shared_ptr <Node> new_parent, int dim_no, int cell_no, int depth,
                        bool &s_connection_found) {
    bool is_on_restricted_dim = this->dims[depth] == dim_no;
    bool is_restricted_cell = old_node->cell_no == cell_no;

    if (is_on_restricted_dim) {

        if (!is_restricted_cell) {
            return false;
        }

        if (new_parent->count != -1 && old_node->count == -1) {
            s_connection_found = true;
            return false;
        }

        if (old_node->is_leaf) {
            new_parent->is_leaf = true;
            if (old_node->count > 0) {
                new_parent->points.insert(new_parent->points.end(), old_node->points.begin(), old_node->points.end());
                new_parent->count += old_node->count;
            }

            return true;
        }

        if (!old_node->is_leaf) {
            for (pair<int, shared_ptr<Node> > child_pair: old_node->children) {
                shared_ptr <Node> old_child = child_pair.second;
                this->restrict_node(old_child, new_parent, dim_no, cell_no, depth + 1,
                                    s_connection_found);
            }
            for (pair<int, shared_ptr<Node> > child_pair: old_node->s_connections) {
                shared_ptr <Node> old_child = child_pair.second;
                this->restrict_node(old_child, new_parent, dim_no, cell_no, depth + 1,
                                    s_connection_found);
            }

            return true;
        }
    } else {
        shared_ptr <Node> new_node(new Node(old_node));
        bool is_included = this->dims[depth] > dim_no;
        if (old_node->count == -1) {
            for (pair<int, shared_ptr<Node> > child_pair: old_node->s_connections) {
                shared_ptr <Node> old_child = child_pair.second;
                is_included = this->restrict_node(old_child, new_node, dim_no, cell_no,
                                                  depth + 1, s_connection_found) || is_included;
            }
            if (is_included)
                new_parent->s_connections.insert(pair < int, shared_ptr < Node >> (new_node->cell_no, new_node));
        } else {
            if (!old_node->is_leaf)
                new_node->count = 0;
            else
                is_included = true;


            for (pair<int, shared_ptr<Node> > child_pair: old_node->children) {
                shared_ptr <Node> old_child = child_pair.second;
                is_included = this->restrict_node(old_child, new_node, dim_no, cell_no,
                                                  depth + 1, s_connection_found) || is_included;
            }

            for (pair<int, shared_ptr<Node> > child_pair: old_node->s_connections) {
                shared_ptr <Node> old_child = child_pair.second;
                is_included = this->restrict_node(old_child, new_node, dim_no, cell_no,
                                                  depth + 1, s_connection_found) || is_included;
            }

            if (is_included) {
                new_parent->children.insert(pair < int, shared_ptr < Node >> (new_node->cell_no, new_node));
                new_parent->count += new_node->count;
            }
        }
        return is_included;
    }
}

void mergeNodes(shared_ptr <Node> node_1, shared_ptr <Node> node_2) {

    if (node_1->count > 0) {
        node_1->count += node_2->count;
        node_1->points.insert(node_1->points.end(), node_2->points.begin(), node_2->points.end());
    }

    for (pair<const int, shared_ptr < Node>> child_pair: node_2->children) {
        int cell_no_2 = child_pair.first;
        shared_ptr <Node> child_2 = child_pair.second;
        if (node_1->children.count(cell_no_2)) {
            shared_ptr <Node> child_1 = node_1->children[cell_no_2];
            mergeNodes(child_1, child_2);
        } else {
            node_1->children.insert(pair < int, shared_ptr < Node >> (cell_no_2, child_2));
        }
    }

    for (pair<const int, shared_ptr < Node>> child_pair: node_2->s_connections) {
        int cell_no_2 = child_pair.first;
        shared_ptr <Node> child_2 = child_pair.second;
        if (node_1->s_connections.count(cell_no_2)) {
            shared_ptr <Node> child_1 = node_1->s_connections[cell_no_2];
            mergeNodes(child_1, child_2);
        } else {
            node_1->s_connections.insert(pair < int, shared_ptr < Node >> (cell_no_2, child_2));
        }
    }
}

void SCY_tree::merge(SCY_tree *other_scy_tree) {
    mergeNodes(this->root, other_scy_tree->root);
}

SCY_tree *SCY_tree::mergeWithNeighbors(SCY_tree *parent_SCYTree, int dim_no, int &cell_no) {
    if (!this->is_s_connected) {
        return this;
    }
    SCY_tree *restricted_scy_tree = this;
    while (restricted_scy_tree->is_s_connected && cell_no < this->number_of_cells - 1) {
        restricted_scy_tree = (SCY_tree *) ((SCY_tree *) parent_SCYTree)->restrict(dim_no, cell_no + 1);
        this->merge(restricted_scy_tree);
        delete restricted_scy_tree;
        cell_no++;
    }
    this->number_of_points = this->root->count;
    this->is_s_connected = false;
    return this;
}

int SCY_tree::get_number_of_nodes() {
    return get_number_of_nodes_in_subtree(this->root);
}

int SCY_tree::get_number_of_nodes_in_subtree(shared_ptr <Node> node) {
    int count = 1;
    for (pair<const int, shared_ptr < Node>> child_pair: node->children) {
        shared_ptr <Node> child = child_pair.second;
        count += get_number_of_nodes_in_subtree(child);
    }
    for (pair<const int, shared_ptr < Node>> child_pair: node->s_connections) {
        shared_ptr <Node> child = child_pair.second;
        count += get_number_of_nodes_in_subtree(child);
    }
    return count;
}

GPU_SCY_tree *SCY_tree::convert_to_GPU_SCY_tree() {
    int number_of_nodes = this->get_number_of_nodes();

    GPU_SCY_tree *scy_tree_array = new GPU_SCY_tree(number_of_nodes, this->number_of_dims,
                                                    this->number_of_restricted_dims,
                                                    this->number_of_points, this->number_of_cells);

    scy_tree_array->h_dims = this->dims;
    scy_tree_array->h_restricted_dims = this->restricted_dims;
    scy_tree_array->cell_size = this->cell_size;
    scy_tree_array->is_s_connected = this->is_s_connected;

    vector <shared_ptr<Node>> next_nodes = vector < shared_ptr < Node >> ();
    next_nodes.push_back(this->root);
    scy_tree_array->h_dim_start[0] = 1;

    int l = 0;
    int j = 0;

    scy_tree_array->h_cells[j] = -1;
    scy_tree_array->h_counts[j] = this->number_of_points;
    scy_tree_array->h_parents[j] = 0;
    for (int point :this->root->points) {
        scy_tree_array->h_points[l] = point;
        scy_tree_array->h_points_placement[l] = j;
        l++;
    }
    j++;


    for (int i = 0; i < this->number_of_dims; i++) {

        vector <shared_ptr<Node>> nodes = next_nodes;
        next_nodes = vector < shared_ptr < Node >> ();
        for (int k = 0; k < nodes.size(); k++) {
            shared_ptr <Node> node = nodes[k];
            for (pair<const int, shared_ptr < Node>> child_pair: node->children) {
                shared_ptr <Node> child = child_pair.second;
                next_nodes.push_back(child);
                scy_tree_array->h_cells[j] = child->cell_no;
                scy_tree_array->h_counts[j] = child->count;
                scy_tree_array->h_parents[j] = i == 0 ? 0 : scy_tree_array->h_dim_start[i - 1] + k;
                for (int point : child->points) {
                    scy_tree_array->h_points[l] = point;
                    scy_tree_array->h_points_placement[l] = j;
                    l++;
                }
                j++;
            }
            for (pair<const int, shared_ptr < Node>> child_pair: node->s_connections) {
                shared_ptr <Node> child = child_pair.second;
                next_nodes.push_back(child);
                scy_tree_array->h_cells[j] = child->cell_no;
                scy_tree_array->h_counts[j] = child->count;
                scy_tree_array->h_parents[j] = i == 0 ? 0 : scy_tree_array->h_dim_start[i - 1] + k;
                j++;
            }
        }
        if (i < this->number_of_dims - 1)
            scy_tree_array->h_dim_start[i + 1] = j;
    }

    return scy_tree_array;
}


void SCY_tree::propergate_count(shared_ptr <Node> node) {
    if (node->children.empty() && node->s_connections.empty()) {
        // do nothing
    } else {
        node->count = 0;
        vector<int> cells_to_be_removed;
        for (pair<const int, shared_ptr < Node>> child_pair: node->children) {
            shared_ptr <Node> child = child_pair.second;
            this->propergate_count(child);
            if (child->count == 0) {
                cells_to_be_removed.push_back(child->cell_no);
            } else {
                node->count += child->count;
            }
        }

        for (int cell: cells_to_be_removed) {
            node->children.erase(cell);
            node->s_connections.erase(cell);
        }
    }
}

void SCY_tree::get_leafs(shared_ptr <Node> &node, vector <shared_ptr<Node>> &leafs) {
    if (node->children.empty() && node->s_connections.empty()) {
        leafs.push_back(node);
    } else {
        for (pair<const int, shared_ptr < Node>> child_pair: node->children) {
            shared_ptr <Node> &child = child_pair.second;
            this->get_leafs(child, leafs);
        }
    }
}

bool SCY_tree::pruneRecursionAndRemove2(int min_size, SCY_tree *neighborhood_tree, at::Tensor X,
                                           float neighborhood_size,
                                           int *subspace, int subspace_size, float F, int num_obj, int n, int d,
                                           bool rectangular) {

    vector <shared_ptr<Node>> leafs;
    this->get_leafs(this->root, leafs);

    float a = alpha(d, neighborhood_size, n);
    float w = omega(d);


    float v = 1.;
    float ex = n * c(d) * pow(neighborhood_size, d);
    ex = ex / pow(v, d);
    int pruned_size = 0;
    for (shared_ptr <Node> &leaf: leafs) {
        vector<int> points;
        bool is_weak_dense[leaf->points.size()];
        int count = 0;
        for (int i = 0; i < leaf->points.size(); i++) {
            int p_id = leaf->points[i];
            vector<int> neighbors = neighborhood(neighborhood_tree, p_id, X, neighborhood_size, subspace,
                                                 subspace_size);

            bool is_dense;

            if (rectangular) {
                is_dense = neighbors.size() >= max(F * ex, (float) num_obj);
            } else {
                float p = phi(p_id, neighbors, neighborhood_size, X, subspace, subspace_size);
                is_dense = p >= max(F * a, num_obj * w);
            }

            if (is_dense) {
                pruned_size++;
                count++;
                is_weak_dense[i] = true;
                points.push_back(p_id);
            } else {
                is_weak_dense[i] = false;
            }
        }

        leaf->points = points;

        leaf->count = count;
    }
    this->propergate_count(this->root);
    this->number_of_points = this->root->count;


    return pruned_size >= min_size;
}

vector<int> SCY_tree::get_points() {
    vector<int> result;

    this->get_points_node(this->root, result);

    return result;
}

void SCY_tree::get_points_node(shared_ptr <Node> node, vector<int> &result) {
    if (node->children.empty()) {
        result.insert(result.end(), node->points.begin(), node->points.end());
    }
    for (pair<const int, shared_ptr < Node>> child_pair: node->children) {
        shared_ptr <Node> child = child_pair.second;
        get_points_node(child, result);
    }
}

bool SCY_tree::pruneRedundancy(float r, map <vector<int>, vector<int>, vec_cmp> result) {

    int max_min_size = 0;

    vector<int> subspace(this->restricted_dims, this->restricted_dims + this->number_of_restricted_dims);
    vector<int> max_min_subspace;

    for (std::pair <vector<int>, vector<int>> subspace_clustering : result) {

        // find sizes of clusters
        vector<int> subspace_mark = subspace_clustering.first;
        if (subspace_of(subspace, subspace_mark)) {

            vector<int> clustering_mark = subspace_clustering.second;
            map<int, int> cluster_sizes;
            map<int, bool> cluster_to_use;
            for (int cluster_id: clustering_mark) {
                if (cluster_id >= 0) {
                    if (cluster_sizes.count(cluster_id)) {
                        cluster_sizes[cluster_id]++;
                    } else {
                        cluster_sizes.insert(pair<int, int>(cluster_id, 1));
                        cluster_to_use.insert(pair<int, bool>(cluster_id, false));
                    }
                }
            }

            for (int p_id: this->get_points()) {
                int cluster_id = clustering_mark[p_id];
                if (cluster_id >= 0) {
                    cluster_to_use[cluster_id] = true;
                }
            }

            // find the minimum size for each subspace
            int min_size = -1;
            for (std::pair<int, int> cluster_size : cluster_sizes) {
                int cluster_id = cluster_size.first;
                int size = cluster_size.second;
                if (cluster_to_use[cluster_id]) {
                    if (min_size == -1 || size < min_size) {
                        min_size = size;
                    }
                }

            }

            // find the maximum minimum size for each subspace
            if (min_size > max_min_size) {
                max_min_size = min_size;
                max_min_subspace = subspace_mark;
            }
        }
    }

    if (max_min_size == 0) {
        return true;
    }

    return this->number_of_points * r > max_min_size * 1.;
}


void
SCY_tree::get_possible_neighbors_from(vector<int> &list, float *p, shared_ptr <Node> node, int depth,
                                         int subspace_index, int *subspace, int subspace_size,
                                         float neighborhood_size) {

    if (node->children.empty()) {
        list.insert(list.end(), node->points.begin(), node->points.end());
        return;
    }

    depth = depth + 1;
    int center_cell_no = 0;
    bool is_restricted_dim = subspace_index < subspace_size && this->dims[depth] == subspace[subspace_index];
    if (is_restricted_dim) {
        center_cell_no = this->get_cell_no(p[subspace[subspace_index]]);
        subspace_index = subspace_index + 1;
    }

    for (pair<const int, shared_ptr < Node>> child_pair : node->children) {
        int cell_no = child_pair.first;
        shared_ptr <Node> child = child_pair.second;
        bool with_in_possible_neighborhood = false;
        if (is_restricted_dim) {
            if (center_cell_no - 1 <= cell_no && cell_no <= center_cell_no + 1) {
                with_in_possible_neighborhood = true;
            }
        } else {
            with_in_possible_neighborhood = true;
        }
        if (with_in_possible_neighborhood) {
            get_possible_neighbors_from(list, p, child, depth, subspace_index, subspace, subspace_size,
                                        neighborhood_size);
        }
    }
}