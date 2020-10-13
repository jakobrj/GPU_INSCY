#ifndef GPU_INSCY_NEIGHBORHOOD_TREE_H
#define GPU_INSCY_NEIGHBORHOOD_TREE_H

class N_Node {
public:
    vector<int> points;
    map<int, shared_ptr<N_Node>> children;
};

class Neighborhood_tree {
public:

    shared_ptr <N_Node> root;
    float *mins;


    shared_ptr <N_Node> set_node(shared_ptr <N_Node> node, int &no) {
        shared_ptr <N_Node> child(nullptr);
        if (node->children.find(no) == node->children.end()) {
            child = make_shared<N_Node>();
            node->children.insert(pair < int, shared_ptr < N_Node >> (no, child));

        } else {
            child = node->children[no];
        }
        return child;
    }

    Neighborhood_tree(at::Tensor X, float neighborhood_size, float *mins) {

        this->mins = mins;

        int n = X.size(0);
        int d = X.size(1);


        shared_ptr <N_Node> root(new N_Node());
        this->root = root;

        for (int i = 0; i < n; i++) {
            float *x_i = X[i].data_ptr<float>();
            shared_ptr <N_Node> node = root;
            for (int j = 0; j < d; j++) {

                float x_ij = x_i[j];
                int no = (x_ij - this->mins[j]) / neighborhood_size;

                shared_ptr <N_Node> child = set_node(node, no);

                node = child;

            }
            node->points.push_back(i);
        }
    }

    vector<int> get_possible_neighbors(float *p, int *subspace, int subspace_size, float neighborhood_size) {
        vector<int> list;
        int depth = -1;
        int subspace_index = 0;
        get_possible_neighbors_from(list, p, root, depth, subspace_index, subspace, subspace_size, neighborhood_size);
        return list;
    }


    void get_possible_neighbors_from(vector<int> &list, float *p, shared_ptr <N_Node> node, int depth,
                                     int subspace_index, int *subspace, int subspace_size, float neighborhood_size) {

        if (node->children.empty()) {
            list.insert(list.end(), node->points.begin(), node->points.end());
            return;
        }

        depth = depth + 1;
        int center_cell_no = 0;
        bool is_restricted_dim = subspace_index < subspace_size && depth == subspace[subspace_index];
        if (is_restricted_dim) {
            int j = subspace[subspace_index];
            center_cell_no = (p[j] - this->mins[j]) / neighborhood_size;
            subspace_index = subspace_index + 1;
        }

        for (pair<const int, shared_ptr <N_Node>> child_pair : node->children) {
            int cell_no = child_pair.first;
            shared_ptr <N_Node> child = child_pair.second;
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
};


#endif //GPU_INSCY_NEIGHBORHOOD_TREE_H
