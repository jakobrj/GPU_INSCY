#ifndef GPU_INSCY_TMPMALLOC_CUH
#define GPU_INSCY_TMPMALLOC_CUH

#include <map>
#include <vector>
#include <queue>

using namespace std;


class TmpMalloc {
public:
    int number_of_nodes;
    int number_of_dims;
    int number_of_points;
    std::queue<int *> q_points;
    std::queue<int *> q_nodes;
    std::queue<int *> q_dims;
    std::queue<int *> q_one;

    std::map<int, std::queue<int *>> q;

    int points_count;
    int nodes_count;
    int dims_count;
    int one_count;

    const int CLUSTERING = -1;

    int bool_array_counter = 0;
    map<int, bool *> bool_arrays;
    map<int, int> bool_array_sizes;
    int float_array_counter = 0;
    map<int, float *> float_arrays;
    map<int, int> float_array_sizes;
    int int_array_counter = 0;
    map<int, int *> int_arrays;
    map<int, int> int_array_sizes;
    int int_pointer_array_counter = 0;
    map<int, int **> int_pointer_arrays;
    map<int, int> int_pointer_array_sizes;

    TmpMalloc();

    ~TmpMalloc();

    bool *get_bool_array(int name, int size);

    float *get_float_array(int name, int size);

    int *get_int_array(int name, int size);

    int **get_int_pointer_array(int name, int size);

    void reset_counters();

    int *malloc_points();

    void free_points(int *memory);

    int *malloc_nodes();

    void free_nodes(int *memory);

    int *malloc_dims();

    void free_dims(int *memory);

    int *malloc_one();

    void free_one(int *memory);

    void set(int number_of_points, int number_of_nodes, int number_of_dims);

    void free_all();

    bool not_free = true;

    int *malloc_any(int n);

    void free_any(int *memory, int n);
};


#endif //GPU_INSCY_TMPMALLOC_CUH
