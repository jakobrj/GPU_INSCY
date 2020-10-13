#include "TmpMalloc.cuh"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <vector>

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void TmpMalloc::free_all() {
    for (pair<int, bool *> p: this->bool_arrays) {
        cudaFree(p.second);
    }
    for (pair<int, float *> p: this->float_arrays) {
        cudaFree(p.second);
    }
    for (pair<int, int *> p: this->int_arrays) {
        cudaFree(p.second);
    }
    for (pair<int, int **> p: this->int_pointer_arrays) {
        cudaFree(p.second);
    }

    int *tmp;
    while (!this->q_points.empty()) {
        tmp = this->q_points.front();
        this->q_points.pop();
        cudaFree(tmp);
        points_count--;
    }
    if (points_count) {
        printf("memory leak for points: %d\n", points_count);
    }
    while (!this->q_nodes.empty()) {
        tmp = this->q_nodes.front();
        this->q_nodes.pop();
        cudaFree(tmp);
        nodes_count--;
    }
    if (nodes_count) {
        printf("memory leak for nodes: %d\n", nodes_count);
    }
    while (!this->q_dims.empty()) {
        tmp = this->q_dims.front();
        this->q_dims.pop();
        cudaFree(tmp);
        dims_count--;
    }
    if (dims_count) {
        printf("memory leak for dims: %d\n", dims_count);
    }
    while (!this->q_one.empty()) {
        tmp = this->q_one.front();
        this->q_one.pop();
        cudaFree(tmp);
        one_count--;
    }
    if (one_count) {
        printf("memory leak for one: %d\n", one_count);
    }


    for (const auto& q_pair : this->q) {
        queue<int*> q = q_pair.second;
        while (!q.empty()) {
            tmp = q.front();
            q.pop();
            cudaFree(tmp);
        }
    }

    not_free = false;
}

TmpMalloc::~TmpMalloc() {
    if (not_free) {
        this->free_all();
    }
}

bool *TmpMalloc::get_bool_array(int name, int size) {
    bool *tmp;
    map<int, bool *>::iterator it = this->bool_arrays.find(name);
    if (it != this->bool_arrays.end()) {
        tmp = this->bool_arrays[name];
        int tmp_size = bool_array_sizes[name];
        if (size > tmp_size) {
            cudaFree(tmp);
            cudaMalloc(&tmp, size * sizeof(bool));
            this->bool_arrays[name] = tmp;
            this->bool_array_sizes[name] = size;
        }
    } else {
        cudaMalloc(&tmp, size * sizeof(bool));
        this->bool_arrays.insert(pair<int, bool *>(name, tmp));
        this->bool_array_sizes.insert(pair<int, int>(name, size));
    }
    return tmp;
}

float *TmpMalloc::get_float_array(int name, int size) {
    float *tmp;
    map<int, float *>::iterator it = this->float_arrays.find(name);
    if (it != this->float_arrays.end()) {
        tmp = this->float_arrays[name];
        int tmp_size = float_array_sizes[name];
        if (size > tmp_size) {
            cudaFree(tmp);
            cudaMalloc(&tmp, size * sizeof(float));
            this->float_arrays[name] = tmp;
            this->float_array_sizes[name] = size;
        }
    } else {
        cudaMalloc(&tmp, size * sizeof(float));
        this->float_arrays.insert(pair<int, float *>(name, tmp));
        this->float_array_sizes.insert(pair<int, int>(name, size));
    }
    return tmp;
}

int *TmpMalloc::get_int_array(int name, int size) {
    int *tmp;
    map<int, int *>::iterator it = this->int_arrays.find(name);
    if (it != this->int_arrays.end()) {
        tmp = this->int_arrays[name];
        int tmp_size = int_array_sizes[name];
        if (size > tmp_size) {
            cudaFree(tmp);
            cudaMalloc(&tmp, size * sizeof(int));
            this->int_arrays[name] = tmp;
            this->int_array_sizes[name] = size;
        }
    } else {
        cudaMalloc(&tmp, size * sizeof(int));
        this->int_arrays.insert(pair<int, int *>(name, tmp));
        this->int_array_sizes.insert(pair<int, int>(name, size));
    }
    return tmp;
}

int **TmpMalloc::get_int_pointer_array(int name, int size) {
    int **tmp;
    map<int, int **>::iterator it = this->int_pointer_arrays.find(name);
    if (it != this->int_pointer_arrays.end()) {
        tmp = this->int_pointer_arrays[name];
        int tmp_size = int_pointer_array_sizes[name];
        if (size > tmp_size) {
            cudaFree(tmp);
            cudaMalloc(&tmp, size * sizeof(int *));
            this->int_pointer_arrays[name] = tmp;
            this->int_pointer_array_sizes[name] = size;
        }
    } else {
        cudaMalloc(&tmp, size * sizeof(int *));
        this->int_pointer_arrays.insert(pair<int, int **>(name, tmp));
        this->int_pointer_array_sizes.insert(pair<int, int>(name, size));
    }
    return tmp;
}

void TmpMalloc::reset_counters() {
    bool_array_counter = 0;
    float_array_counter = 0;
    int_array_counter = 0;
    int_pointer_array_counter = 0;
}

TmpMalloc::TmpMalloc() {
    bool_array_counter = 0;
    float_array_counter = 0;
    int_array_counter = 0;
    int_pointer_array_counter = 0;

    points_count = 0;
    nodes_count = 0;
    dims_count = 0;
    one_count = 0;
}

void TmpMalloc::set(int number_of_points, int number_of_nodes, int number_of_dims) {
    this->number_of_points = number_of_points;
    this->number_of_nodes = number_of_nodes;
    this->number_of_dims = number_of_dims;
}

int *TmpMalloc::malloc_points() {
    int *tmp;
    if (!this->q_points.empty()) {
        tmp = this->q_points.front();
        this->q_points.pop();
    } else {
        cudaMalloc(&tmp, this->number_of_points * sizeof(int));
        points_count++;
    }
    return tmp;
}

void TmpMalloc::free_points(int *memory) {
    this->q_points.push(memory);
}

int *TmpMalloc::malloc_nodes() {
    int *tmp;
    if (!this->q_nodes.empty()) {
        tmp = this->q_nodes.front();
        this->q_nodes.pop();
    } else {
        cudaMalloc(&tmp, number_of_nodes * sizeof(int));
        nodes_count++;
    }
    return tmp;
}

void TmpMalloc::free_nodes(int *memory) {
    this->q_nodes.push(memory);
}

int *TmpMalloc::malloc_dims() {
    int *tmp;
    if (!this->q_dims.empty()) {
        tmp = this->q_dims.front();
        this->q_dims.pop();
    } else {
        cudaMalloc(&tmp, number_of_dims * sizeof(int));
        dims_count++;
    }
    return tmp;
}

void TmpMalloc::free_dims(int *memory) {
    this->q_dims.push(memory);
}

int *TmpMalloc::malloc_one() {
    int *tmp;
    if (!this->q_one.empty()) {
        tmp = this->q_one.front();
        this->q_one.pop();
    } else {
        cudaMalloc(&tmp, sizeof(int));
        one_count++;
    }
    return tmp;
}

void TmpMalloc::free_one(int *memory) {
    this->q_one.push(memory);
}

int *TmpMalloc::malloc_any(int n) {
    int key = int(ceil(log2(n)));

    if (this->q.find(key) == this->q.end()) {
        this->q[key] = std::queue<int *>();
    }

    int *tmp;
    if (!this->q[key].empty()) {
        tmp = this->q[key].front();
        this->q[key].pop();
    } else {
        cudaMalloc(&tmp, pow(2, key) * sizeof(int));
    }
    return tmp;
}

void TmpMalloc::free_any(int *memory, int n) {
    int key = int(ceil(log2(n)));

    this->q[key].push(memory);
}
