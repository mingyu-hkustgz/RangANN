//
// Created by mingyu on 23-8-21.
//

#include <chrono>
#include <queue>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <unordered_map>
#include <cassert>
#include <chrono>
#include <cstring>
#include <random>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <boost/unordered_map.hpp>
#include <stack>
#include <x86intrin.h>
#include <immintrin.h>
#include <malloc.h>
#include <set>
#include <cmath>
#include <queue>
#include <Eigen/Dense>

#ifndef WIN32

#include<sys/resource.h>

#endif


#ifndef RANGANN_UTILS_H

unsigned index_dist_calc = 0;
unsigned filter_dist_calc = 0;

struct Neighbor {
    unsigned id;
    float distance;
    bool flag;

    Neighbor() = default;

    Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

    inline bool operator<(const Neighbor &other) const {
        return distance < other.distance;
    }
};

struct SegQuery {
    SegQuery(unsigned left_range, unsigned right_range, float *data) {
        L = left_range;
        R = right_range;
        data_ = data;
    }
    unsigned L, R;
    float *data_;
};

typedef std::vector<std::pair<float, unsigned>> ResultPool;

#define RANGANN_UTILS_H


bool isFileExists_ifstream(const char *name) {
    std::ifstream f(name);
    return f.good();
}


void load_float_data(char *filename, float *&data, unsigned &num,
                     unsigned &dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t) ss;
    num = (unsigned) (fsize / (dim + 1) / 4);
    data = new float[num * dim * sizeof(float)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char *) (data + i * dim), dim * 4);
    }
    in.close();
}

void load_int_data(char *filename, int *&data, unsigned &num,
                   unsigned &dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t) ss;
    num = (unsigned) (fsize / (dim + 1) / 4);
    data = new int[num * dim * sizeof(int)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char *) (data + i * dim), dim * 4);
    }
    in.close();
}

float naive_l2_dist_calc(const float *q, const float *p, const unsigned &dim) {
    float ans = 0.0;
    for (unsigned i = 0; i < dim; i++) {
        ans += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return ans;
}

float naive_inner_product(const float *q, const float *p, const unsigned &dim) {
    float ans = 0.00;
    for (unsigned i = 0; i < dim; i++) {
        ans += p[i] * q[i];
    }
    return ans;
}

int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
    // find the location to insert
    int left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
        memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }
    if (addr[right].distance < nn.distance) {
        addr[K] = nn;
        return K;
    }
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (addr[mid].distance > nn.distance)right = mid;
        else left = mid;
    }
    //check equal ID

    while (left > 0) {
        if (addr[left].distance < nn.distance) break;
        if (addr[left].id == nn.id) return K + 1;
        left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id)return K + 1;
    memmove((char *) &addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
}

__attribute__((always_inline))
std::vector<std::pair<float, unsigned> >
merge_sort(std::vector<std::pair<float, unsigned> > &left, std::vector<std::pair<float, unsigned> > &right,
           unsigned K) {
    std::vector<std::pair<float, unsigned> > ans(K);
    unsigned L = 0, R = 0, cur = 0;
    while (cur < K) {
        if (L < left.size() && R < right.size()) {
            if (left[L].first < right[R].first) ans[cur] = left[L], L++;
            else ans[cur] = right[R], R++;
        }
        else if (L < left.size()) ans[cur] = left[L], L++;
        else if (R < right.size()) ans[cur] = right[R], R++;
        else break;
        cur++;
    }
    return ans;
}

#endif //RANGANN_UTILS_H
