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
#include "hnswlib/hnswlib.h"
#ifndef WIN32

#include<sys/resource.h>

#endif
#define RANG_BOUND 1024

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

class RangeFilter : public hnswlib::BaseFilterFunctor {
public:
    hnswlib::labeltype L, R;

    RangeFilter(hnswlib::labeltype left, hnswlib::labeltype right) {
        L = left;
        R = right;
    }

    bool operator()(hnswlib::labeltype id) override {
        if (L <= id && id <= R) return true;
        else return false;
    }
};


typedef std::priority_queue<std::pair<float, hnswlib::labeltype>> ResultQueue;

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

inline float sqr_dist(float *d, float *q, uint32_t L) {
    float PORTABLE_ALIGN32 TmpRes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t num_blk16 = L >> 4;
    uint32_t l = L & 0b1111;

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);
    for (int i = 0; i < num_blk16; i++) {
        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    for (int i = 0; i < l / 8; i++) {
        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    _mm256_store_ps(TmpRes, sum);

    float ret = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    for (int i = 0; i < l % 8; i++) {
        float tmp = (*q) - (*d);
        ret += tmp * tmp;
        d++;
        q++;
    }
    return ret;
}

ResultQueue bruteforce_range_search(SegQuery Q,float *base,unsigned D, unsigned K) {
    ResultQueue res;
    for (unsigned i = Q.L; i <= Q.R; i++) {
        float dist = sqr_dist(Q.data_, base + i * D, D);
        if (res.size() < K)
            res.emplace(dist, i);
        else if (dist < res.top().first) {
            res.pop();
            res.emplace(dist, i);
        }
    }
    return res;
}

#endif //RANGANN_UTILS_H
