//
// Created by Mingyu on 2023/8/24.
//

#ifndef RANGANN_INDEX_H
#define RANGANN_INDEX_H

#include "utils.h"
#include <faiss/IndexHNSW.h>
#include <faiss/AutoTune.h>
#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>
#include <faiss/IndexIVFFlat.h>
namespace Index {
    class Index {
    public:

        virtual void naive_search(const float *query, unsigned int K, unsigned int nprobs, ResultPool &ans) = 0;

        virtual void naive_filter_search(SegQuery Q, unsigned K, unsigned nprobs, ResultPool &ans) = 0;

        virtual void load_centroid(std::ifstream &fin) {}

        virtual void build_index(bool verbose) = 0;

        virtual void load_index(std::ifstream &in) = 0;

        virtual void save_index(std::ofstream &out) = 0;

        __attribute__((always_inline))
        float inner_id_dist(unsigned id, const float *query) const {
            unsigned outer_id = id + Left_Range;
            return naive_l2_dist_calc(data_ + outer_id * dimension, query, dimension);
        }


        static float *data_;
        static unsigned dimension;
        unsigned nd_, Left_Range, Right_Range;

        typedef std::vector<std::vector<unsigned> > CompactGraph;
        typedef std::vector<std::vector<std::vector<unsigned> > > HierarchicalGraph;
    };
}

#endif //RANGANN_INDEX_H
