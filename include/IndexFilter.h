//
// Created by bld on 24-11-30.
//

#ifndef RANGANN_INDEXFILTER_H
#define RANGANN_INDEXFILTER_H
//
// Created by Mingyu on 2023/8/24.
//

#ifndef RANGANN_INDEX1D_H
#define RANGANN_INDEX1D_H

#include "utils.h"
#include "matrix.h"
#include "hnswlib/hnswalg.h"
#include "hnswlib/hnswlib.h"

#define HNSW_M 16
#define HNSW_efConstruction 500

class IndexFilter {
public:

    IndexFilter() {}

    IndexFilter(unsigned dim) {
        D = dim;
    }

    IndexFilter(unsigned num, unsigned dim) {
        N = num;
        D = dim;
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype> >
    naive_search(const float *query, unsigned K, unsigned nprobs) {
        appr_alg->setEf(nprobs);
        return appr_alg->searchKnn(query, K);
    }


    std::priority_queue<std::pair<float, hnswlib::labeltype> >
    naive_range_search(SegQuery Q, unsigned K, unsigned nprobs) {
        appr_alg->setEf(nprobs);
        RangeFilter range(Q.L, Q.R);
        return appr_alg->searchKnn(Q.data_, K, &range);
    }


    void build_index_and_save(char *input, char *output) {
        Matrix<float> *X = new Matrix<float>(input);
        D = X->d;
        N = X->n;
        size_t report = 50000;

        hnswlib::L2Space l2space(D);

        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, N, HNSW_M, HNSW_efConstruction);
        appr_alg->addPoint(X->data, 0);
        unsigned check_tag = 1;
#pragma omp parallel for schedule(dynamic, 144)
        for(int i=1;i<N;i++){
            appr_alg->addPoint(X->data + i * D, i);
#pragma omp critical
            {
                check_tag++;
                if(check_tag % report == 0){
                    std::cerr << "Processing - " << check_tag << " / " << N << std::endl;
                }
            }
        }
        appr_alg->saveIndex(output);
    }

    void load_index(const char *input) {
        hnswlib::L2Space l2space(D);
        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, input, false);
        std::cerr<<"Load HNSW Finished"<<std::endl;
    }

    unsigned N, D;
    hnswlib::HierarchicalNSW<float> *appr_alg;
};


#endif //RANGANN_INDEX1D_H

#endif //RANGANN_INDEXFILTER_H
