//
// Created by Mingyu on 2023/8/24.
//

#ifndef RANGANN_INDEX1D
#define RANGANN_INDEX1D

#include "utils.h"
#include "matrix.h"
#include "hnswlib/hnswalg.h"
#include "hnswlib/hnswlib.h"

#define M 16
#define efConstruction 500


class Index1D {
public:
    Index1D() {}

    Index1D(unsigned dim){
        D = dim ;
    }

    Index1D(unsigned num, unsigned dim){
        N = num;
        D = dim;
    }
    std::priority_queue<std::pair<float, hnswlib::labeltype> >
    naive_search(const float *query, unsigned K, unsigned nprobs) {
        auto appr_alg = appr_alg_list.back();
        appr_alg->setEf(nprobs);
        return appr_alg->searchKnn(query, K);
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype> >
    naive_range_search(SegQuery Q, unsigned K, unsigned nprobs) {
        int idx = 0;
        for(int i = 0; i < index_range_list.size(); i++){
            if(Q.R < index_range_list[i].second){
                idx = i;
                break;
            }
        }
        auto appr_alg = appr_alg_list[idx];
        appr_alg->setEf(nprobs);
        RangeFilter range(Q.L, Q.R);
        return appr_alg->searchKnn(Q.data_, K, &range);
    }



    void build_1D_index_and_save(char *input, char *output) {
        Matrix<float> *X = new Matrix<float>(input);
        D = X->d;
        N = X->n;
        size_t report = 50000;

        hnswlib::L2Space l2space(D);
        hnswlib::labeltype Up = N, Now = 0;
        while (Up > RANG_BOUND) {
            index_range_list.emplace_back(0, Up);
            Up >>= 1;
        }
        std::sort(index_range_list.begin(), index_range_list.end());
        auto appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, N, M, efConstruction);
        appr_alg->addPoint(X->data, 0);
        Now = 1;
        std::ofstream fout(output, std::ios::binary);
        unsigned check_tag = 1;
        for (auto range: index_range_list) {
#pragma omp parallel for schedule(dynamic, 144)
            for (hnswlib::labeltype i = Now; i < range.second; i++) {
                appr_alg->addPoint(X->data + i * D, i);
#pragma omp critical
                {
                    check_tag++;
                    if (check_tag % report == 0) {
                        std::cerr << "Processing - " << check_tag << " / " << N << std::endl;
                    }
                }
            }
            Now = range.second;
            fout.write((char *) &appr_alg->enterpoint_node_, sizeof(unsigned int));
            fout.write((char *) &appr_alg->maxlevel_, sizeof(unsigned int));
            for (size_t j = 0; j < Now; j++) {
                unsigned int linkListSize = appr_alg->element_levels_[j] > 0 ? appr_alg->size_links_per_element_ *
                                                                               appr_alg->element_levels_[j] : 0;
                fout.write((char *) &linkListSize, sizeof(unsigned));
                if (linkListSize)
                    fout.write((char *) appr_alg->linkLists_[j], linkListSize);
            }
            fout.write((char *) appr_alg->data_level0_memory_, appr_alg->cur_element_count * appr_alg->size_data_per_element_);
        }
    }

    void load_1D_index(char *input) {
        std::ifstream fin(input, std::ios::binary);
        hnswlib::labeltype Up = N, Now;
        while (Up > RANG_BOUND) {
            index_range_list.emplace_back(0, Up);
            Up >>= 1;
        }
        std::sort(index_range_list.begin(), index_range_list.end());
        for (auto range: index_range_list) {
            Now = range.second;

            auto l2space = new hnswlib::L2Space(D);
            auto appr_alg = new hnswlib::HierarchicalNSW<float>(l2space, Now, M, efConstruction);
            fin.read((char *) &appr_alg->enterpoint_node_, sizeof(unsigned int));
            fin.read((char *) &appr_alg->maxlevel_, sizeof(unsigned int));
            appr_alg->cur_element_count = Now;
            for (size_t j = 0; j < Now; j++) {
                unsigned int linkListSize;
                fin.read((char *) &linkListSize, sizeof(unsigned));
                if (linkListSize == 0) {
                    appr_alg->element_levels_[j] = 0;
                    appr_alg->linkLists_[j] = nullptr;
                } else {
                    appr_alg->element_levels_[j] = linkListSize / appr_alg->size_links_per_element_;
                    appr_alg->linkLists_[j] = (char *) malloc(linkListSize);
                    if (appr_alg->linkLists_[j] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    fin.read(appr_alg->linkLists_[j], linkListSize);
                }

            }
            fin.read((char*)appr_alg->data_level0_memory_, appr_alg->cur_element_count * appr_alg->size_data_per_element_);
            appr_alg_list.push_back(appr_alg);
        }
        std::cerr<<"Load Finished"<<std::endl;
    }

    unsigned N, D;
    std::vector<hnswlib::HierarchicalNSW<float> *> appr_alg_list;
    std::vector<std::pair<hnswlib::labeltype, hnswlib::labeltype> > index_range_list;
};


#endif //RANGANN_INDEX1D
