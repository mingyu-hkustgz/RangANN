//
// Created by bld on 24-12-2.
//

#include "hnswlib/hnswlib.h"
#include "hnswlib/hnswalg-static.h"
#include "matrix.h"
#include "utils.h"

#define HNSW2D_M 16
#define HNSW2D_efConstruction 200
template<typename dist_t> char *hnswlib::HierarchicalNSWStatic<dist_t>::static_base_data_ = NULL;

class FanoutTree {
public:
    FanoutTree() {}

    FanoutTree(hnswlib::labeltype left, hnswlib::labeltype right) {
        L = left;
        R = right;
    }

    hnswlib::labeltype L{0}, R{0};
    FanoutTree **children = nullptr;
    hnswlib::HierarchicalNSWStatic<float> *appr_alg = nullptr;
};


class Index2DF {
public:
    Index2DF() {}

    Index2DF(unsigned dim) {
        D = dim;
    }

    Index2DF(unsigned num, unsigned dim) {
        N = num;
        D = dim;
    }

    Index2DF(unsigned num, unsigned dim, unsigned fan) {
        N = num;
        D = dim;
        Fanout = fan;
    }

    void build_2D_index_and_save(char *input, char *output, unsigned fan = 4) {
        Matrix<float> *X = new Matrix<float>(input);
        D = X->d;
        N = X->n;
        Fanout = fan;
        hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) X->data;
        std::ofstream fout(output, std::ios::binary);
        incremental_build(0, N - 1, fout);
    }

    void save_single_static_index(hnswlib::HierarchicalNSWStatic<float> *&appr_alg, std::ofstream &fout) {
        fout.write((char *) &appr_alg->enterpoint_node_, sizeof(unsigned int));
        fout.write((char *) &appr_alg->maxlevel_, sizeof(unsigned int));
        for (size_t j = 0; j < appr_alg->cur_element_count; j++) {
            unsigned int linkListSize = appr_alg->element_levels_[j] > 0 ? appr_alg->size_links_per_element_ *
                                                                           appr_alg->element_levels_[j] : 0;
            fout.write((char *) &linkListSize, sizeof(unsigned));
            if (linkListSize)
                fout.write((char *) appr_alg->linkLists_[j], linkListSize);
        }
        fout.write((char *) appr_alg->data_level0_memory_,
                   appr_alg->cur_element_count * appr_alg->size_data_per_element_);
    }

    void load_single_static_index(hnswlib::HierarchicalNSWStatic<float> *&appr_alg, std::ifstream &fin,
                                  hnswlib::tableint L, hnswlib::tableint R) {
        auto l2space = new hnswlib::L2Space(D);
        appr_alg = new hnswlib::HierarchicalNSWStatic<float>(l2space, (R - L + 1), HNSW2D_M, HNSW2D_efConstruction);
        fin.read((char *) &appr_alg->enterpoint_node_, sizeof(unsigned int));
        fin.read((char *) &appr_alg->maxlevel_, sizeof(unsigned int));
        appr_alg->cur_element_count = (R - L + 1);
        appr_alg->label_begin_ = L;
        for (size_t j = 0; j < (R - L + 1); j++) {
            unsigned int linkListSize;
            fin.read((char *) &linkListSize, sizeof(unsigned));
            if (linkListSize == 0) {
                appr_alg->element_levels_[j] = 0;
                appr_alg->linkLists_[j] = nullptr;
            } else {
                appr_alg->element_levels_[j] = linkListSize / appr_alg->size_links_per_element_;
                appr_alg->linkLists_[j] = (char *) malloc(linkListSize);
                if (appr_alg->linkLists_[j] == nullptr) {
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                }
                fin.read(appr_alg->linkLists_[j], linkListSize);
            }

        }
        fin.read((char *) appr_alg->data_level0_memory_,
                 appr_alg->cur_element_count * appr_alg->size_data_per_element_);
    }


    hnswlib::HierarchicalNSWStatic<float> *
    incremental_build(hnswlib::labeltype L, hnswlib::labeltype R, std::ofstream &fout) {
        if ((R - L + 1) < RANG_BOUND) return nullptr;
        hnswlib::labeltype length = (R - L + 1);
        hnswlib::labeltype sub_length = length % Fanout == 0 ? length / Fanout : (length / Fanout) + 1;

        auto left_index = incremental_build(L, L + sub_length - 1, fout);

        //can not use incremental construct approach
        if (left_index == nullptr) {
            delete left_index;
            auto l2space = new hnswlib::L2Space(D);
            auto cur_index = new hnswlib::HierarchicalNSWStatic<float>(l2space, length, HNSW2D_M,
                                                                       HNSW2D_efConstruction);
            cur_index->label_begin_ = L;
#pragma omp parallel for
            for (hnswlib::labeltype i = L; i <= R; i++) {
                cur_index->addPoint(
                        hnswlib::HierarchicalNSWStatic<float>::static_base_data_ + i * cur_index->data_size_, i - L);
            }
            save_single_static_index(cur_index, fout);
            return cur_index;
        }
        unsigned left_bound = L + sub_length;
        //Incremental construct based on previous Index
        left_index->resizeIndex(length);
#pragma omp parallel for
        for (hnswlib::labeltype i = left_bound; i <= R; i++) {
            left_index->addPoint(
                    hnswlib::HierarchicalNSWStatic<float>::static_base_data_ + i * left_index->data_size_,
                    i - L);
        }
        save_single_static_index(left_index, fout);
        for (int i = 1; i < Fanout; i++) {
            unsigned right_bound = std::min(left_bound + sub_length - 1, R);
            auto right_index = incremental_build(left_bound, right_bound, fout);
            left_bound = right_bound + 1;
            delete right_index;
        }
        return left_index;
    }

    FanoutTree *build_segment_tree(hnswlib::labeltype L, hnswlib::labeltype R, std::ifstream &fin) {
        if ((R - L) < RANG_BOUND) {
            return nullptr;
        }
        auto cur_node = new FanoutTree(L, R);
        hnswlib::labeltype length = (R - L + 1);
        hnswlib::labeltype sub_length = length % Fanout == 0 ? length / Fanout : (length / Fanout) + 1;
        cur_node->children = new FanoutTree *[Fanout];
        cur_node->children[0] = build_segment_tree(L, L + sub_length - 1, fin);
        load_single_static_index(cur_node->appr_alg, fin, L, R);
        unsigned left_bound = L + sub_length;
        for (int i = 1; i < Fanout; i++) {
            unsigned right_bound = std::min(left_bound + sub_length - 1, R);
            cur_node->children[i] = build_segment_tree(left_bound, right_bound, fin);
            left_bound = right_bound + 1;
        }
        return cur_node;
    }


    void load_index(char *input) {
        std::ifstream fin(input, std::ios::binary);
        root = build_segment_tree(0, N - 1, fin);
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype> >
    bruteforce_search(float *&query, float *base, unsigned L, unsigned R, unsigned K) {
        std::priority_queue<std::pair<float, hnswlib::labeltype> > Q;
        for (size_t i = L; i <= R; i++) {
            float dist = sqr_dist(query, base + i * D, D);
            if (Q.size() < K)
                Q.emplace(dist, i);
            else if (dist < Q.top().first) {
                Q.pop();
                Q.emplace(dist, i);
            }
        }
        return Q;
    }

    static bool check_overlap(SegQuery Q, unsigned L, unsigned R) {
        if (L <= Q.L && Q.L <= R) return true;
        if (L <= Q.R && Q.R <= R) return true;
        if (Q.L <= L && R <= Q.R) return true;
        return false;
    }


    std::priority_queue<std::pair<float, hnswlib::labeltype> >
    half_blood_search(SegQuery Q, unsigned K, unsigned nprobs, FanoutTree *&cur) {
        if (cur->L <= Q.L && Q.R <= cur->R && (Q.R - Q.L) * Fanout > (cur->R - cur->L + 1)) {
            RangeFilter range(Q.L - cur->L, Q.R - cur->L);
            cur->appr_alg->setEf(nprobs);
            return cur->appr_alg->searchKnn(Q.data_, K, &range);
        }
        if (cur==nullptr) {
            return bruteforce_search(Q.data_, (float *) (cur->appr_alg->static_base_data_), Q.L,
                                     Q.R, K);
        }
        bool check_all_son_die = true;
        for (int i = 0; i < Fanout; i++) if (cur->children[i] != nullptr) check_all_son_die = false;
        if (check_all_son_die) {
            return bruteforce_search(Q.data_, (float *) (cur->appr_alg->static_base_data_), Q.L,
                                     Q.R, K);
        }
        hnswlib::labeltype length = (cur->R - cur->L + 1), left_bound, right_bound;
        hnswlib::labeltype sub_length = length % Fanout == 0 ? length / Fanout : (length / Fanout) + 1;
        std::priority_queue<std::pair<float, hnswlib::labeltype> > cur_res, child_res;
        left_bound = cur->L;
        for (int i = 0; i < Fanout; i++) {
            right_bound = std::min(left_bound + sub_length - 1, cur->R);
            if (check_overlap(Q, left_bound, right_bound)) {
                SegQuery Q_son = Q;
                Q_son.R = std::min(Q_son.R, right_bound);
                Q_son.L = std::max(Q_son.L, left_bound);
                child_res = half_blood_search(Q_son, K, nprobs, cur->children[i]);
                cur_res = merge_res(cur_res, child_res, K);
            }
            left_bound = right_bound + 1;
        }
        return cur_res;
    }


    unsigned N, D, Fanout = 4;
    FanoutTree *root = nullptr;
};

