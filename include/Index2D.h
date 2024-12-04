//
// Created by bld on 24-12-2.
//

#ifndef RANGANN_INDEX2D_H
#define RANGANN_INDEX2D_H

#include "hnswlib/hnswlib.h"
#include "hnswlib/hnswalg-static.h"
#include "matrix.h"

#define HNSW_M 16
#define HNSW_efConstruction 500
template<typename dist_t> char *hnswlib::HierarchicalNSWStatic<dist_t>::static_base_data_ = NULL;

class SegmentTree {
public:
    SegmentTree(hnswlib::labeltype left, hnswlib::labeltype right) {
        L = left;
        R = right;
    }

    hnswlib::labeltype L{0}, R{0};
    SegmentTree *left = nullptr, *right = nullptr;
    hnswlib::HierarchicalNSWStatic<float> *appr_alg = nullptr;
};


class Index2D {
public:
    Index2D() {}

    Index2D(unsigned dim) {
        D = dim;
    }

    Index2D(unsigned num, unsigned dim) {
        N = num;
        D = dim;
    }

    void build_2D_index_and_save(char *input, char *output) {
        Matrix<float> *X = new Matrix<float>(input);
        D = X->d;
        N = X->n;
        hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) X->data;
        std::ofstream fout(output, std::ios::binary);
        incremental_build(0, N - 1, fout);
    }

    void save_single_static_index(hnswlib::HierarchicalNSWStatic<float> *appr_alg, std::ofstream &fout) {
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

    void load_single_static_index(hnswlib::HierarchicalNSWStatic<float> *appr_alg, std::ifstream &fin,
                                  hnswlib::tableint L, hnswlib::tableint R) {
        auto l2space = new hnswlib::L2Space(D);
        appr_alg = new hnswlib::HierarchicalNSWStatic<float>(l2space, (R - L + 1), HNSW_M, HNSW_efConstruction);
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
                if (appr_alg->linkLists_[j] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                fin.read(appr_alg->linkLists_[j], linkListSize);
            }

        }
        fin.read((char *) appr_alg->data_level0_memory_,
                 appr_alg->cur_element_count * appr_alg->size_data_per_element_);
    }


    hnswlib::HierarchicalNSWStatic<float> *
    incremental_build(hnswlib::labeltype L, hnswlib::labeltype R, std::ofstream &fout) {
        if ((R - L + 1) < RANG_BOUND) return nullptr;
        hnswlib::labeltype mid = (L + R) >> 1;
        auto left_index = incremental_build(L, mid, fout);
        if (left_index == NULL) {
            auto l2space = new hnswlib::L2Space(D);
            left_index = new hnswlib::HierarchicalNSWStatic<float>(l2space, (R - L + 1), HNSW_M, HNSW_efConstruction);
            std::cerr << "BEGIN RANGE BOTTOM:: " << L << " " << R << std::endl;
#pragma omp parallel for
            for (hnswlib::labeltype i = L; i <= R; i++) {
                left_index->addPoint(
                        hnswlib::HierarchicalNSWStatic<float>::static_base_data_ + i * left_index->data_size_, i - L);
            }
            save_single_static_index(left_index, fout);
            return left_index;
        }
        if (L == 0 && R == N - 1) {
            delete left_index;
        } else {
            left_index->resizeIndex(R - L + 1);
            std::cerr << "BEGIN RANGE ICE:: " << L << " " << R << std::endl;
#pragma omp parallel for
            for (hnswlib::labeltype i = mid + 1; i <= R; i++) {
                left_index->addPoint(
                        hnswlib::HierarchicalNSWStatic<float>::static_base_data_ + i * left_index->data_size_,
                        i - (mid + 1));
            }
            save_single_static_index(left_index, fout);
        }
        auto right_index = incremental_build(mid + 1, R, fout);
        delete right_index;
        return left_index;
    }

    SegmentTree *build_segment_tree(hnswlib::labeltype L, hnswlib::labeltype R, std::ifstream &fin) {
        if ((R - L) < RANG_BOUND) {
            return nullptr;
        }
        auto cur_node = new SegmentTree(L, R);
        hnswlib::labeltype mid = (L + R) >> 1;
        cur_node->left = build_segment_tree(L, mid, fin);
        load_single_static_index(cur_node->appr_alg, fin, L, R);
        cur_node->right = build_segment_tree(mid + 1, R, fin);
        return cur_node;
    }


    void load_index(char *input) {
        std::ifstream fin(input, std::ios::binary);
        root = build_segment_tree(1, N - 1, fin);
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype> >
    bruteforce_range_search(float *query, float *base, unsigned L, unsigned R, unsigned K) {
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


    std::priority_queue<std::pair<float, hnswlib::labeltype> > half_blood_search(SegQuery Q, unsigned K, unsigned nprobs, SegmentTree *cur) {
        if (cur->L <= Q.L && Q.R <= cur->R && (Q.R - Q.L + 1) * 2 > (cur->R - cur->L + 1)) {
            RangeFilter range(Q.L, Q.R);
            return cur->appr_alg->searchKnn(Q.data_, K, &range);
        }
        if (cur->left == nullptr && cur->right == nullptr) {
            return bruteforce_range_search(Q.data_, (float *) (cur->appr_alg->static_base_data_), Q.L,
                                           Q.R, K);
        }
        hnswlib::labeltype mid = (cur->L + cur->R) >> 1;
        std::priority_queue<std::pair<float, hnswlib::labeltype> > res_left, res_right;
        auto QL = Q;
        auto QR = Q;
        QL.R = mid;
        QR.L = mid+1;
        if (cur->left != nullptr) {
            if (check_overlap(QL, cur->L, mid)) {
                res_left = segment_tree_search(QL, K, nprobs, cur->left);
            }
        } else {
            if (check_overlap(QL, cur->L, mid)) {
                res_left = bruteforce_range_search(QL.data_, (float *) (cur->appr_alg->static_base_data_),
                                                   QL.L, QL.R, K);
            }
        }
        if (cur->right != nullptr) {
            if (check_overlap(QR, mid + 1, cur->R)) {
                res_right = segment_tree_search(QR, K, nprobs, cur->right);
            }
        } else {
            if (check_overlap(QR, mid + 1, cur->R)) {
                res_right = bruteforce_range_search(QR.data_, (float *) (cur->appr_alg->static_base_data_),
                                                    QR.L, QR.R, K);
            }
        }
        return merge_res(res_left, res_right);
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype> >
    segment_tree_search(SegQuery Q, unsigned K, unsigned nprobs, SegmentTree *cur) {
        if (Q.L <= cur->L && cur->R <= Q.R) {
            return cur->appr_alg->searchKnn(Q.data_, K);
        }
        if (cur->left == nullptr && cur->right == nullptr) {
            return bruteforce_range_search(Q.data_, (float *) (cur->appr_alg->static_base_data_), std::max(Q.L, cur->L),
                                           std::min(Q.R, cur->R), K);
        }
        hnswlib::labeltype mid = (cur->L + cur->R) >> 1;
        std::priority_queue<std::pair<float, hnswlib::labeltype> > res_left, res_right;
        if (cur->left != nullptr) {
            if (check_overlap(Q, cur->L, mid)) {
                res_left = segment_tree_search(Q, K, nprobs, cur->left);
            }
        } else {
            if (check_overlap(Q, cur->L, mid)) {
                res_left = bruteforce_range_search(Q.data_, (float *) (cur->appr_alg->static_base_data_),
                                                   std::max(Q.L, cur->left->L),
                                                   std::min(Q.R, cur->left->R), K);
            }
        }
        if (cur->right != nullptr) {
            if (check_overlap(Q, mid + 1, cur->R)) {
                res_right = segment_tree_search(Q, K, nprobs, cur->right);
            }
        } else {
            if (check_overlap(Q, mid + 1, cur->R)) {
                res_right = bruteforce_range_search(Q.data_, (float *) (cur->appr_alg->static_base_data_),
                                                    std::max(Q.L, cur->right->L),
                                                    std::min(Q.R, cur->right->R), K);
            }
        }
        return merge_res(res_left, res_right);
    }


    unsigned N, D;
    SegmentTree *root = nullptr;
};


#endif //RANGANN_INDEX2D_H
