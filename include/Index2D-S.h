//
// Created by bld on 24-12-2.
//

#include "hnswlib/hnswlib.h"
#include "hnswlib/hnswalg-static.h"
#include "matrix.h"
#include "utils.h"

#define HNSW2DS_M 16
#define HNSW2DS_efConstruction 200
template<typename dist_t> char *hnswlib::HierarchicalNSWStatic<dist_t>::static_base_data_ = NULL;

class Index2DS {
public:
    Index2DS() {}

    Index2DS(unsigned dim) {
        D = dim;
    }

    Index2DS(unsigned num, unsigned dim) {
        N = num;
        D = dim;
    }

    void build_2D_index_and_save(char *input, char *output) {
        Matrix<float> *X = new Matrix<float>(input);
        D = X->d;
        N = X->n;
        hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) X->data;
        std::ofstream fout(output, std::ios::binary);
        build_and_save(fout);
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
        appr_alg = new hnswlib::HierarchicalNSWStatic<float>(l2space, (R - L + 1), HNSW2DS_M, HNSW2DS_efConstruction);
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


    void build_and_save(std::ofstream &fout) {
        unsigned level = 0;
        while ((N >> level) > RANG_BOUND) {
            size_t length = N % (1 << level) == 0 ? (N >> level) : (N >> level) + 1, delta =
                    length % 2 == 0 ? length >> 1 : (length >> 1) + 1;
            unsigned index_count = (1 << (level + 1)) - 1;
            level_range.resize(level + 1);
            level_index.resize(level + 1);
            level_range[level].resize(index_count);
            level_index[level].resize(index_count);
            size_t id_begin = 0;
            for (int cur = 0; cur < index_count; cur++) {
                unsigned index_length = length;
                if (id_begin + index_length > N) {
                    index_length = N - id_begin;
                }
                auto l2space = new hnswlib::L2Space(D);
                auto index = new hnswlib::HierarchicalNSWStatic<float>(l2space, index_length, HNSW2DS_M,
                                                                       HNSW2DS_efConstruction);
                index->label_begin_ = id_begin;
#pragma omp parallel for
                for (hnswlib::labeltype i = 0; i < index_length; i++) {
                    index->addPoint(
                            hnswlib::HierarchicalNSWStatic<float>::static_base_data_ + (i+id_begin) * index->data_size_, i);
                }
                save_single_static_index(index, fout);
                id_begin += delta;
            }
            level++;
        }
    }


    void load_index(char *input) {
        std::ifstream fin(input, std::ios::binary);
        unsigned level = 0;
        while ((N >> level) > RANG_BOUND) {
            size_t length = N % (1 << level) == 0 ? (N >> level) : (N >> level) + 1, delta =
                    length % 2 == 0 ? length >> 1 : (length >> 1) + 1;
            unsigned index_count = (1 << (level + 1)) - 1;
            level_range.resize(level + 1);
            level_index.resize(level + 1);
            level_range[level].resize(index_count);
            level_index[level].resize(index_count);
            size_t id_begin = 0;
            for (int cur = 0; cur < index_count; cur++) {
                unsigned index_length = length;
                if (id_begin + index_length > N) {
                    index_length = N - id_begin;
                }
                hnswlib::HierarchicalNSWStatic<float> *index = nullptr;
                load_single_static_index(index, fin, id_begin, id_begin + index_length - 1);
                level_range[level].push_back(id_begin);
                level_index[level].push_back(index);
                id_begin += delta;
            }
            level++;
        }
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


    unsigned binary_search(std::vector<hnswlib::labeltype> &id_list, hnswlib::labeltype left_bound) {
        int l = 0, r = id_list.size() - 1, ans = 0;
        while (l <= r) {
            int mid = (l + r) >> 1;
            if (id_list[mid] <= left_bound) {
                ans = mid;
                l = mid + 1;
            } else r = mid - 1;
        }
        return ans;
    }


    std::priority_queue<std::pair<float, hnswlib::labeltype> >
    super_post_search(SegQuery Q, unsigned K, unsigned nprobs) {
        unsigned level = 0;
        int index_id = -1, index_level = -1;
        while ((N >> level) > RANG_BOUND) {
            size_t length = N % (1 << level) == 0 ? (N >> level) : (N >> level) + 1;
            if (Q.R - Q.L + 1 > length) break;
            auto tag = binary_search(level_range[level],Q.L);
            if(level_range[level][tag] + length >= Q.R){
                index_level = level;
                index_id = tag;
            }
            else break;
            level++;
        }
        if (index_id == -1) {
            return bruteforce_search(Q.data_, (float *) (hnswlib::HierarchicalNSWStatic<float>::static_base_data_), Q.L,
                                     Q.R, K);
        } else {
            auto &index = level_index[index_level][index_id];
            RangeFilter range(Q.L - level_range[index_level][index_id], Q.R - level_range[index_level][index_id]);
            level_index[index_level][index_id]->setEf(nprobs);
            return level_index[index_level][index_id]->searchKnn(Q.data_, K, &range);
        }
    }

    unsigned N, D;
    std::vector<std::vector<hnswlib::labeltype>> level_range;
    std::vector<std::vector<hnswlib::HierarchicalNSWStatic<float> *> > level_index;
};

