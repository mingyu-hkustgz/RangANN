//
// Created by Mingyu on 23-8-25.
//

#ifndef RANGANN_INDEXIVF_H
#define RANGANN_INDEXIVF_H
#include "Index.h"

namespace Index{
    class IndexIVF: public Index{
    public:

        IndexIVF(unsigned L, unsigned R) {
            Left_Range = L;
            Right_Range = R;
            nd_ = R - L + 1;
        }

        ~IndexIVF(){
            delete [] centroid_;
            delete [] id_;
            delete [] start_id_;
            delete [] cluster_size_;
        }

        void naive_search(const float *query, unsigned int K, unsigned int nprobs, ResultPool &ans) override{
            std::vector<std::pair<float, unsigned> > file_dist_order, candidates;
            for (unsigned i = 0; i < centroid_nd_; i++) {
                float dist = naive_l2_dist_calc(query, centroid_ + i * dimension, dimension);
                file_dist_order.emplace_back(dist, i);
            }
            std::partial_sort(file_dist_order.begin(), file_dist_order.begin() + nprobs, file_dist_order.end());
            unsigned scan_sum_points = 0;
            for (unsigned i = 0; i < nprobs; i++) {
                scan_sum_points += cluster_size_[file_dist_order[i].second];
            }
            candidates.resize(scan_sum_points);
            unsigned cur = 0;
            for (unsigned i = 0; i < nprobs; i++) {
                unsigned cluster_id = file_dist_order[i].second;
                for (unsigned j = 0; j < cluster_size_[cluster_id]; j++) {
                    unsigned id = id_[start_id_[cluster_id] + j];
                    float dist = inner_id_dist(id,query);
                    candidates[cur] = std::make_pair(dist, id);
                    cur++;
                }
            }
            std::partial_sort(candidates.begin(), candidates.begin() + K, candidates.end());
            std::sort(candidates.begin(), candidates.begin() + K);
            for (unsigned i = 0; i < K; i++) ans.emplace_back(candidates[i].first,candidates[i].second + Left_Range);
            std::vector<std::pair<float, unsigned> >().swap(file_dist_order);
            std::vector<std::pair<float, unsigned> >().swap(candidates);
        }

        void load_index(std::ifstream &fin)  override{
            fin.read((char *) &centroid_nd_, sizeof(unsigned));
            fin.read((char *) &centroid_dim_, sizeof(unsigned));
            std::cout<<"centroid num::  "<<centroid_nd_<<" centroid dim:: "<<centroid_dim_<<"\n";
            centroid_ = new float[centroid_nd_ * centroid_dim_];
            for (unsigned i = 0; i < centroid_nd_; i++) {
                fin.read((char *) (centroid_ + i * centroid_dim_), sizeof(float) * centroid_dim_);
            }
            id_ = new unsigned[nd_];
            start_id_ = new unsigned[centroid_nd_];
            cluster_size_ = new unsigned[centroid_nd_];
            fin.read((char *) id_, sizeof(unsigned) * nd_);
            fin.read((char *) start_id_, sizeof(unsigned) * centroid_nd_);
            fin.read((char *) cluster_size_, sizeof(unsigned) * centroid_nd_);
        }

        private:
            float *centroid_;
            unsigned centroid_nd_, centroid_dim_;
            unsigned *id_, *start_id_, *cluster_size_;
    };
}


#endif //RANGANN_INDEXIVF_H
