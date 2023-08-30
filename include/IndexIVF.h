//
// Created by Mingyu on 23-8-25.
//

#ifndef RANGANN_INDEXIVF_H
#define RANGANN_INDEXIVF_H

#include "Index.h"

namespace Index {
    class IndexIVF : public Index {
    public:

        IndexIVF(unsigned L, unsigned R) {
            Left_Range = L;
            Right_Range = R;
            nd_ = R - L + 1;
        }

        ~IndexIVF() {
            delete[] centroid_;
            delete[] id_;
            delete[] start_id_;
            delete[] cluster_size_;
        }

        void naive_search(const float *query, unsigned int K, unsigned int nprobs, ResultPool &ans) override {
            std::vector<std::pair<float, unsigned> > file_dist_order, candidates;
            nprobs = std::min(nprobs, centroid_nd_);
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
                    float dist = inner_id_dist(id, query);
                    candidates[cur] = std::make_pair(dist, id);
                    cur++;
                }
            }
            std::partial_sort(candidates.begin(), candidates.begin() + K, candidates.end());
            std::sort(candidates.begin(), candidates.begin() + K);
            for (unsigned i = 0; i < K; i++) ans.emplace_back(candidates[i].first, candidates[i].second + Left_Range);
            std::vector<std::pair<float, unsigned> >().swap(file_dist_order);
            std::vector<std::pair<float, unsigned> >().swap(candidates);
        }

        void naive_filter_search(SegQuery Q, unsigned int K, unsigned int nprobs, ResultPool &ans) override {
            std::vector<std::pair<float, unsigned> > file_dist_order;
            std::priority_queue<std::pair<float, unsigned> > ans_queue;
            nprobs = std::min(nprobs, centroid_nd_);
            for (unsigned i = 0; i < centroid_nd_; i++) {
                float dist = naive_l2_dist_calc(Q.data_, centroid_ + i * dimension, dimension);
                file_dist_order.emplace_back(dist, i);
            }
            std::partial_sort(file_dist_order.begin(), file_dist_order.begin() + nprobs, file_dist_order.end());
            for (unsigned i = 0; i < nprobs; i++) {
                unsigned cluster_id = file_dist_order[i].second;
                for (unsigned j = 0; j < cluster_size_[cluster_id]; j++) {
                    unsigned id = id_[start_id_[cluster_id] + j];
                    if (id + Left_Range > Q.R || id + Left_Range < Q.L) continue;
                    float dist = inner_id_dist(id, Q.data_);
                    if (ans_queue.size() < K) ans_queue.emplace(dist, id + Left_Range);
                    else if (dist < ans_queue.top().first) {
                        ans_queue.pop();
                        ans_queue.emplace(dist, id + Left_Range);
                    }
                }
            }
            int queue_size = (int)ans_queue.size();
            ans.resize(K);
            for (int i = queue_size-1; i >= 0; i--) {
                ans[i] = ans_queue.top();
                ans_queue.pop();
            }
            std::vector<std::pair<float, unsigned> >().swap(file_dist_order);
            std::priority_queue<std::pair<float, unsigned> >().swap(ans_queue);
        }


        void build_index(bool verbose = true) {
            std::cerr << "start generate invert file" << std::endl;
            std::vector<std::vector<unsigned> > reorder_map;
            reorder_map.resize(centroid_nd_);
            unsigned logger_sum = 0;
#pragma omp parallel for
            for (unsigned i = 0; i < nd_; i++) {
#pragma omp critical
                if (verbose && ++logger_sum % (nd_ / 20) == 0)
                    std::cerr << "current count :: " << logger_sum << std::endl;
                float dist = naive_l2_dist_calc(data_ + i * dimension, centroid_, dimension);
                unsigned belong = 0;
                for (unsigned j = 1; j < centroid_nd_; j++) {
                    float new_dist = naive_l2_dist_calc(data_ + i * dimension, centroid_ + j * dimension, dimension);
                    if (dist > new_dist) {
                        belong = j;
                        dist = new_dist;
                    }
                }
#pragma omp critical
                reorder_map[belong].push_back(i);
            }
            id_ = new unsigned[nd_];
            start_id_ = new unsigned[centroid_nd_];
            cluster_size_ = new unsigned[centroid_nd_];
            unsigned pre_sum = 0;
            for (unsigned i = 0; i < centroid_nd_; i++) {
                start_id_[i] = pre_sum;
                cluster_size_[i] = reorder_map[i].size();
                pre_sum += reorder_map[i].size();
            }
#pragma omp parallel for
            for (unsigned i = 0; i < centroid_nd_; i++) {
                for (unsigned j = 0; j < reorder_map[i].size(); j++) {
                    id_[start_id_[i] + j] = reorder_map[i][j];
                }
            }
            std::cerr << "finish generate invert file" << std::endl;
            std::vector<std::vector<unsigned> >().swap(reorder_map);
        }


        void load_centroid(std::ifstream &fin) override {
            fin.read((char *) &centroid_nd_, sizeof(unsigned));
            fin.read((char *) &centroid_dim_, sizeof(unsigned));
            centroid_ = new float[centroid_nd_ * centroid_dim_];
            for (unsigned i = 0; i < centroid_nd_; i++) {
                fin.read((char *) (centroid_ + i * centroid_dim_), sizeof(float) * centroid_dim_);
            }
        }


        void load_index(std::ifstream &fin) override {
            fin.read((char *) &centroid_nd_, sizeof(unsigned));
            fin.read((char *) &centroid_dim_, sizeof(unsigned));
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


        void save_index(std::ofstream &fout) override {
            fout.write((char *) &centroid_nd_, sizeof(unsigned));
            fout.write((char *) &centroid_dim_, sizeof(unsigned));
            for (unsigned i = 0; i < centroid_nd_; i++) {
                fout.write((char *) (centroid_ + i * centroid_dim_), sizeof(float) * centroid_dim_);
            }
            fout.write((char *) id_, sizeof(unsigned) * nd_);
            fout.write((char *) start_id_, sizeof(unsigned) * centroid_nd_);
            fout.write((char *) cluster_size_, sizeof(unsigned) * centroid_nd_);
        }

    private:
        float *centroid_;
        unsigned centroid_nd_, centroid_dim_;
        unsigned *id_, *start_id_, *cluster_size_;
    };
}


#endif //RANGANN_INDEXIVF_H
