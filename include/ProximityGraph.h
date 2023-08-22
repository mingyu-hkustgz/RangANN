//
// Created by mingyu on 23-8-21.
//

#ifndef RANGANN_PROXIMITYGRAPH_H
#define RANGANN_PROXIMITYGRAPH_H

#include "utils.h"

namespace PGraph {
    class PGraph {
    public:

        PGraph(unsigned L, unsigned R, unsigned dim, unsigned w) {
            nd_ = R - L + 1;
            offset = L;
            dimension_ = dim;
            width = w;
            ep_ = 0;
        }

        float *data_;
        unsigned nd_, dimension_, offset, width, ep_, range = 50;
        typedef std::vector<std::vector<unsigned> > CompactGraph;
        typedef std::vector<std::vector<float> > DistanceGraph;
        CompactGraph temp_graph_, final_graph_;
        DistanceGraph dist_graph_;

        void bruteforce_build() {
            temp_graph_.resize(nd_);
            dist_graph_.resize(nd_);
            final_graph_.resize(nd_);
#pragma omp parallel for
            for (int i = 0; i < nd_; i++) {
                std::priority_queue<std::pair<float, unsigned> > Q;
                for (int j = 0; j < nd_; j++) {
                    if (i == j) continue;
                    float dist = offset_distance(i, j);
                    if (Q.size() < width) {
                        Q.emplace(dist, j);
                    } else if (dist < Q.top().first) {
                        Q.pop();
                        Q.emplace(dist, j);
                    }
                }
                temp_graph_[i].resize(width);
                dist_graph_[i].resize(width);
                for (int j = (int) width - 1; j >= 0; j--) {
                    temp_graph_[i][j] = Q.top().second;
                    dist_graph_[i][j] = Q.top().first;
                    Q.pop();
                }
            }
            rng_edge_occlude();
        }

        void rng_edge_occlude() {
#pragma omp parallel for
            for (int i = 0; i < nd_; i++) {
                int cur_tag = 1;
                final_graph_[i].emplace_back(temp_graph_[i][0]);
                while (final_graph_[i].size() < range && cur_tag < width) {
                    bool check_occlude = false;
                    unsigned candidate = temp_graph_[i][cur_tag];
                    for (auto u: final_graph_[i]) {
                        if (offset_distance(u, candidate) < dist_graph_[i][cur_tag]) {
                            check_occlude = true;
                            break;
                        }
                    }
                    if (!check_occlude) final_graph_[i].emplace_back(candidate);
                    cur_tag++;
                }
            }
            DistanceGraph().swap(dist_graph_);
            CompactGraph().swap(temp_graph_);
        }


        void merge_build(PGraph *left_graph, PGraph *right_graph) {
            unsigned L = left_graph->offset, R = right_graph->offset + right_graph->nd_ - 1;
            temp_graph_.resize(nd_);
            dist_graph_.resize(nd_);
            final_graph_.resize(nd_);
#pragma omp parallel for
            for (unsigned i = L; i <= R; i++) {
                unsigned inner_id = get_inner_id(i);
                float *cur_node = get_float_pointer(inner_id);
                std::vector<std::pair<float, unsigned> > pool;
                left_graph->naive_search(cur_node, width, width, pool);
                right_graph->naive_search(cur_node, width, width, pool);
                std::sort(pool.begin(), pool.end());
                for (auto u: pool) {
                    temp_graph_[inner_id].push_back(get_inner_id(u.second));
                    dist_graph_[inner_id].push_back(u.first);
                }
            }
            rng_edge_occlude();
        }

        __attribute__((always_inline))
        float offset_distance(unsigned i, unsigned j) const {
            return naive_l2_dist_calc(data_ + (i + offset) * dimension_, data_ + (j + offset) * dimension_, dimension_);
        }

        __attribute__((always_inline))
        float offset_distance(unsigned i, const float *query) const {
            return naive_l2_dist_calc(data_ + (i + offset) * dimension_, query, dimension_);
        }

        __attribute__((always_inline))
        unsigned get_outer_id(unsigned id) const {
            return id + offset;
        }

        __attribute__((always_inline))
        unsigned get_inner_id(unsigned id) const {
            return id - offset;
        }

        __attribute__((always_inline))
        float *get_float_pointer(unsigned id) const {
            return data_ + (id + offset) * dimension_;
        }

        void naive_search(const float *query, unsigned int K, unsigned int L, std::vector<std::pair<float, unsigned>> &ans) {
            L = std::min(L, nd_);
            std::vector<Neighbor> retset(L + 1);
            boost::dynamic_bitset<> flags{nd_, 0};
            float cur_dist = offset_distance(ep_, query);
            retset[0] = Neighbor(ep_, cur_dist, true);
            flags[ep_] = true;
            int k = 0;
            int dynamic_length = 1;
            while (k < (int) dynamic_length) {
                int nk = dynamic_length;
                if (retset[k].flag) {
                    retset[k].flag = false;
                    unsigned n = retset[k].id;
                    for (auto id: final_graph_[n]) {
                        if (flags[id]) continue;
                        flags[id] = true;
                        float dist = offset_distance(id, query);
                        if (dist >= retset[dynamic_length - 1].distance && dynamic_length == L) continue;
                        Neighbor nn(id, dist, true);
                        int r = InsertIntoPool(retset.data(), dynamic_length, nn);
                        if (dynamic_length < L) dynamic_length++;
                        if (r < nk) nk = r;
                    }
                }
                if (nk <= k)
                    k = nk;
                else
                    ++k;
            }
            for (unsigned i = 0; i < K; i++) {
                ans.emplace_back(retset[i].distance, get_outer_id(retset[i].id));
            }
        }


    };

}


#endif //RANGANN_PROXIMITYGRAPH_H
