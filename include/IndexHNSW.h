//
// Created by Mingyu on 23-8-25.
//

#ifndef RANGANN_INDEXHNSW_H
#define RANGANN_INDEXHNSW_H

#include "Index.h"

namespace Index {
    class IndexHNSW : public Index {
    public:

        IndexHNSW(unsigned L, unsigned R) {
            Left_Range = L;
            Right_Range = R;
            nd_ = R - L + 1;
        }


        ~IndexHNSW() {
            HierarchicalGraph().swap(level_graph_);
        }

        void naive_search(const float *query, unsigned int K, unsigned int nprobs, ResultPool &ans) override {
            unsigned currObj = ep_;
            float curdist = inner_id_dist(currObj, query);
            for (unsigned level = max_level_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (const auto &u: level_graph_[level][currObj]) {
                        float d = inner_id_dist(u, query);
                        if (d < curdist) {
                            curdist = d;
                            currObj = u;
                            changed = true;
                        }
                    }
                }
            }

            std::vector<Neighbor> retset(nprobs + 1);
            boost::dynamic_bitset<> flags{nd_, 0};
            retset[0] = Neighbor(currObj, curdist, true);
            flags[currObj] = true;
            int k = 0;
            int dynamic_length = 1;
            while (k < (int) dynamic_length) {
                int nk = dynamic_length;
                if (retset[k].flag) {
                    retset[k].flag = false;
                    unsigned n = retset[k].id;
                    for (auto id: level_graph_[0][n]) {
                        if (flags[id]) continue;
                        flags[id] = true;
                        float dist = inner_id_dist(id, query);
                        if (dist >= retset[dynamic_length - 1].distance && dynamic_length == nprobs) continue;
                        Neighbor nn(id, dist, true);
                        int r = InsertIntoPool(retset.data(), dynamic_length, nn);
                        if (dynamic_length < nprobs) dynamic_length++;
                        if (r < nk) nk = r;
                    }
                }
                if (nk <= k)
                    k = nk;
                else
                    ++k;
            }
            for (unsigned i = 0; i < K; i++) {
                ans.emplace_back(retset[i].distance, retset[i].id + Left_Range);
            }
        }


        void naive_filter_search(SegQuery Q, unsigned K, unsigned nprobs, ResultPool &ans) override {
            std::priority_queue<std::pair<float, unsigned>> ans_queue;
            unsigned currObj = ep_;
            float curdist = inner_id_dist(currObj, Q.data_);
            for (unsigned level = max_level_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (const auto &u: level_graph_[level][currObj]) {
                        float d = inner_id_dist(u, Q.data_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = u;
                            changed = true;
                        }
                    }
                }
            }

            std::vector<Neighbor> retset(nprobs + 1);
            boost::dynamic_bitset<> flags{nd_, 0};
            retset[0] = Neighbor(currObj, curdist, true);
            flags[currObj] = true;
            int k = 0;
            int dynamic_length = 1;
            while (k < (int) dynamic_length) {
                int nk = dynamic_length;
                if (retset[k].flag) {
                    retset[k].flag = false;
                    unsigned n = retset[k].id;
                    for (auto id: level_graph_[0][n]) {
                        if (flags[id]) continue;
                        flags[id] = true;
                        float dist = inner_id_dist(id, Q.data_);
                        unsigned outer_id = id + Left_Range;
                        if (Q.L <= outer_id && outer_id <= Q.R) {
                            if (ans_queue.empty() || ans_queue.size() < K) ans_queue.emplace(dist, outer_id);
                            else if (dist < ans_queue.top().first) {
                                ans_queue.pop();
                                ans_queue.emplace(dist, outer_id);
                            }
                        }

                        if (dist >= retset[dynamic_length - 1].distance && dynamic_length == nprobs) continue;
                        Neighbor nn(id, dist, true);
                        int r = InsertIntoPool(retset.data(), dynamic_length, nn);
                        if (dynamic_length < nprobs) dynamic_length++;
                        if (r < nk) nk = r;
                    }
                }
                if (nk <= k)
                    k = nk;
                else
                    ++k;
            }
            unsigned ans_size = std::min((unsigned long) K, ans_queue.size());
            ans.resize(K);
            for (int i = (int) ans_size - 1; i >= 0; i--) {
                ans[i] = ans_queue.top();
                ans_queue.pop();
            }
        }


        void load_index(std::ifstream &in) override {
            in.read((char *) &max_level_, sizeof(unsigned));
            in.read((char *) &ep_, sizeof(unsigned));
            in.read((char *) &nd_, sizeof(unsigned));
            level_graph_.resize(max_level_ + 1);
//            std::cout << "Graph label " << max_level_ << " " << ep_ << " " << nd_ << "\n";
            for (unsigned i = 0; i <= max_level_; i++) {
                level_graph_[i].resize(nd_);
            }
            for (unsigned id = 0; id < nd_; id++) {
                for (unsigned i = 0; i <= max_level_; i++) {
                    unsigned num;
                    in.read((char *) &num, sizeof(unsigned));
                    if (num != 0) {
                        std::vector<unsigned> tmp(num);
                        in.read((char *) tmp.data(), num * sizeof(unsigned));
                        level_graph_[i][id] = tmp;
                    }
                }
            }
//            std::cerr << "graph nodes " << level_graph_[0].size() << "\n";
//            std::cerr << "begin node is " << ep_ << " first steps is " << level_graph_[max_level_][ep_].size() << "\n";
        }


    private:
        HierarchicalGraph level_graph_;
        unsigned width{};
        unsigned ep_{};
        unsigned max_level_{};
    };
}


#endif //RANGANN_INDEXHNSW_H
