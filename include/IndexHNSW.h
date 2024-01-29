//
// Created by Mingyu on 23-8-25.
//

#ifndef RANGANN_INDEXHNSW_H
#define RANGANN_INDEXHNSW_H
#define count_dist
#define count_segment

#include "Index.h"
#include <faiss/impl/HNSW.h>

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
#ifdef count_dist
            index_dist_calc++;
#endif
            float curdist = inner_id_dist(currObj, query);
            for (int level = max_level_; level > 0; level--) {
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
#ifdef count_dist
                        index_dist_calc++;
#endif
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
#ifdef count_dist
            filter_dist_calc++;
#endif
            for (int level = max_level_; level > 0; level--) {
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
#ifdef count_dist
                        filter_dist_calc++;
#endif
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
            unsigned ans_size = ans_queue.size();
            if (ans_size > K) ans_size = K;
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
        }

        void save_index(std::ofstream &out) override {
            out.write((char *) &max_level_, sizeof(unsigned));
            out.write((char *) &ep_, sizeof(unsigned));
            out.write((char *) &nd_, sizeof(unsigned));
            for (unsigned id = 0; id < nd_; id++) {
                for (unsigned i = 0; i <= max_level_; i++) {
                    unsigned num = level_graph_[i][id].size();
                    out.write((char *) &num, sizeof(unsigned));
                    if (num != 0) {
                        out.write((char *) level_graph_[i][id].data(), num * sizeof(unsigned));
                    }
                }
            }
        }

        std::vector<unsigned> get_hnsw_neighbor(faiss::HNSW &hnsw, unsigned &id, unsigned &level) {
            std::vector<unsigned> res;
            size_t be, ed;
            hnsw.neighbor_range(id, (int) level, &be, &ed);
            for (size_t j = be; j < ed; j++) {
                int next = hnsw.neighbors.at(j);
                if(next!=-1){
                    res.push_back((unsigned)next);
                }
            }
            return res;
        }


        void build_index(bool verbose) override {
            auto *index = new faiss::IndexHNSWFlat((int) dimension, M);
            index->hnsw.efConstruction = efConst;
            index->verbose = verbose;
            index->add(nd_, data_ + Left_Range);
            max_level_ = index->hnsw.max_level;
            level_graph_.resize(max_level_ + 1);
            for (unsigned i = 0; i <= max_level_; i++) {
                level_graph_[i].resize(nd_);
            }
            if(verbose){
                std::cerr<<"max level:: "<<max_level_<<" "<<"size:: "<<index->hnsw.levels.size()<<std::endl;
            }
            for (unsigned cur_level = 0; cur_level <= max_level_; cur_level++) {
#pragma omp parallel for
                for (unsigned i = 0; i < index->hnsw.levels.size(); i++) {
                    if (cur_level >= index->hnsw.levels.at(i)) {
                        level_graph_[cur_level][i].clear();
                    } else {
                        level_graph_[cur_level][i] = get_hnsw_neighbor(index->hnsw, i, cur_level);
                    }
                }
                std::cerr<<"finish level cur::"<<cur_level<<std::endl;
            }
            ep_ = index->hnsw.entry_point;
            delete index;
        }


    private:
        HierarchicalGraph level_graph_;
        unsigned width{};
        unsigned ep_{};
        unsigned max_level_{};
        int M = 16, efConst = 256;
    };
}


#endif //RANGANN_INDEXHNSW_H
