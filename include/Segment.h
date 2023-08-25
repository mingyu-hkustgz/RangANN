//
// Created by mingyu on 23-8-21.
//
#include "utils.h"
#include "index.h"

#ifndef RANGANN_SEGMENT_H
struct SegQuery {
    unsigned L, R;
    float *data_;
    unsigned dimension_;
};
#define RANGANN_SEGMENT_H
namespace Segment {

    class SegmentTree {
    public:
        unsigned Left_Range, Right_Range;
        SegmentTree **children = nullptr;
        Index::Index *index;
        unsigned block_bound, width, range, fan_out;
        float *data_;
        unsigned nd_, dimension_;

        SegmentTree(unsigned L, unsigned R, unsigned dim) {
            Left_Range = L;
            Right_Range = R;
            nd_ = R - L + 1;
            dimension_ = dim;
        }


        SegmentTree *build_segment_graph(unsigned L, unsigned R) {
            auto temp_root = new SegmentTree(L, R, dimension_);
            if (temp_root->nd_ / fan_out >= block_bound) {
                temp_root->children = new SegmentTree *[fan_out];
                unsigned length = temp_root->nd_ / fan_out;
                unsigned cur = temp_root->Left_Range;
                for (int i = 0; i < fan_out; i++) {
                    unsigned sub_l = cur, sub_r = std::min(R, cur + length - 1);
                    temp_root->children[i] = build_segment_graph(sub_l, sub_r);
                    cur += length;
                }
            }
            return temp_root;
        }

        void bruteforce_range_search(const float *query, unsigned L, unsigned R, unsigned K,
                                     std::vector<std::pair<float, unsigned >> &ans) const {
            std::priority_queue<std::pair<float, unsigned> > Q;
            for (unsigned i = L; i <= R; i++) {
                float dist = naive_l2_dist_calc(query, data_ + i * dimension_, dimension_);
                if (Q.size() < K)
                    Q.emplace(dist, i);
                else if (dist < Q.top().first) {
                    Q.pop();
                    Q.emplace(dist, i);
                }
            }
            while (!Q.empty()) {
                ans.push_back(Q.top());
                Q.pop();
            }
        }

        static bool check_overlap(SegQuery Q, unsigned L, unsigned R) {
            if (L <= Q.L && Q.L <= R) return true;
            if (L <= Q.R && Q.R <= R) return true;
            return false;
        }


        void
        range_search(SegQuery Q, unsigned pool_size, unsigned K, std::vector<std::pair<float, unsigned >> &ans) const {
            if (Q.L <= Left_Range && Q.R >= Right_Range) {
                index->naive_search();
                return;
            }
            std::vector<std::pair<float, unsigned >> cur_ans;
            if (children != nullptr) {
                for (int i = 0; i < fan_out; i++) {
                    if (check_overlap(Q, children[i]->Left_Range, children[i]->Right_Range)) {
                        children[i]->range_search(Q, pool_size, K, ans);
                        ans = merge_sort(ans, cur_ans, K);
                    }
                }
            } else {
                bruteforce_range_search(Q.data_, std::max(Q.L, Left_Range), std::min(Q.R, Right_Range), K, ans);
            }
        }
//
//        void save_index(char *filename,unsigned L,uns){
//            std::ofstream out(filename, std::ios::binary);
//            out.write((char*) &nd_,sizeof(unsigned));
//            out.write((char*) &width,sizeof(unsigned));
//            out.write((char*) &block_bound,sizeof(unsigned));
//            out.write((char*) &dimension_,sizeof(unsigned));
//
//        }
//
//        void load_index(char* filename){
//
//        }
    };

}

#endif //RANGANN_SEGMENT_H
