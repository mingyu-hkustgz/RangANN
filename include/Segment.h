//
// Created by mingyu on 23-8-21.
//
#include "utils.h"
#include "ProximityGraph.h"


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
        SegmentTree *left_child = nullptr, *right_child = nullptr;
        PGraph::PGraph *compact_graph;
        unsigned block_bound, width;
        float *data_;
        unsigned nd_, dimension_;

        SegmentTree(unsigned L, unsigned R, unsigned dim) {
            Left_Range = L;
            Right_Range = R;
            nd_ = R - L + 1;
            dimension_ = dim;
        }

        SegmentTree(unsigned L, unsigned R, float *data, unsigned dim, unsigned w, unsigned block) {
            Left_Range = L;
            Right_Range = R;
            nd_ = R - L + 1;
            width = w;
            block_bound = block;
            data_ = data;
            dimension_ = dim;
        }


        SegmentTree *build_segment_graph(unsigned L, unsigned R) {
            auto temp_root = new SegmentTree(L, R, data_, dimension_, width, block_bound);
            unsigned mid = (L + R) >> 1;
            if (mid - L + 1 > block_bound) {
                temp_root->left_child = build_segment_graph(L, mid);
            }
            if (R - mid > block_bound) {
                temp_root->right_child = build_segment_graph(mid + 1, R);
            }
            std::cout << "build begin:: " << L << " " << mid << " " << R << std::endl;
            temp_root->compact_graph = new PGraph::PGraph(L, R, dimension_, width);
            temp_root->compact_graph->data_ = data_;
            if (temp_root->left_child != nullptr && temp_root->right_child != nullptr) {
                temp_root->compact_graph->merge_build(temp_root->left_child->compact_graph,
                                                      temp_root->right_child->compact_graph);
            } else {
                temp_root->compact_graph->bruteforce_build();
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

        void range_search(SegQuery Q, unsigned pool_size, unsigned K, std::vector<std::pair<float, unsigned >> &ans) {
            if (Q.L <= Left_Range && Q.R >= Right_Range) {
                if (nd_ > block_bound)
                    compact_graph->naive_search(Q.data_, K, pool_size, ans);
                else
                    bruteforce_range_search(Q.data_, Left_Range, Right_Range, K, ans);
                return;
            }
            unsigned mid = (Left_Range + Right_Range) >> 1;
            std::vector<std::pair<float, unsigned >> left_ans, right_ans;
            if (Q.L <= mid) {
                if (left_child != nullptr) left_child->range_search(Q, pool_size, K, left_ans);
                else bruteforce_range_search(Q.data_, Q.L, mid, K, left_ans);
            }
            if (Q.R > mid) {
                if (right_child != nullptr) right_child->range_search(Q, pool_size, K, right_ans);
                else bruteforce_range_search(Q.data_, mid + 1, Q.R, K, right_ans);
            }
            ans = merge_sort(left_ans, right_ans, K);
        }
    };

}

#endif //RANGANN_SEGMENT_H
