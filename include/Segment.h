//
// Created by mingyu on 23-8-21.
//
#include "utils.h"
#include "Index.h"
#include "IndexHNSW.h"
#include "IndexIVF.h"

#ifndef RANGANN_SEGMENT_H
#define RANGANN_SEGMENT_H


namespace Segment {

    class SegmentTree {
    public:
        SegmentTree **children = nullptr;
        Index::Index *index = nullptr;
        static SegmentTree *root;
        static unsigned block_bound, fan_out, dimension_;
        static float *data_;
        static bool verbose;
        static std::vector<std::pair<unsigned, unsigned> > Segments;
        unsigned Left_Range, Right_Range;
        unsigned nd_;

        SegmentTree(unsigned L, unsigned R) {
            Left_Range = L;
            Right_Range = R;
            nd_ = R - L + 1;
        }

        SegmentTree *build_segment_tree(unsigned L, unsigned R) {
            auto temp_root = new SegmentTree(L, R);
            Segments.emplace_back(L, R);
            if (verbose)
                std::cerr << L << " " << R << std::endl;
            if (temp_root->nd_ / fan_out >= block_bound) {
                temp_root->children = new SegmentTree *[fan_out];
                unsigned length = temp_root->nd_ / fan_out;
                unsigned cur = temp_root->Left_Range;
                for (int i = 0; i < fan_out; i++) {
                    unsigned sub_l = cur, sub_r = std::min(R, cur + length - 1);
                    temp_root->children[i] = build_segment_tree(sub_l, sub_r);
                    cur += length;
                }
            }
            return temp_root;
        }

        void build_segment_index(const std::string &index_type) {
            if (index_type == "hnsw") index = new Index::IndexHNSW(Left_Range, Right_Range);
            else if (index_type == "ivf") index = new Index::IndexIVF(Left_Range, Right_Range);
            if (this != root)
                index->build_index(verbose);
            if (nd_ / Segment::SegmentTree::fan_out >= Segment::SegmentTree::block_bound) {
                for (int i = 0; i < Segment::SegmentTree::fan_out; i++) {
                    children[i]->build_segment_index(index_type);
                }
            }
        }


        void load_segment_index(std::ifstream &in, const std::string &index_type) {
            unsigned L, R;
            in.read((char *) &L, sizeof(unsigned));
            in.read((char *) &R, sizeof(unsigned));
            if (index_type == "hnsw") index = new Index::IndexHNSW(L, R);
            else if (index_type == "ivf") index = new Index::IndexIVF(L, R);
            if (this != root)
                index->load_index(in);
            if (nd_ / fan_out >= block_bound) {
                for (int i = 0; i < fan_out; i++) {
                    children[i]->load_segment_index(in, index_type);
                }
            }
        }

        void save_segment_index(std::ofstream &out) {
            out.write((char *) &Left_Range, sizeof(unsigned));
            out.write((char *) &Right_Range, sizeof(unsigned));
            if (this != root)
                index->save_index(out);
            if (nd_ / fan_out >= block_bound) {
                for (int i = 0; i < fan_out; i++) {
                    children[i]->save_segment_index(out);
                }
            }
        }


        void save_segment(char *filename) {
            std::ofstream fout(filename);
            fout << Segments.size() << std::endl;
            for (auto u: Segments) {
                fout << u.first << " " << u.second << std::endl;
            }
        }


        static void bruteforce_range_search(const float *query, unsigned L, unsigned R, unsigned K,
                                            std::vector<std::pair<float, unsigned >> &ans) {
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
            if (Q.L <= L && R <= Q.R) return true;
            return false;
        }


        void range_search(SegQuery Q, unsigned pool_size, unsigned K, ResultPool &ans) const {
            if (Q.L <= Left_Range && Q.R >= Right_Range) {
                index->naive_search(Q.data_, K, pool_size, ans);
                return;
            }
            if (children != nullptr) {
                for (int i = 0; i < fan_out; i++) {
                    if (check_overlap(Q, children[i]->Left_Range, children[i]->Right_Range)) {
                        std::vector<std::pair<float, unsigned >> cur_ans;
                        children[i]->range_search(Q, std::max(pool_size >> 1, (unsigned) K), K, cur_ans);
                        ans = merge_sort(ans, cur_ans, K);
                    }
                }
            } else {
#ifdef count_dist
                index_dist_calc += std::min(Q.R, Right_Range) - std::max(Q.L, Left_Range) + 1;
#endif
                bruteforce_range_search(Q.data_, std::max(Q.L, Left_Range), std::min(Q.R, Right_Range), K, ans);
            }
        }

        void merge_range_search(SegQuery Q, unsigned pool_size, unsigned K, ResultPool &ans) const {
            if (Left_Range <= Q.L && Q.R <= Right_Range && (Q.R - Q.L + 1) * 2 > nd_) {
                index->naive_search(Q.data_, K, pool_size, ans);
                return;
            }
            if (children != nullptr) {
                for (int i = 0; i < fan_out; i++) {
                    if (check_overlap(Q, children[i]->Left_Range, children[i]->Right_Range)) {
                        std::vector<std::pair<float, unsigned >> cur_ans;
                        children[i]->merge_range_search(Q, std::max(pool_size >> 1, (unsigned) K), K, cur_ans);
                        ans = merge_sort(ans, cur_ans, K);
                    }
                }
            } else {
#ifdef count_dist
                index_dist_calc += std::min(Q.R, Right_Range) - std::max(Q.L, Left_Range) + 1;
#endif
                bruteforce_range_search(Q.data_, std::max(Q.L, Left_Range), std::min(Q.R, Right_Range), K, ans);
            }
        }


    };
}

#endif //RANGANN_SEGMENT_H
