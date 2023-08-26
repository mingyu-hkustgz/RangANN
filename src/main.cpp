//
// Created by mingyu on 23-8-21.
//
#include <iostream>
#include "utils.h"
#include "Segment.h"
#include <getopt.h>

using namespace std;


double test_recall(std::vector<std::pair<float, unsigned >> ans, int *gt, unsigned K) {
    std::unordered_map<unsigned, bool> check_mp;
    double base = (double) K, recall = 0.0;
    for (int i = 0; i < K; i++) check_mp[gt[i]] = true;
    for (auto u: ans) {
        if (check_mp[u.second]) recall += 1.0;
    }
    recall /= base;
    return recall;
}

unsigned Segment::SegmentTree::block_bound = 1000;
unsigned Segment::SegmentTree::fan_out = 10;
unsigned Segment::SegmentTree::dimension_ = 0;
float *Segment::SegmentTree::data_ = nullptr;
std::vector<std::pair<unsigned, unsigned> > Segment::SegmentTree::Segments;
float *Index::Index::data_ = nullptr;
unsigned Index::Index::dimension = 0;


int main(int argc, char *argv[]) {

    const struct option longopts[] = {
            {"dataset",          required_argument, 0, 'n'},
            {"index_path",       required_argument, 0, 'i'},
            {"query_path",       required_argument, 0, 'q'},
            {"groundtruth_path", required_argument, 0, 'g'},
            {"result_path",      required_argument, 0, 'r'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char index_path[256] = "";
    char query_path[256] = "";
    char segment_path[256] = "";
    char groundtruth_path[256] = "";
    char dataset[256] = "";
    int K = 10, efSearch;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "i:q:g:n:s:k:r:", longopts, &ind);
        switch (iarg) {
            case 'i':
                if (optarg)strcpy(index_path, optarg);
                break;
            case 'q':
                if (optarg)strcpy(query_path, optarg);
                break;
            case 'g':
                if (optarg)strcpy(groundtruth_path, optarg);
                break;
            case 'n':
                if (optarg)strcpy(dataset, optarg);
                break;
            case 's':
                if (optarg) efSearch = atoi(optarg);
                break;
            case 'k':
                if (optarg) K = atoi(optarg);
                break;
            case 'r':
                if (optarg)strcpy(segment_path, optarg);
                break;
        }
    }
    unsigned points_num, dim;
    float *data;
    load_float_data(dataset, data, points_num, dim);
    unsigned query_num, query_dim;
    float *query_data;
    load_float_data(query_path, query_data, query_num, query_dim);
    unsigned ground_num, ground_dim;
    int *ground_data;
    load_int_data(groundtruth_path, ground_data, ground_num, ground_dim);
    Segment::SegmentTree::dimension_ = dim;
    Segment::SegmentTree::data_ = data;
    Index::Index::dimension = dim;
    Index::Index::data_ = data;
    auto index = new Segment::SegmentTree(0, points_num);

    auto root = index->build_segment_tree(0, points_num - 1);
    root->save_segment(segment_path);
    std::ifstream fin("./DATA/faiss_sift.hnsw", std::ios::binary);
    root->load_segment_index(fin, "hnsw");
    fin.close();
    srand(0);
    double segment_recall = 0.0, filter_recall = 0.0;
    double all_index_search_time = 0.0, all_brute_search_time = 0.0, all_filter_search_time = 0.0;
    K = 10;
    std::cerr << "test begin" << std::endl;
    for (int i = 0; i < 1000; i++) {
        unsigned L = points_num /15;
        unsigned R = points_num / 6;
        SegQuery Q(L, R, query_data + i * dim);
        ResultPool ans1, ans2, ans3;

        auto s = chrono::high_resolution_clock::now();
        root->range_search(Q, 50, K, ans1);
        auto e = chrono::high_resolution_clock::now();
        chrono::duration<double> diff = e - s;
        double time_slap1 = diff.count();

        s = chrono::high_resolution_clock::now();
        root->bruteforce_range_search(Q.data_, L, R, K, ans2);
        e = chrono::high_resolution_clock::now();
        diff = e - s;
        double time_slap2 = diff.count();

        s = chrono::high_resolution_clock::now();
        root->index->naive_filter_search(Q, K, 500, ans3);
        e = chrono::high_resolution_clock::now();
        diff = e - s;
        double time_slap3 = diff.count();


        double segment = 0, filter = 0;
        unordered_map<unsigned, bool> mp;
        mp.clear();
        for (auto u: ans2) mp[u.second] = true;
        for (auto v: ans1) {
            if (mp[v.second]) segment += 1.0;
        }
        for (auto v: ans3) {
            if (mp[v.second]) filter += 1.0;
        }


        segment /= K;
        filter /= K;
        segment_recall += segment;
        filter_recall += filter;
        all_index_search_time += time_slap1;
        all_brute_search_time += time_slap2;
        all_filter_search_time += time_slap3;
//        std::cerr<<recall<<endl;
    }
    std::cerr << " segment recall:: " << segment_recall / 1000 << std::endl;
    std::cerr << " filter recall:: " << filter_recall / 1000 << std::endl;
    std::cerr << "index time:: " << all_index_search_time << " brute search time:: " << all_brute_search_time<<" filter search time:: "<<all_filter_search_time << endl;

    return 0;
}

