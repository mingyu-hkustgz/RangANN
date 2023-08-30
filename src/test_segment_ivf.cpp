//
// Created by mingyu on 23-8-21.
//
#include <iostream>
#include "utils.h"
#include "Segment.h"
#include <getopt.h>

#define count_dist
using namespace std;


unsigned Segment::SegmentTree::block_bound = 256;
unsigned Segment::SegmentTree::fan_out = 8;
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
    char dataset[256] = "";
    char logger_path[256] = "";
    int K = 1, efSearch = 8, length_bound;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "i:q:n:s:k:r:b:l:", longopts, &ind);
        switch (iarg) {
            case 'i':
                if (optarg)strcpy(index_path, optarg);
                break;
            case 'q':
                if (optarg)strcpy(query_path, optarg);
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
            case 'b':
                if (optarg)length_bound = atoi(optarg);
            case 'l':
                if (optarg)strcpy(logger_path, optarg);
        }
    }
    unsigned points_num, dim;
    float *data;
    load_float_data(dataset, data, points_num, dim);
    unsigned query_num, query_dim;
    float *query_data;
    load_float_data(query_path, query_data, query_num, query_dim);
    Segment::SegmentTree::dimension_ = dim;
    Segment::SegmentTree::data_ = data;
    Index::Index::dimension = dim;
    Index::Index::data_ = data;
    auto index = new Segment::SegmentTree(0, points_num);

    auto root = index->build_segment_tree(0, points_num - 1);
    root->save_segment(segment_path);

    if (!isFileExists_ifstream(index_path)) return 0;
    std::ifstream fin(index_path, std::ios::binary);
    root->load_segment_index(fin, "ivf");
    fin.close();
    srand(0);
    double segment_recall = 0.0, filter_recall = 0.0;
    double all_index_search_time = 0.0, all_brute_search_time = 0.0, all_filter_search_time = 0.0;
    unsigned brute_node_calc = 0;
    std::cerr << "test begin" << std::endl;
    efSearch = 64;
    std::ofstream fout(logger_path, std::ios::app);
    for (int i = 0; i < query_num; i++) {
        unsigned L = rand() % length_bound;
        unsigned R = rand() % length_bound + L;
        if (R >= points_num) R = points_num - 1;
        brute_node_calc += (R - L + 1);
        SegQuery Q(L, R, query_data + i * dim);
        ResultPool ans1, ans2, ans3;
        auto s = chrono::high_resolution_clock::now();
        root->range_search(Q, efSearch, K, ans1);
        auto e = chrono::high_resolution_clock::now();
        chrono::duration<double> diff = e - s;
        double time_slap1 = diff.count();
        s = chrono::high_resolution_clock::now();
        root->bruteforce_range_search(Q.data_, L, R, K, ans2);
        e = chrono::high_resolution_clock::now();
        diff = e - s;
        double time_slap2 = diff.count();
        s = chrono::high_resolution_clock::now();
        root->index->naive_filter_search(Q, K, efSearch, ans3);
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
    }
    fout << "efSearch:: " << efSearch << std::endl;
    fout << "ave length:: " << brute_node_calc / query_num << std::endl;
    fout << index_dist_calc << " " << brute_node_calc << " " << filter_dist_calc << std::endl;
    fout << "segment recall:: " << segment_recall / query_num << std::endl;
    fout << "filter recall:: " << filter_recall / query_num << std::endl;
    fout << "index search time:: " << all_index_search_time << " brute search time:: " << all_brute_search_time
         << " filter search time:: " << all_filter_search_time << endl;

    return 0;
}

