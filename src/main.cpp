//
// Created by mingyu on 23-8-21.
//
#include <iostream>
#include "utils.h"
#include "Segment.h"
#include "ProximityGraph.h"
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
    char groundtruth_path[256] = "";
    char dataset[256] = "";
    int K = 100, efSearch;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "i:q:g:n:s:k:", longopts, &ind);
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

    auto index = new Segment::SegmentTree(0, points_num, data,dim, 256, 256);
    auto root = index->build_segment_graph(0, points_num-1);
    double recall = 0.0;
    K = 100;
    for (int i = 0; i < query_num; i++) {
        SegQuery Q;
        std::vector<std::pair<float, unsigned> > ans;
        Q.data_ = query_data + i * query_dim;
        Q.L = points_num/3;
        Q.R = points_num-1;
        root->range_search(Q, 100, 100, ans);
        double temp_recall = test_recall(ans, ground_data + i * ground_dim, K);
        recall += temp_recall;
        std::cout<<temp_recall<<endl;
    }
    recall/=query_num;
    std::cout<<"recall:: "<<recall<<endl;

    return 0;
}

