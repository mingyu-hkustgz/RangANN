//
// Created by mingyu on 23-8-30.
//
#include <iostream>
#include <fstream>
#include <cstdio>
#include <getopt.h>
#include "utils.h"
#include "IndexFilter.h"
#include "Index2D.h"

using namespace std;

int main(int argc, char *argv[]) {
    const struct option longopts[] = {
            // General Parameter
            {"help",    no_argument,       0, 'h'},

            // Indexing Path
            {"dataset", required_argument, 0, 'd'},
            {"source",  required_argument, 0, 's'},
    };

    int ind;
    int iarg = 0;
    unsigned length_bound = 1000, K, efSearch;
    opterr = 1;    //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";
    char index_path[256] = "";
    char data_path[256] = "";
    char query_path[256] = "";
    char result_path[256] = "";
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:s:l:k:e:", longopts, &ind);
        switch (iarg) {
            case 'd':
                if (optarg) {
                    strcpy(dataset, optarg);
                }
                break;
            case 's':
                if (optarg) {
                    strcpy(source, optarg);
                }
                break;
            case 'l':
                if (optarg) length_bound = atoi(optarg);
                break;
            case 'k':
                if (optarg) K = atoi(optarg);
                break;
            case 'e':
                if (optarg) efSearch = atoi(optarg);
                break;
        }
    }
    sprintf(query_path, "%s%s_query.fvecs", source, dataset);
    sprintf(data_path, "%s%s_base.fvecs", source, dataset);
    sprintf(result_path, "./results/%s_hnsw2D.log", dataset);
    sprintf(index_path, "./DATA/%s/%s_2D.hnsw", dataset, dataset);
    Matrix<float> X(data_path);
    Matrix<float> Q(query_path);
    hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) X.data;
    Index2D hnsw2D(X.n, X.d);
    std::cerr << index_path << std::endl;
    hnsw2D.load_index(index_path);

    srand(0);
    double segment_recall = 0.0, half_blood_recall = 0.0;
    double all_index_search_time = 0.0, all_half_search_time = 0.0, all_brute_search_time = 0.0;
    unsigned long long brute_node_calc = 0;
    std::cerr << "test begin" << std::endl;
    std::ofstream fout(result_path, std::ios::app);
    std::cerr << K << std::endl;
    unsigned query_num = 1000;
    for (int i = 0; i < query_num; i++) {
        unsigned L = 0;
        unsigned R = rand() % length_bound;
        if (R >= X.n) R = X.n - 1;
        brute_node_calc += (R - L + 1);

        SegQuery SeQ(L, R, Q.data + i * Q.d);
        ResultQueue ans1, ans2, ans3;

        auto s = chrono::high_resolution_clock::now();
        ans1 = hnsw2D.segment_tree_search(SeQ, K, efSearch, hnsw2D.root);
        auto e = chrono::high_resolution_clock::now();
        chrono::duration<double> diff = e - s;
        double time_slap1 = diff.count();

        s = chrono::high_resolution_clock::now();
        ans2 = hnsw2D.half_blood_search(SeQ, K, efSearch, hnsw2D.root);
        e = chrono::high_resolution_clock::now();
        diff = e - s;
        double time_slap2 = diff.count();

        s = chrono::high_resolution_clock::now();
        ans3 = bruteforce_range_search(SeQ, X.data, X.d, K);
        e = chrono::high_resolution_clock::now();
        diff = e - s;
        double time_slap3 = diff.count();


        double segment = 0, half = 0;
        unordered_map<unsigned, bool> mp;
        mp.clear();
        float dist_bound = 0, dist_index, dist_half;
        unsigned gt, index_gt, half_gt;
        while (!ans3.empty()) {
            auto u = ans3.top();
            mp[u.second] = true, dist_bound = std::max(dist_bound, u.first);
            gt = u.second;
            ans3.pop();
        }
        while (!ans2.empty()) {
            auto v = ans2.top();
            dist_half = v.first;
            half_gt = v.second;
//            if (v.first <= dist_bound) half += 1.0;
            ans2.pop();
        }
        if (half_gt == gt) half += 1.0;
        while (!ans1.empty()) {
            auto v = ans1.top();
            dist_index = v.first;
            index_gt = v.second;
//            if (v.first <= dist_bound) segment += 1.0;
            ans1.pop();
        }
        if (index_gt == gt) segment += 1.0;
        if (segment == 0)
        {
            std::cerr<<i<<std::endl;
            std::cerr << L << " " << R << " " << dist_bound << " " << dist_half << " " << dist_index << std::endl;
            std::cerr<<gt<<" " <<half_gt<<" "<<index_gt<<std::endl;
        }

        segment /= K;
        half /= K;
        segment_recall += segment;
        half_blood_recall += half;
        all_index_search_time += time_slap1;
        all_half_search_time += time_slap2;
        all_brute_search_time += time_slap3;
    }
    fout << "efSearch:: " << efSearch << std::endl;
    fout << "ave length:: " << brute_node_calc /query_num << std::endl;
    fout << index_dist_calc << " " << brute_node_calc << " " << filter_dist_calc << std::endl;
    fout << "segment recall:: " << segment_recall /query_num << " half recall:: " << half_blood_recall /query_num
         << std::endl;
    fout << "index search time:: " << all_index_search_time << "  half search time:: " << all_half_search_time
         << "  brute search time:: " << all_brute_search_time
         << std::endl;
    return 0;
}

