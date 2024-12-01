//
// Created by mingyu on 23-8-30.
//
#include <iostream>
#include <fstream>
#include <cstdio>
#include <getopt.h>
#include "utils.h"
#include "IndexFilter.h"
#include "Index1D.h"
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
    sprintf(result_path, "./results/%s_hnsw1D.log",dataset);
    sprintf(index_path, "./DATA/%s/%s_1D.hnsw", dataset, dataset);
    Matrix<float> X(data_path);
    Matrix<float> Q(query_path);
    Index1D hnsw1D(X.n, X.d);
    std::cerr<<index_path<<std::endl;
    hnsw1D.load_1D_index(index_path);

    srand(0);
    double segment_recall = 0.0;
    double all_index_search_time = 0.0, all_brute_search_time = 0.0;
    unsigned long long brute_node_calc = 0;
    std::cerr << "test begin" << std::endl;
    std::ofstream fout(result_path, std::ios::app);
    for (int i = 0; i < Q.n; i++) {
        unsigned L = 0;
        unsigned R =  rand()%1000000;
        if (R >= X.n) R = X.n - 1;
        brute_node_calc += (R - L + 1);

        SegQuery SeQ(L, R, Q.data + i * Q.d);
        ResultQueue ans1, ans2;

        auto s = chrono::high_resolution_clock::now();
        ans1 = hnsw1D.naive_range_search(SeQ, K, efSearch);
        auto e = chrono::high_resolution_clock::now();
        chrono::duration<double> diff = e - s;
        double time_slap1 = diff.count();

        s = chrono::high_resolution_clock::now();
        ans2 = bruteforce_range_search(SeQ, X.data, X.d, K);
        e = chrono::high_resolution_clock::now();
        diff = e - s;
        double time_slap2 = diff.count();


        double segment = 0;
        unordered_map<unsigned, bool> mp;
        mp.clear();
        float dist_bound = 0;
        while(!ans2.empty()){
            auto u = ans2.top();
            mp[u.second] = true, dist_bound = std::max(dist_bound, u.first);
            ans2.pop();
        }
        while(!ans1.empty()) {
            auto v = ans1.top();
            if (v.first <= dist_bound) segment += 1.0;
            ans1.pop();
        }
        segment /= K;
        segment_recall += segment;
        all_index_search_time += time_slap1;
        all_brute_search_time += time_slap2;
    }
    fout << "efSearch:: " << efSearch << std::endl;
    fout << "ave length:: " << brute_node_calc /Q.n << std::endl;
    fout << index_dist_calc << " " << brute_node_calc << " " << filter_dist_calc << std::endl;
    fout << "segment recall:: " << segment_recall /Q.n << std::endl;
    fout << "index search time:: " << all_index_search_time << " brute search time:: " << all_brute_search_time << std::endl;
    return 0;
}

