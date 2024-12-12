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
    unsigned length_bound = 1000, K, ef_base;
    opterr = 1;    //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";
    char index_path[256] = "";
    char data_path[256] = "";
    char query_path[256] = "";
    char seg_result_path[256] = "";
    char half_result_path[256] = "";
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
                if (optarg) ef_base = atoi(optarg);
                break;
        }
    }
    sprintf(query_path, "%s%s_query.fvecs", source, dataset);
    sprintf(data_path, "%s%s_base.fvecs", source, dataset);
    sprintf(seg_result_path, "./results/%s/%s_hnsw_seg_2D_%d.log", dataset, dataset, length_bound);
    sprintf(half_result_path, "./results/%s/%s_hnsw_half_2D_%d.log", dataset, dataset, length_bound);
    sprintf(index_path, "./DATA/%s/%s_2D.hnsw", dataset, dataset);
    Matrix<float> X(data_path);
    Matrix<float> Q(query_path);
    hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) X.data;
    Index2D hnsw2D(X.n, X.d);
    std::cerr << index_path << std::endl;
    hnsw2D.load_index(index_path);

    srand(0);
    double segment_recall, half_blood_recall;
    double all_index_search_time, all_half_search_time;
    std::cerr << "test begin" << std::endl;
    std::ofstream segout(seg_result_path, std::ios::app);
    std::ofstream halfout(half_result_path, std::ios::app);
    std::vector<SegQuery> SegQVec;
    std::vector<std::vector<unsigned>> gt;
    unsigned query_num = 1000;
    generata_range_ground_truth_with_fix_length(query_num, X.n, length_bound, Q.d, K, X.data, Q.data, SegQVec, gt);
    std::vector efSearch{1, 2, 4, 8, 16, 32, 50, 64, 128, 150, 256, 300};
    std::cerr<<"Index Memory:: "<<getPeakRSS()<<std::endl;
    for (auto ef: efSearch) {
        ef *= ef_base;
        segment_recall = 0;
        half_blood_recall = 0;
        all_index_search_time = 0;
        all_half_search_time = 0;
        for (int i = 0; i < query_num; i++) {
            ResultQueue ans1, ans2;

            auto s = chrono::high_resolution_clock::now();
            ans1 = hnsw2D.segment_tree_search(SegQVec[i], K, ef, hnsw2D.root);
            auto e = chrono::high_resolution_clock::now();
            chrono::duration<double> diff = e - s;
            double seg_time = diff.count();

            s = chrono::high_resolution_clock::now();
            ans2 = hnsw2D.half_blood_search(SegQVec[i], K, ef, hnsw2D.root);
            e = chrono::high_resolution_clock::now();
            diff = e - s;
            double half_time = diff.count();
            float dist_bound = sqr_dist(SegQVec[i].data_, X.data + gt[i][K - 1] * X.d, X.d);
            double segment = 0, half = 0;
            while (!ans1.empty()) {
                auto v = ans1.top();
                if (v.first <= dist_bound + 1e-6) segment += 1.0;
                ans1.pop();
            }
            segment /= K;
            segment_recall += segment;
            all_index_search_time += seg_time;
            while (!ans2.empty()) {
                auto v = ans2.top();
                if (v.first <= dist_bound + 1e-6) half += 1.0;
                ans2.pop();
            }
            half /= K;
            half_blood_recall += half;
            all_half_search_time += half_time;

        }
        segment_recall /= (double) query_num;
        double Seg_Qps = (double) query_num / all_index_search_time;
        half_blood_recall /= (double) query_num;
        double Half_Qps = (double) query_num / all_half_search_time;
        std::cerr<< "Please Not the Qps need Times 1000"<<std::endl;
        segout << "(" << segment_recall * 100 << "," << Seg_Qps/1000 << ")" << std::endl;
        halfout << "(" << half_blood_recall * 100 << "," << Half_Qps/1000 << ")" << std::endl;
    }
    return 0;
}

