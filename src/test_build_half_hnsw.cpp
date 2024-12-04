//
// Created by mingyu on 23-8-30.
//
#include <iostream>
#include <fstream>
#include <cstdio>
#include <getopt.h>
#include "utils.h"
#include "IndexFilter.h"

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
    opterr = 1;    //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";
    char index_path[256] = "";
    char data_path[256] = "";
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:s:i:", longopts, &ind);
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
            case 'i':
                if (optarg) {
                    strcpy(index_path, optarg);
                }
                break;
        }
    }
    sprintf(data_path, "%s%s_base.fvecs", source, dataset);
    sprintf(index_path, "%s%s_naive_half.hnsw", index_path, dataset);
    Matrix<float> *X = new Matrix<float>(data_path);
    unsigned N = X->n, D = X->d;
    N>>=1;
    hnswlib::L2Space l2space(D);
    auto appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, N, HNSW_M, HNSW_efConstruction);
    appr_alg->addPoint(X->data, 0);
    unsigned check_tag = 1, report = 50000;
#pragma omp parallel for schedule(dynamic, 144)
    for (int i = 1; i < N; i++) {
        appr_alg->addPoint(X->data + i * D, i);
#pragma omp critical
        {
            check_tag++;
            if (check_tag % report == 0) {
                std::cerr << "Processing - " << check_tag << " / " << N << std::endl;
            }
        }
    }
    appr_alg->saveIndex(index_path);
    return 0;
}

