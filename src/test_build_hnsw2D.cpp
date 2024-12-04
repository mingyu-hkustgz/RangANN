//
// Created by mingyu on 23-8-30.
//
#include <iostream>
#include <fstream>
#include <cstdio>
#include <getopt.h>
#include "utils.h"
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
    sprintf(index_path, "%s%s_2D.hnsw", index_path, dataset);
    Index2D hnsw2D;
    hnsw2D.build_2D_index_and_save(data_path, index_path);
    return 0;
}

