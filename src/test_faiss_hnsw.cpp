//
// Created by mingyu on 23-8-30.
//
#include <iostream>
#include <fstream>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
using namespace std;

void load_float_data(char *filename, float *&data, unsigned &num,
                     unsigned &dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t) ss;
    num = (unsigned) (fsize / (dim + 1) / 4);
    data = new float[num * dim * sizeof(float)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char *) (data + i * dim), dim * 4);
    }
    in.close();
}

int main(int argc, char *argv[]) {
    unsigned points_num, dim;
    float *data;
    load_float_data(argv[1], data, points_num, dim);
    faiss::IndexHNSWFlat index((int)dim,32);
    index.hnsw.efConstruction=256;
    index.verbose=true;
    index.add(points_num,data);
    return 0;
}

