//
// Created by mingyu on 23-8-30.
//
#include <iostream>
#include <fstream>
#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>
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
    unsigned points_num, learn_num, dim;
    float *data, *learn;
    load_float_data(argv[1], data, points_num, dim);
    load_float_data(argv[2], learn, learn_num, dim);
    const char *index_key = "IVF4096,Flat";
    faiss::Index *index;
    index = faiss::index_factory((int) dim, index_key);
    faiss::IndexIVFFlat *IVF;
    IVF = dynamic_cast<faiss::IndexIVFFlat*>(index);
    IVF->verbose = true;
    IVF->train(learn_num, learn);
    return 0;
}

