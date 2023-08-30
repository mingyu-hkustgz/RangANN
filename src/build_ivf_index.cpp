//
// Created by mingyu on 23-8-30.
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


void load_segment_centroid(std::ifstream &in, Segment::SegmentTree *root) {
    unsigned L, R;
    in.read((char *) &L, sizeof(unsigned));
    in.read((char *) &R, sizeof(unsigned));
    root->index = new Index::IndexIVF(L, R);
    root->index->load_centroid(in);
    if (root->nd_ / Segment::SegmentTree::fan_out >= Segment::SegmentTree::block_bound) {
        for (int i = 0; i < Segment::SegmentTree::fan_out; i++) {
            load_segment_centroid(in, root->children[i]);
        }
    }
}

void build_segment_index(Segment::SegmentTree *root) {
    root->index->build_index();
    if (root->nd_ / Segment::SegmentTree::fan_out >= Segment::SegmentTree::block_bound) {
        for (int i = 0; i < Segment::SegmentTree::fan_out; i++) {
            build_segment_index(root->children[i]);
        }
    }
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
    char dataset[256] = "";
    char centroid_path[256] = "";
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "i:q:n:r:l:c:", longopts, &ind);
        switch (iarg) {
            case 'i':
                if (optarg)strcpy(index_path, optarg);
                break;
            case 'n':
                if (optarg)strcpy(dataset, optarg);
                break;
            case 'c':
                if (optarg)strcpy(centroid_path, optarg);
                break;
        }
    }
    unsigned points_num, dim;
    float *data;
    load_float_data(dataset, data, points_num, dim);

    Segment::SegmentTree::dimension_ = dim;
    Segment::SegmentTree::data_ = data;
    Index::Index::dimension = dim;
    Index::Index::data_ = data;
    auto index = new Segment::SegmentTree(0, points_num);

    auto root = index->build_segment_tree(0, points_num - 1);
    std::ifstream fin(centroid_path, std::ios::binary);
    load_segment_centroid(fin, root);
    build_segment_index(root);
    std::ofstream fout(index_path, std::ios::binary);
    root->save_segment_index(fout);
    return 0;
}

