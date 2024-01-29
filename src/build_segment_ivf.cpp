//
// Created by mingyu on 23-8-30.
//
#include <iostream>
#include "utils.h"
#include "Segment.h"
#include <getopt.h>

#define count_dist
using namespace std;


unsigned Segment::SegmentTree::block_bound = 1024;
unsigned Segment::SegmentTree::fan_out = 4;
unsigned Segment::SegmentTree::dimension_ = 0;
bool Segment::SegmentTree::verbose = false;
float *Segment::SegmentTree::data_ = nullptr;
Segment::SegmentTree* Segment::SegmentTree::root = nullptr;
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
    char dataset[256] = "";
    unsigned length_bound;
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "i:q:n:r:l:c:", longopts, &ind);
        switch (iarg) {
            case 'i':
                if (optarg)strcpy(index_path, optarg);
                break;
            case 'n':
                if (optarg)strcpy(dataset, optarg);
                break;
            case 'b':
                if (optarg)length_bound = atoi(optarg);
        }
    }
    unsigned points_num, dim;
    float *data;
    load_float_data(dataset, data, points_num, dim);

    Segment::SegmentTree::dimension_ = dim;
    Segment::SegmentTree::data_ = data;
    Index::Index::dimension = dim;
    Index::Index::data_ = data;
    Segment::SegmentTree::verbose= true;
    auto index = new Segment::SegmentTree(0, points_num);
    auto root = index->build_segment_tree(0, points_num - 1);
    Segment::SegmentTree::root = root;
    root->build_segment_index("ivf");
    std::ofstream fout(index_path, std::ios::binary);
    root->save_segment_index(fout);
    return 0;
}

