import math

import numpy as np
import faiss
import struct
import os
import tqdm
import argparse
from utils import fvecs_read, fvecs_write

source = '/home/BLD/mingyu/DATA/vector_data'
random_seed = 100


def save_invert_index(centroid, filename, L, R):
    with open(filename, 'ab') as fp:
        left_range = struct.pack('I', int(L))
        right_range = struct.pack('I', int(R))
        fp.write(left_range)
        fp.write(right_range)
        n, d = centroid.shape
        item = struct.pack('I', n)
        fp.write(item)
        item = struct.pack('I', d)
        fp.write(item)
        for y in centroid:
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear regression')
    parser.add_argument('-d', '--dataset', help='dataset', default='sift')
    parser.add_argument('-s', '--segments', help='segments range', default='0')
    parser.add_argument('-e', '--efConstruct', help='centroid number', default='0')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    segment_path = args['segments']
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')
    np.random.seed(random_seed)

    save_path = f"./DATA/faiss_{dataset}.centroid"
    base = fvecs_read(data_path)
    D = base.shape[1]

    with open(segment_path) as f:
        num = int(f.readline())
        for i in range(num):
            x, y = f.readline().split(' ')
            L = int(x)
            R = int(y)
            part_base = base[L:R + 1]
            K = int(math.sqrt(R - L + 1))
            print(f"current num :: {i}, current segment:: {L} <-> {R}")
            index = faiss.index_factory(D, f"IVF{K},Flat")
            index.train(part_base)
            centroids = index.quantizer.reconstruct_n(0, index.nlist)
            save_invert_index(centroids, save_path, L, R)
