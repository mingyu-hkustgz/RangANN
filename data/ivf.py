import numpy as np
import faiss
import struct
import os
import argparse
from utils import fvecs_read, fvecs_write

source = '/home/BLD/mingyu/DATA/vector_data'
random_seed = 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear regression')
    parser.add_argument('-d', '--dataset', help='dataset', default='sift')
    parser.add_argument('-s', '--segments', help='segments range', default='0')
    parser.add_argument('-c', '--centroids', help='construct centroids')


    args = vars(parser.parse_args())
    dataset = args['dataset']
    K = int(args['centroids'])
    segment_path = args['segments']
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')
    np.random.seed(random_seed)

    save_path = f"./DATA/faiss_{dataset}.ivf"
    base = fvecs_read(data_path)
    D = base.shape[1]

    with open(segment_path) as f:
        num = int(f.readline())
        for i in range(num):
            x, y = f.readline().split(' ')
            L = int(x)
            R = int(y)
            part_base = base[L:R + 1]
            print(f"current num :: {i}, current segment:: {L} <-> {R}")
            index = faiss.index_factory(D, f"IVF{K},Flat")
            index.add(part_base)

    print(f"Clustering - {dataset}")
    # path
    path = os.path.join(source, dataset)
    centroids_path = f'./DATA/{dataset}_centroid_{K}.fvecs'

    # read data vectors
    # cluster data vectors
    index = faiss.index_factory(D, f"IVF{K},Flat")
    index.verbose = True
    index.train(X)
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    fvecs_write(f'./DATA/{dataset}_{K}.centroid', centroids)
