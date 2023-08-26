import numpy as np
import faiss
import struct
import os
import argparse
from utils import fvecs_read, fvecs_write

source = '/home/BLD/mingyu/DATA/vector_data'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear regression')
    parser.add_argument('-d', '--dataset', help='dataset', default='sift')
    parser.add_argument('-l', '--left', help='left range', default='0')
    parser.add_argument('-r', '--right', help='right range', default='0')
    parser.add_argument('-k', '--K', help='K centroid')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    L = args['left']
    R = args['right']
    K = args['K']
    print(f"Clustering - {dataset}")
    # path
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_learn.fvecs')
    centroids_path = f'./DATA/{dataset}_centroid_{K}.fvecs'

    # read data vectors
    X = fvecs_read(data_path)
    D = X.shape[1]
    X = X[L:R]
    # cluster data vectors
    index = faiss.index_factory(D, f"IVF{K},Flat")
    index.verbose = True
    index.train(X)
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    fvecs_write(f'./DATA/{dataset}_{K}.centroid', centroids)
