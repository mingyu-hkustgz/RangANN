import numpy as np
import faiss
import struct
import os
import argparse
from tqdm import tqdm
from utils import fvecs_read, fvecs_write

source = '/home/DATA/vector_data'
# the number of clusters
K = 256
M = 32
efConstruction = 512
random_seed = 100


def get_neighbors(hnsw, i, level):
    " list the neighbors for node i at level "
    assert i < hnsw.levels.size()
    assert level < hnsw.levels.at(i)
    be = np.empty(2, 'uint64')
    hnsw.neighbor_range(i, level, faiss.swig_ptr(be), faiss.swig_ptr(be[1:]))
    return [hnsw.neighbors.at(j) for j in range(be[0], be[1])]


def get_level_graph(hnsw, level):
    level_graph = []
    for cur_level in range(level):
        graph = []
        for i in tqdm(range(hnsw.levels.size())):
            if cur_level >= hnsw.levels.at(i):
                tmp_neighbors = []
            else:
                tmp_neighbors = get_neighbors(hnsw, i, cur_level)
                while -1 in tmp_neighbors:
                    tmp_neighbors.remove(-1)
            graph.append(tmp_neighbors)
        level_graph.append(graph)
    return level_graph


def save_level_graph(index, level_graph, filename, L, R):
    with open(filename, 'ab') as fp:
        max_level = struct.pack('I', int(index.hnsw.max_level - 1))
        entry_point = struct.pack('I', int(index.hnsw.entry_point))
        nd_ = struct.pack('I', int(index.hnsw.levels.size()))
        left_range = struct.pack('I', int(L))
        right_range = struct.pack('I', int(R))
        fp.write(left_range)
        fp.write(right_range)
        fp.write(max_level)
        fp.write(entry_point)
        fp.write(nd_)
        for i in tqdm(range(index.hnsw.levels.size())):
            for level in range(index.hnsw.max_level):
                size = len(level_graph[level][i])
                item_size = struct.pack('I', int(size))
                fp.write(item_size)
                if size != 0:
                    for node in level_graph[level][i]:
                        item_node = struct.pack('I', int(node))
                        fp.write(item_node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear regression')
    parser.add_argument('-d', '--dataset', help='dataset', default='sift')
    parser.add_argument('-s', '--segments', help='segments range', default='0')
    parser.add_argument('-e', '--efConstruction', help='construct parameters')

    args = vars(parser.parse_args())
    dataset = args['dataset']
    efConstruction = int(args['efConstruction'])
    segment_path = args['segments']
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')
    np.random.seed(random_seed)

    save_path = f"./DATA/faiss_{dataset}.hnsw"
    base = fvecs_read(data_path)

    with open(segment_path) as f:
        num = int(f.readline())
        for i in range(num):
            x, y = f.readline().split(' ')
            L = int(x)
            R = int(y)
            part_base = base[L:R + 1]
            print(f"current num :: {i}, current segment:: {L} <-> {R}")
            index = faiss.index_factory(int(part_base.shape[1]), "HNSW" + str(M))
            hnsw = index.hnsw
            hnsw.efConstruction = efConstruction
            index.add(part_base)
            level_graph = get_level_graph(index.hnsw, index.hnsw.max_level)
            save_level_graph(index, level_graph, save_path, L, R)
