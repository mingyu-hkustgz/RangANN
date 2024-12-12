from utils import fvecs_read
from utils import fvecs_write, ivecs_read, ivecs_write, mmap_fvecs
import os
import numpy as np
import faiss

source = '/DATA'
datasets = ['deep100M']

def do_compute_gt(xb, xq, topk=100):
    nb, d = xb.shape
    index = faiss.IndexFlatL2(d)
    index.verbose = True
    index.add(xb)
    _, ids = index.search(x=xq, k=topk)
    return ids.astype('int32')


if __name__ == "__main__":
    for dataset in datasets:
        print(f'current dataset: {dataset}')
        path = os.path.join(source, dataset)
        base_path = os.path.join(path, f'{dataset}_base.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        base_data = mmap_fvecs(base_path)
        query_data = fvecs_read(query_path)
        base_data = base_data[:1000000]
        gt = do_compute_gt(base_data, query_data, topk=100)
        save_path = "/DATA/deep"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_base_path = os.path.join(save_path, f'deep_base.fvecs')
        save_query_path = os.path.join(save_path, f'deep_query.fvecs')
        save_ground_path = os.path.join(save_path, f'deep_groundtruth.ivecs')

        fvecs_write(save_base_path, base_data)
        fvecs_write(save_query_path, query_data)
        ivecs_write(save_ground_path, gt)