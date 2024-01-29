import numpy as np
import matplotlib.pyplot as plt
from utils import fvecs_read, fvecs_write
import os
import struct
from tqdm import tqdm
import re

datasets = ["sift", "gist", "deep1M"]


def phrase_log(filename):
    f = open(filename)
    line = f.readline()
    seg_recall = []
    seg_time = []
    filter_recall = []
    filter_time = []
    brute_time = []
    if dataset == "sift":
        query_num = 10000
    else:
        query_num = 1000
    while line:
        raw = line.split(' ')
        if raw[0] == "segment":
            seg_recall.append(float(raw[2]))
        elif raw[0] == "filter":
            filter_recall.append(float(raw[2]))
        elif raw[0] == "index":
            seg_time.append(query_num / float(raw[3]))
            brute_time.append(query_num / float(raw[7]))
            filter_time.append(query_num / float(raw[11]))
        line = f.readline()
    f.close()

    return seg_recall, seg_time, filter_recall, filter_time, brute_time


def reorder_recall(recall, time):
    sort_index = []
    for i in range(len(recall)):
        sort_index.append(i)

    sort_index.sort(key=lambda x: recall[x])

    sort_recall, sort_time = [], []

    for u in sort_index:
        sort_recall.append(recall[u])
        sort_time.append(time[u])

    return sort_recall, sort_time


if __name__ == "__main__":
    plt.rc('font', family='Times New Roman')
    for dataset in datasets:
        for bound in ["100000", "200000", "400000"]:
            for index_type in ["hnsw", "ivf"]:
                filename = f"./logger/{dataset}_{index_type}_result_{bound}.log"
                seg_recall, seg_time, filter_recall, filter_time, brute_time = phrase_log(filename)

                recall_max = max(seg_recall)
                recall_max = max(recall_max, max(filter_recall))

                recall_min = min(filter_recall)
                recall_min = min(recall_min, min(seg_recall))

                seg_recall, seg_time = reorder_recall(seg_recall, seg_time)

                filter_recall, filter_time = reorder_recall(filter_recall, filter_time)

                if index_type == "hnsw":
                    plt.plot(seg_recall, seg_time, marker="o", c="b", label=f"{index_type} segment", alpha=0.5,
                             linestyle="--", markerfacecolor='white')

                if index_type == "hnsw":
                    plt.plot(filter_recall, filter_time, marker="o", c="r", label=f"{index_type} filter", alpha=0.5,
                             linestyle="--", markerfacecolor='white')

                if index_type == "ivf":
                    plt.plot(filter_recall, filter_time, marker="o", c="y", label=f"{index_type} filter", alpha=0.5,
                             linestyle="--", markerfacecolor='white')

                if index_type == "ivf":
                    plt.plot([recall_min, recall_max], [brute_time[0], brute_time[0]], marker="o", c="gray",
                             label="brute force", alpha=0.5,
                             linestyle="--",
                             markerfacecolor='white')

            plt.xlabel("Recall@1")
            plt.ylabel("Qps")
            plt.legend(loc="upper right")
            plt.grid(linestyle='--', linewidth=0.5)
            ax = plt.gca()
            ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
            ratio = int(bound) * 100 // 1000000 // 2
            plt.title(f"{dataset} with ave ratio: {ratio}%")
            plt.savefig(f'./figure/{dataset}_all_result_{bound}.png', dpi=600)
            plt.show()
