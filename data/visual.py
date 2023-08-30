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


if __name__ == "__main__":
    plt.rc('font', family='Times New Roman')
    for dataset in datasets:
        for bound in ["100000", "200000", "400000"]:
            filename = f"./logger/{dataset}_hnsw_result_{bound}.log"
            seg_recall, seg_time, filter_recall, filter_time, brute_time = phrase_log(filename)

            plt.plot(seg_recall, seg_time, marker="o", c="b", label="segment", alpha=0.5, linestyle="--",
                     markerfacecolor='white')
            plt.plot(filter_recall, filter_time, marker="o", c="r", label="filter", alpha=0.5, linestyle="--",
                     markerfacecolor='white')

            recall_max = max(seg_recall)
            recall_max = max(recall_max, max(filter_recall))

            recall_min = min(filter_recall)
            recall_min = min(recall_min, min(seg_recall))

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
            plt.savefig(f'./figure/{dataset}_result_{bound}.png', dpi=600)
            plt.show()
