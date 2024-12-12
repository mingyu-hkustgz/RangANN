import numpy as np
import matplotlib.pyplot as plt
import os
import math
import struct
from tqdm import tqdm

source = '/home/bld/DATA/vector_data'
datasets = ["deep"]
methods = ["half", "general"]
marker = ['o', 'triangle', 'square', 'otimes', 'star', 'diamond', 'pentagon', 'oplus', 'asterisk', 'halfcircle']
half_range = [800000, 600000, 400000, 200000, 100000, 50000]
general_range = [500000, 250000, 125000, 62500, 15625, 3906]
half_index = ["HBI1D", "SERF1D"]
general_index = ["SEG", "HBI2D", "SERF2D", "IRANG"]


def load_result_data(filename):
    f = open(filename)
    tag0, tag1, tag2 = [], [], []
    line = f.readline()
    while line:
        line = line.replace('\n', '').replace('\r', '')
        raw = line.split(' ')
        if raw[0] == "":
            break
        tag0.append(float(raw[0]))
        tag1.append(float(raw[1]))
        line = f.readline()
    f.close()
    return tag0, tag1


if __name__ == "__main__":
    range_map = {"half": half_range, "general": general_range}
    for method in methods:
        out_put_file = open(f"./result-tex-{method}.log", "w")
        count = 0
        for dataset in datasets:
            print(f"Indexing {dataset}")
            real_data = "NONE"
            data_num = 1000000
            if dataset == "deep":
                real_data = "DEEP"
                data_num = 1000000
            base_log = 1000
            file_path = f"./results/{dataset}"
            if not os.path.exists(file_path):
                continue
            if method == "half":
                print("% " + dataset, file=out_put_file)
                print('\\subfigure[' + real_data + ' (Large Range)]{', file=out_put_file)
                for item in range_map[method]:
                    if item <= 200000:
                        break
                    print(r'''
\begin{tikzpicture}[scale=1]
\begin{axis}[
    height=\columnwidth/2.00,
width=\columnwidth/1.50,
xlabel=recall(\%),
ylabel=Qpsx''' + str(base_log) + r''',
title={range=$''' + str(item / data_num) + r'''\times$N},
label style={font=\scriptsize},
tick label style={font=\scriptsize},
title style={font=\scriptsize},
ymajorgrids=true,
xmajorgrids=true,
grid style=dashed,
]''', file=out_put_file)
                    for index in half_index:
                        result_path = f"./results/{dataset}/{dataset}_{index}_{str(item)}.log"
                        if not os.path.exists(result_path):
                            continue
                        recall, Qps = load_result_data(result_path)
                        if index == "HBI1D":
                            print(
                                f"\\addplot[line width=0.15mm,color=navy,mark=otimes,mark size=0.5mm]%HBI 1D {dataset}",
                                file=out_put_file)
                        else:
                            print(
                                f"\\addplot[line width=0.15mm,color=orange,mark=halfcircle,mark size=0.5mm]%SERF1D {dataset}",
                                file=out_put_file)
                        print("plot coordinates {", file=out_put_file)
                        for j in range(len(recall)):
                            if j == 0:
                                continue
                            if round(recall[j], 3) < 70:
                                continue
                            if j > 0 and round(recall[j], 3) == round(recall[j - 1], 3):
                                continue
                            print("    ( " + str(round(recall[j], 3)) + ", " + str(round(Qps[j] / base_log, 3)) + " )",
                                  file=out_put_file)
                        print("};", file=out_put_file)
                    print(r'''
\end{axis}
\end{tikzpicture}\hspace{2mm}''', file=out_put_file)
                print('}', file=out_put_file)

                print('\\subfigure[' + real_data + ' (Small Range)]{', file=out_put_file)
                for item in range_map[method]:
                    if item > 200000:
                        continue
                    print(r'''
\begin{tikzpicture}[scale=1]
\begin{axis}[
    height=\columnwidth/2.00,
width=\columnwidth/1.50,
xlabel=recall(\%),
ylabel=Qpsx''' + str(base_log) + r''',
title={range=$''' + str(item / data_num) + r'''\times$N},
label style={font=\scriptsize},
tick label style={font=\scriptsize},
title style={font=\scriptsize},
ymajorgrids=true,
xmajorgrids=true,
grid style=dashed,
                ]''', file=out_put_file)
                    for index in half_index:
                        result_path = f"./results/{dataset}/{dataset}_{index}_{str(item)}.log"
                        if not os.path.exists(result_path):
                            continue
                        recall, Qps = load_result_data(result_path)
                        if index == "HBI1D":
                            print(
                                f"\\addplot[line width=0.15mm,color=navy,mark=otimes,mark size=0.5mm]%HBI 1D {dataset}",
                                file=out_put_file)
                        else:
                            print(
                                f"\\addplot[line width=0.15mm,color=orange,mark=halfcircle,mark size=0.5mm]%SERF1D {dataset}",
                                file=out_put_file)
                        print("plot coordinates {", file=out_put_file)
                        for j in range(len(recall)):
                            if j == 0:
                                continue
                            if round(recall[j], 3) < 70:
                                continue
                            if j > 0 and round(recall[j], 3) == round(recall[j - 1], 3):
                                continue
                            print("    ( " + str(round(recall[j], 3)) + ", " + str(round(Qps[j] / base_log, 3)) + " )",
                                  file=out_put_file)
                        print("};", file=out_put_file)
                    print(r'''
\end{axis}
\end{tikzpicture}\hspace{2mm}''', file=out_put_file)
                print('}', file=out_put_file)

            if method == "general":
                print("% " + dataset, file=out_put_file)
                print('\\subfigure[' + real_data + ' (Large Range)]{', file=out_put_file)
                for item in range_map[method]:
                    if item < 125000:
                        break
                    print(r'''
\begin{tikzpicture}[scale=1]
\begin{axis}[
    height=\columnwidth/2.00,
width=\columnwidth/1.50,
xlabel=recall(\%),
ylabel=Qpsx''' + str(base_log) + r''',
title={range=$2^{-''' + str(math.log(data_num/item,2)) + r'''}$},
label style={font=\scriptsize},
tick label style={font=\scriptsize},
title style={font=\scriptsize},
ymajorgrids=true,
xmajorgrids=true,
grid style=dashed,
]''', file=out_put_file)
                    for index in general_index:
                        result_path = f"./results/{dataset}/{dataset}_{index}_{str(item)}.log"
                        if not os.path.exists(result_path):
                            continue
                        recall, Qps = load_result_data(result_path)
                        if index == "HBI2D":
                            print(
                                f"\\addplot[line width=0.15mm,color=violate,mark=square,mark size=0.5mm]%HBI 2D {dataset}",
                                file=out_put_file)
                        elif index == "SERF2D":
                            print(
                                f"\\addplot[line width=0.15mm,color=amber,mark=pentagon,mark size=0.5mm]%SERF2D {dataset}",
                                file=out_put_file)
                        elif index == "IRANG":
                            print(
                                f"\\addplot[line width=0.15mm,color=amaranth,mark=o,mark size=0.5mm]%IRANGE {dataset}",
                                file=out_put_file)
                        elif index == "SEG":
                            print(
                                f"\\addplot[line width=0.15mm,color=forestgreen,mark=triangle,mark size=0.5mm]%IRANGE {dataset}",
                                file=out_put_file)
                        print("plot coordinates {", file=out_put_file)
                        for j in range(len(recall)):
                            if j == 0:
                                continue
                            if round(recall[j], 3) < 70:
                                continue
                            if j > 0 and round(recall[j], 3) == round(recall[j - 1], 3):
                                continue
                            print("    ( " + str(round(recall[j], 3)) + ", " + str(round(Qps[j] / base_log, 3)) + " )",
                                  file=out_put_file)
                        print("};", file=out_put_file)
                    print(r'''
\end{axis}
\end{tikzpicture}\hspace{2mm}''', file=out_put_file)
                print('}', file=out_put_file)

                print('\\subfigure[' + real_data + ' (Small Range)]{', file=out_put_file)
                for item in range_map[method]:
                    if item >= 125000:
                        continue
                    print(r'''
\begin{tikzpicture}[scale=1]
\begin{axis}[
    height=\columnwidth/2.00,
width=\columnwidth/1.50,
xlabel=recall(\%),
ylabel=Qpsx''' + str(base_log) + r''',
title={range=$2^{-''' + str(round(math.log(data_num/item,2),0)) + r'''}$},
label style={font=\scriptsize},
tick label style={font=\scriptsize},
title style={font=\scriptsize},
ymajorgrids=true,
xmajorgrids=true,
grid style=dashed,
]''', file=out_put_file)
                    for index in general_index:
                        result_path = f"./results/{dataset}/{dataset}_{index}_{str(item)}.log"
                        if not os.path.exists(result_path):
                            continue
                        recall, Qps = load_result_data(result_path)
                        if index == "HBI2D":
                            print(
                                f"\\addplot[line width=0.15mm,color=violate,mark=square,mark size=0.5mm]%HBI 2D {dataset}",
                                file=out_put_file)
                        elif index == "SERF2D":
                            print(
                                f"\\addplot[line width=0.15mm,color=amber,mark=pentagon,mark size=0.5mm]%SERF2D {dataset}",
                                file=out_put_file)
                        elif index == "IRANG":
                            print(
                                f"\\addplot[line width=0.15mm,color=amaranth,mark=o,mark size=0.5mm]%IRANGE {dataset}",
                                file=out_put_file)
                        elif index == "SEG":
                            print(
                                f"\\addplot[line width=0.15mm,color=forestgreen,mark=triangle,mark size=0.5mm]%IRANGE {dataset}",
                                file=out_put_file)
                        print("plot coordinates {", file=out_put_file)
                        for j in range(len(recall)):
                            if j == 0:
                                continue
                            if round(recall[j], 3) < 70:
                                continue
                            if j > 0 and round(recall[j], 3) == round(recall[j - 1], 3):
                                continue
                            print("    ( " + str(round(recall[j], 3)) + ", " + str(round(Qps[j] / base_log, 3)) + " )",
                                  file=out_put_file)
                        print("};", file=out_put_file)
                    print(r'''
\end{axis}
\end{tikzpicture}\hspace{2mm}''', file=out_put_file)
                print('}', file=out_put_file)