# -*- coding:utf-8 -*-
# @Time： 4/9/21 9:07 AM
# @Author: dyf-2316
# @FileName: PageRank.py
# @Software: PyCharm
# @Project: PageRank
# @Description:
import os
import time
import numpy as np


class PageRank:
    def __init__(self, alpha=0.85, max_iter=100, tol=2.0e-16):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.N = 0
        self.new_rank = None
        self.old_rank = None
        self.out_links = {}
        self.block_num = 0
        self.blocks = []
        self.out_degree = {}

    def load_data(self, data_path):
        # 不分块处理
        if self.block_num == 1:
            data = np.loadtxt(data_path, dtype=int)
            for edge in data:
                self.out_links.setdefault(edge[0], [0, []])
                self.out_links[edge[0]][1].append(edge[1])
                self.out_links[edge[0]][0] += 1
                if edge[0] > self.N:
                    self.N = edge[0]
                if edge[1] > self.N:
                    self.N = edge[1]
        # 分块处理
        else:
            with open(data_path, 'r') as f:
                # 统计所有的节点个数
                edge = f.readline()
                while edge:
                    edge = np.array(edge.split()).astype(int)
                    if int(edge[0]) > self.N:
                        self.N = edge[0]
                    if int(edge[1]) > self.N:
                        self.N = edge[1]
                    edge = f.readline()
                # 为节点分块并将块存储到磁盘
                step = int(self.N / self.block_num)
                for block_id, start_node in enumerate(range(1, self.N + 1, step)):
                    f.seek(0)
                    out_links = {}
                    edge = f.readline()
                    while edge:
                        edge = np.array(edge.split()).astype(int)
                        if start_node <= edge[1] < start_node + step:
                            out_links.setdefault(edge[0], [])
                            out_links[edge[0]].append(edge[1])
                            self.out_degree.setdefault(edge[0], 0)
                            self.out_degree[edge[0]] += 1
                        edge = f.readline()
                    self.blocks.append('block' + str(block_id + 1) + '.npy')
                    np.save(self.blocks[-1], out_links)

    def initialize_rank(self):
        self.old_rank = np.full(self.N, 1 / self.N, dtype=float)

    def page_rank(self):
        for i in range(self.max_iter):
            self.new_rank = np.zeros(self.N, dtype=float)
            if self.block_num == 1:
                for node, [degree, links] in self.out_links.items():
                    for link in links:
                        self.new_rank[link - 1] += self.alpha * self.old_rank[node - 1] / degree
            else:
                for block_path in self.blocks:
                    block = np.load(block_path, allow_pickle=True).item()
                    for node, links in block.items():
                        for link in links:
                            self.new_rank[link - 1] += self.alpha * self.old_rank[node - 1] / self.out_degree[node]
            self.new_rank += (1 - np.sum(self.new_rank)) / self.N
            convergence = np.sum(np.fabs(self.old_rank - self.new_rank))
            print('iteration times:', i + 1, ', convergence:', convergence)
            if convergence < self.tol:
                break
            self.old_rank = self.new_rank

    def save_result(self, result_path):
        result = sorted(zip(self.new_rank, range(1, self.N + 1)), reverse=True)
        with open(result_path, 'w') as f:
            for i in range(100):
                f.write('[' + str(result[i][1]) + '] [' + str(result[i][0]) + ']\n')
        if self.block_num > 1:
            for block_path in self.blocks:
                os.remove(block_path)

    def exec(self, block_num, data_path, result_path):
        start = time.perf_counter()
        self.block_num = block_num

        print('Start loading data')
        start_load_data = time.perf_counter()
        self.load_data(data_path)
        end_load_data = time.perf_counter()
        print('Running time: %s Seconds' % (end_load_data - start_load_data))

        self.initialize_rank()

        print('Start paging rank')
        start_page_rank = time.perf_counter()
        self.page_rank()
        end_page_rank = time.perf_counter()
        print('Running time: %s Seconds' % (end_page_rank - start_page_rank))

        self.save_result(result_path)
        end = time.perf_counter()
        print('Total running time: %s Seconds' % (end - start))


data_path = "WikiData.txt"
result_path = "result.txt"
block_num = 1

if __name__ == "__main__":
    test = PageRank()
    test.exec(block_num=block_num, data_path=data_path, result_path=result_path)
