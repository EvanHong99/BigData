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
from functools import wraps
from math import log

from pagerank_extension.extensions import update_rank


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : elapse time {}'.format(func.__module__, func.__name__, end - start))
        return r

    return wrapper


class GammaCompressor:
    @staticmethod
    def int_to_bin(int_in):
        """

        :param int_in:
        :return:
            ret: str

        """
        if int_in == 0:
            return '0'
        # 去除第一个位，因此是print(bin(5)) 0b101
        ret = '1' * int(log(int_in, 2)) + '0' + bin(int_in)[3:]
        return ret

    @staticmethod
    def encode(postings_list):
        """Encodes `postings_list`

        Parameters
        ----------
        postings_list: List[int]
            The postings list to be encoded

        Returns
        -------
        bytes:
            bytes reprsentation of the compressed postings list
        """
        ### Begin your code
        encoded_postings_list = ''
        for i in range(0, len(postings_list)):
            # 加一是为了处理0和1的情况
            encoded_postings_list += GammaCompressor.int_to_bin(postings_list[i] + 1)
        return encoded_postings_list.encode()

    @staticmethod
    def decode(encoded_postings):
        """Decodes a byte representation of compressed postings list

        Parameters
        ----------
        encoded_postings: bytes
            Bytes representation as produced by `CompressedPostings.encode`

        Returns
        -------
        List[int]
            Decoded postings list (each posting is a docId)
        """
        res = []
        i = 0
        byteslen = len(encoded_postings)
        encoded_postings = encoded_postings.decode()
        while True:
            length = 0
            while encoded_postings[i] != '0':
                i += 1
                length += 1
            i += 1
            val = int('1' + encoded_postings[i:i + length], 2) - 1
            res.append(val)
            i = i + length
            if i >= byteslen:
                return res


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
        '''
        构建倒排列表
        Parameters
        ----------
        data_path

        Returns
        -------

        '''
        # 不分块处理
        if self.block_num == 1:
            data = np.loadtxt(data_path, dtype=int)
            for edge in data:
                self.out_links.setdefault(edge[0], [0, []])
                self.out_links[edge[0]][1].append(edge[1])
                self.out_links[edge[0]][0] += 1

                # 统计节点
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
                # note 根据节点id分块
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
            # print('iteration times:', i + 1, ', convergence:', convergence)
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


class PageRankPlus:
    def __init__(self, alpha=0.85, max_iter=100, tol=2.0e-16, isCompressed=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.N = 0
        self.new_rank = None
        self.old_rank = None
        self.out_links = {}  # node: [degree, links]
        self.block_num = 0
        self.blocks = []
        self.out_degree = {}
        self.isCompressed = isCompressed

    def load_data(self, data_path, block_num=1, block_id=0, isCompressed=True):
        '''
        加载数据，可以从之前压缩的文件获取，也可以从原始文件获取
        Parameters
        ----------
        block_num
        data_path
        isCompressed

        Returns
        -------

        '''
        # 从压缩文件获取 [dst... src degree],每一行是一个src node
        if isCompressed:
            # 不分块处理
            if block_num == 1:
                # 通过节点id计算节点总数，包括孤立点
                with open('compressed_' + data_path, 'rb') as fr:
                    metalist = fr.readlines()
                    for meta in metalist:
                        m = GammaCompressor.decode(meta.strip())
                        self.out_links.setdefault(m[-2], [0, []])
                        self.out_links[m[-2]][1] = m[:-2]  # posting list
                        self.out_links[m[-2]][0] = m[-1]  # degree

            # 分块处理
            else:
                filename = 'compressed_block' + str(block_id + 1) + '_' + data_path
                with open(filename, 'rb') as fr:
                    metalist = fr.readlines()
                    for meta in metalist:
                        m = GammaCompressor.decode(meta.strip())
                        self.out_links.setdefault(m[-2], [0, []])
                        self.out_links[m[-2]][1] = m[:-2]  # posting list
                        self.out_links[m[-2]][0] = m[-1]  # degree




        else:
            # 不分块处理
            if block_num == 1:
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
                filename = str(block_id + 1) + '.npy'
                block = np.load(filename, allow_pickle=True).item()

                with open(filename, 'rb') as fr:
                    metalist = fr.readlines()
                    for meta in metalist:
                        m = GammaCompressor.decode(meta.strip())
                        self.out_links.setdefault(m[-2], [0, []])
                        self.out_links[m[-2]][1] = m[:-2]  # posting list
                        self.out_links[m[-2]][0] = m[-1]  # degree

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
                    step = int(self.N / block_num)
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

    # @timethis
    def preprocess_data(self, data_path, block_num=1, compress=False):
        '''
        input: original nodes data
        output: write into file with encoded data presented as bytes, [dst... src degree ]形式

        根据是否分块，将文件以不同的形式写入同一个文件（如果分块即用换行来表示）
        且文件不需要全部立即读入内存，只需要留存着FileDescriptor就可以继续读文件，实现分块
        Parameters
        ----------
        data_path
        block_num

        Returns
        -------

        '''
        if compress:
            with open(data_path, 'r') as fr:

                if block_num == 1:
                    with open('compressed_' + data_path, 'wb') as fw:
                        data = np.loadtxt(data_path, dtype=int)  # n*2的一个矩阵
                        for edge in data:
                            # 如果没有这个节点，那么创建，并且以格式 [dst... src degree ] 展现
                            self.out_links.setdefault(edge[0], [0, []])
                            self.out_links[edge[0]][1].append(edge[1])
                            self.out_links[edge[0]][0] += 1
                        for src in self.out_links.keys():
                            # 将src和degree加入list，并写入一行
                            temp = self.out_links.get(src)[1]  # [dst... src degree ]
                            temp.extend([src, self.out_links.get(src)[0]])
                            fw.write(GammaCompressor.encode(temp))
                            fw.write('\n'.encode())
                else:
                    # 统计所有的节点个数
                    self.initialize_rank(data_path)
                    # note 根据节点id分块
                    step = int(self.N / block_num)
                    for block_id, start_node in enumerate(range(1, self.N + 1, step)):
                        fr.seek(0)
                        metadict = {}
                        edge = fr.readline()
                        # 扫描整个文件，并根据当前块存储相应数据
                        while edge:
                            edge = np.array(edge.split()).astype(int)
                            if start_node <= edge[1] < start_node + step:
                                metadict.setdefault(edge[0], [0, []])
                                metadict[edge[0]][1].append(edge[1])
                                metadict[edge[0]][0] += 1
                            edge = fr.readline()
                        filename = 'compressed_block' + str(block_id + 1) + '_' + data_path
                        self.blocks.append(filename)
                        with open(filename, 'wb') as fw:
                            for meta in metadict.items():
                                temp = meta[1][1]
                                temp.extend([meta[0], meta[1][0]])
                                fw.write(GammaCompressor.encode(temp))
                                fw.write('\n'.encode())
        # 不压缩，且需要分块
        else:
            assert block_num > 1
            with open(data_path, 'r') as fr:
                # 统计所有的节点个数
                self.initialize_rank(data_path)

                # note 根据节点id分块
                step = int(self.N / block_num)
                for block_id, start_node in enumerate(range(1, self.N + 1, step)):
                    fr.seek(0)
                    metadict = {}
                    edge = fr.readline()
                    # 扫描整个文件，并根据当前块存储相应数据
                    while edge:
                        edge = np.array(edge.split()).astype(int)
                        if start_node <= edge[1] < start_node + step:
                            metadict.setdefault(edge[0], [0, []])
                            metadict[edge[0]][1].append(edge[1])
                            metadict[edge[0]][0] += 1
                        edge = fr.readline()

                    filename = str(block_id + 1) + '_' + data_path
                    self.blocks.append(filename)
                    with open(filename, 'w') as fw:
                        for meta in metadict.items():
                            temp = meta[1][1]
                            temp.extend([meta[0], meta[1][0]])
                            fw.write(GammaCompressor.encode(temp))
                            fw.write('\n'.encode())

    def initialize_rank(self, data_path):
        # 通过读取源文件统计所有的节点个数
        if self.N == 0:
            with open(data_path, 'r') as fr:
                edge = fr.readline()
                while edge:
                    edge = np.array(edge.split()).astype(int)
                    if int(edge[0]) > self.N:
                        self.N = edge[0]
                    if int(edge[1]) > self.N:
                        self.N = edge[1]
                    edge = fr.readline()
        return np.full(self.N, 1 / self.N, dtype=float)

    def page_rank(self, data_path, block_num):
        old_rank = self.initialize_rank(data_path)
        for i in range(self.max_iter):
            new_rank = np.zeros(self.N, dtype=float)
            if block_num == 1:
                self.load_data(data_path, block_num)
                # for node, [degree, links] in self.out_links.items():
                #     for link in links:
                #         new_rank[link - 1] += self.alpha * old_rank[node - 1] / degree

                # use cython code
                new_rank = update_rank(self.out_links, new_rank, old_rank, self.alpha)
            else:
                for block_id in range(block_num + 1):
                    self.load_data(data_path, block_num=block_num, block_id=block_id, isCompressed=self.isCompressed)
                    print(self.out_links)
                    # for node, [degree, links] in self.out_links.items():
                    #     for link in links:
                    #         new_rank[link - 1] += self.alpha * old_rank[node - 1] / degree

                    # use cython code
                    new_rank = update_rank(self.out_links, new_rank, old_rank, self.alpha)

            new_rank += (1 - new_rank.sum()) / self.N  # temp
            print('iter {}, {}'.format(i, round(new_rank.sum(), 2)))

            convergence = sum(abs(old_rank - new_rank))
            if convergence < self.tol:
                self.new_rank = new_rank
                return
            old_rank = new_rank
        print('iteration exceed')
        self.new_rank = old_rank
        self.new_rank /= self.new_rank.sum()

    def save_result(self, result_path):
        result = sorted(zip(self.new_rank, range(1, self.N + 1)), reverse=True)
        with open(result_path, 'w') as f:
            for i in range(100):
                f.write('[' + str(result[i][1]) + '] [' + str(result[i][0]) + ']\n')
        # if self.block_num > 1:
        #     for block_path in self.blocks:
        #         os.remove(block_path)

    def exec(self, block_num: int, data_path, result_path):
        start = time.perf_counter()
        self.block_num = block_num

        print('Start paging rank')
        start_page_rank = time.perf_counter()
        self.page_rank(data_path, block_num)
        end_page_rank = time.perf_counter()
        print('Running time: %s Seconds' % (end_page_rank - start_page_rank))

        self.save_result(result_path)
        end = time.perf_counter()
        print('Total running time: %s Seconds' % (end - start))


data_path = "WikiData1.txt"
result_path = "result.txt"
block_num = 1
check_res = False

if __name__ == "__main__":
    test = PageRankPlus(isCompressed=True)
    # test.compress_data(data_path, block_num)
    test.exec(block_num=block_num, data_path=data_path, result_path=result_path)

    if check_res:
        import json

        with open('result.json', 'r')   as fr:
            j = json.load(fr)
        with open('result.txt', 'r') as f:
            txt = np.array(f.read().replace('[', ' ').replace(']', ' ').strip().split())[::2]

        print(txt.astype(int) == np.array(list(j.keys())).astype(int)[:100])
