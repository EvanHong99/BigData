import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import numba
from numba import jit, vectorize
from collections import defaultdict

from math import log

from functools import wraps


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


class PageRankPlus_DF:
    '''

    Attributes
    ----------
    nodes_vals: pandas.Series
        存储所有节点的值
    in_edges: pandas.Series
        入边. 不需要存储完整的结果，因为会稀疏
    out_degrees: pandas.Series
        出度的倒数，节约成本，不需要存储完整的结果，因为会稀疏
    '''

    def __init__(self, threshold=1e-10, max_iter=10000):
        # configs
        self.threshold = threshold
        self.max_iter = max_iter

        self.nnum = 0  # node num
        self.nodes_vals = None
        self.in_edges = None
        self.out_degrees = None

    def load_data(self, filePath):
        '''

        Parameters
        ----------
        filePath

        Returns
        -------
        in_edges: pandas.Series
            记录某个节点有哪些节点指向它。可以通过loc来索引需要的节点
        node_dict: numpy.ndarray
            所有的节点，其中包括dead ends

        '''
        print('begin load data')
        txt = np.loadtxt(filePath, dtype=int)
        edges = pd.DataFrame(data=txt, columns=['from', 'to'])
        from_node = set(edges['from'].values)
        from_num = len(from_node)

        # list的构造函数已经默认保证有序了
        all_nodes = list(from_node.union(set(edges['to'].values)))
        self.nnum = len(all_nodes)
        self.nodes_vals = pd.Series(data=1 / self.nnum, index=all_nodes)

        # 可以通过loc来索引需要的节点
        self.in_edges = edges.groupby('to')['from'].apply(np.array)
        # note 这里存储的是倒数
        self.out_degrees = 1 / edges.groupby('from')['to'].apply(np.array).apply(lambda x: len(x))

        print('load data finish, there are {} nodes, {} dead ends'.format(self.nnum, self.nnum - from_num))

    def is_convergence(self, nodes_old, nodes_new, threshold):
        if abs(nodes_new - nodes_old).sum() > threshold:
            return False
        return True

    @timethis
    def pagerank_basic(self, in_edges: pd.Series, nodes_vals: pd.Series, teleport_rate: float):
        '''
        dataframe实现的PageRank，查阅资料后发现索引特别慢
        Parameters
        ----------
        in_edges
        nodes_vals
        teleport_rate

        Returns
        -------

        '''
        all_nodes = nodes_vals.index
        iter = 0
        while True:
            if iter > self.max_iter:
                print('iter exceed')
                return None
            old_vals = nodes_vals.copy()
            for node_id in all_nodes:
                in_nodes = in_edges.get(node_id)
                if in_nodes is not None:
                    sum = 0
                    for in_node in in_nodes:
                        sum += teleport_rate * self.out_degrees.get(in_node) * old_vals.get(in_node)
                    nodes_vals.loc[node_id] = sum
                else:
                    nodes_vals.loc[node_id] = 0

                # add the value of teleport
                nodes_vals += np.full_like(nodes_vals, (1 - nodes_vals.sum()) / nodes_vals.shape[0])

            if self.is_convergence(old_vals, nodes_vals, self.threshold):
                print('finish iteration after {} times'.format(iter))
                return nodes_vals

            iter += 1

    def pagerank_basic_dict(self, in_edges: pd.Series, nodes_vals: pd.Series, teleport_rate: float):
        '''
        dict实现的PageRank，查阅资料后发现索引特别慢

        Parameters
        ----------
        in_edges
        nodes_vals
        teleport_rate

        Returns
        -------

        '''
        all_nodes = nodes_vals.index
        iter = 0
        while True:
            if iter > self.max_iter:
                print('iter exceed')
                return None
            old_vals = nodes_vals.copy()
            for node_id in all_nodes:
                in_nodes = in_edges.get(node_id)
                if in_nodes is not None:
                    sum = 0
                    for in_node in in_nodes:
                        sum += teleport_rate * self.out_degrees.get(in_node) * old_vals.get(in_node)
                    nodes_vals.loc[node_id] = sum
                else:
                    nodes_vals.loc[node_id] = 0

                # add the value of teleport
                nodes_vals += np.full_like(nodes_vals, (1 - nodes_vals.sum()) / nodes_vals.shape[0])

            if self.is_convergence(old_vals, nodes_vals, self.threshold):
                print('finish iteration after {} times'.format(iter))
                return nodes_vals

            iter += 1


class PageRankPlus_Dict:
    '''

    Attributes
    ----------
    nodes_vals: pandas.Series
        存储所有节点的值
    in_edges: pandas.Series
        入边. 不需要存储完整的结果，因为会稀疏
    out_degrees: pandas.Series
        出度的倒数，节约成本，不需要存储完整的结果，因为会稀疏
    '''

    def __init__(self, threshold=1e-15, max_iter=200):
        # configs
        self.threshold = threshold
        self.max_iter = max_iter

        self.nnum = 0  # node num
        self.nodes_dict = None
        self.in_edges = None
        self.out_degrees = None

    def load_data(self, filePath):
        '''

        Parameters
        ----------
        filePath

        Returns
        -------
        in_edges: pandas.Series
            记录某个节点有哪些节点指向它。可以通过loc来索引需要的节点
        node_dict: numpy.ndarray
            所有的节点，其中包括dead ends

        '''
        print('begin load data')
        txt = np.loadtxt(filePath, dtype=int)
        edges = pd.DataFrame(data=txt, columns=['from', 'to'])
        from_node = set(edges['from'].values)
        from_num = len(from_node)

        # list的构造函数已经默认保证有序了
        all_nodes = list(from_node.union(set(edges['to'].values)))
        self.nnum = len(all_nodes)
        self.nodes_dict = pd.Series(data=1 / self.nnum, index=all_nodes).to_dict()

        # 可以通过loc来索引需要的节点
        self.in_edges = edges.groupby('to')['from'].apply(np.array).to_dict()
        # note 这里存储的是倒数
        self.out_degrees = (1 / edges.groupby('from')['to'].apply(np.array).apply(lambda x: len(x))).to_dict()

        print('load data finish, there are {} nodes, {} dead ends'.format(self.nnum, self.nnum - from_num))

    def is_convergence(self, nodes_old: np.ndarray, nodes_new: np.ndarray, threshold):
        if abs(nodes_new - nodes_old).sum() > threshold:
            return False
        return True

    @timethis
    def pagerank_basic(self, in_edges: defaultdict(list), node_dict: dict, teleport_rate: float):
        '''
        dict实现的PageRank，查阅资料后发现索引特别慢

        Parameters
        ----------
        in_edges
        node_dict
        teleport_rate

        Returns
        -------

        '''
        keys = node_dict.keys()
        key_id_map = dict(zip(keys, np.arange(len(keys))))
        old_vals=np.array(list(node_dict.values()))
        iter = 0
        while True:
            if iter > self.max_iter:
                print('iter exceed')
                return dict(zip(keys, old_vals))
            new_values = np.zeros_like(old_vals)

            for i, node_id in enumerate(keys):
                in_nodes = in_edges.get(node_id)
                if in_nodes is not None:
                    sum = 0
                    for in_node in in_nodes:
                        sum += teleport_rate * self.out_degrees.get(in_node) * old_vals[key_id_map[in_node]]
                    new_values[i] = sum
                else:
                    new_values[i] = 0

            # add the value of teleport
            new_values += np.full(shape=(len(new_values),), fill_value=(1 - new_values.sum()) / len(new_values))
            # print('iter {}. {}'.format(iter,new_values.sum()))

            if self.is_convergence(old_vals, new_values, self.threshold):
                print('finish iteration after {} times'.format(iter))
                node_dict = dict(zip(keys, new_values))
                return node_dict
            old_vals = new_values.copy()
            iter += 1

    def pagerank_block(self, ):

        pass

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
                            out_links.setdefault(edge[0], [0,[]])
                            out_links[edge[0]][1].append(edge[1])
                            out_links[edge[0]][0] += 1
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
                # for node, [degree, links] in self.out_links.items():
                #     for link in links:
                #         self.new_rank[link - 1] += self.alpha * self.old_rank[node - 1] / degree

                self.new_rank = update_rank(self.out_links, self.new_rank, self.old_rank, self.alpha)


            else:
                for block_path in self.blocks:
                    block = np.load(block_path, allow_pickle=True).item()
                    self.new_rank = update_rank(block, self.new_rank, self.old_rank, self.alpha)

                    # for node, links in block.items():
                    #     for link in links:
                    #         self.new_rank[link - 1] += self.alpha * self.old_rank[node - 1] / self.out_degree[node]
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

        return (end_page_rank - start_page_rank)

if __name__ == '__main__':
    # note dataframe版本特别慢，我做了优化，转为dict版本，实现了80倍加速比。查阅资料得知是df索引特别慢
    # p = PageRankPlus_DF(1e-16,200)
    # p.load_data('WikiData.txt')
    # print(p.pagerank_basic(p.in_edges, p.node_dict, 0.85))
    # result=p.node_dict.sort_values(ascending=False).head(100)
    # print(result)
    # result.to_csv('result.csv')

    p = PageRankPlus_Dict(1e-16, 150)  # fixme
    p.load_data('WikiData.txt')
    res = p.pagerank_basic(p.in_edges, p.nodes_dict, 0.85)
    result = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
    import json

    with open('result.json', 'w') as f:
        json.dump(result, f)
