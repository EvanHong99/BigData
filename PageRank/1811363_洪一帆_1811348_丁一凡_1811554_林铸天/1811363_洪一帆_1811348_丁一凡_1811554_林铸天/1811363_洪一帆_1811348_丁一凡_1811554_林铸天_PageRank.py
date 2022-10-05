import time
import numpy as np
import matplotlib.pyplot as plt


class PageRank:
    def __init__(self,
                 beta=0.85,
                 max_iter=100,
                 tol=1.0e-16,
                 block_num=0,
                 data_path='WikiData.txt',
                 result_path='result.txt',
                 report_top_num=100):
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.N = 0
        self.out_links = {}
        self.block_num = block_num
        self.blocks = []
        self.out_degree = {}
        self.data_path = data_path
        self.result_path = result_path
        self.report_top_num = report_top_num
        self.new_rank = None
        self.old_rank = None

    def run_workflow(self):
        self.load_and_process_data()
        start = time.time()

        print('Start paging rank')
        start_page_rank = time.time()
        self.page_rank()
        end_page_rank = time.time()
        print('Running time: %s Seconds' % (end_page_rank - start_page_rank))

        self.save_result()
        end = time.time()
        print('Total running time: %s Seconds' % (end - start))

        statistics = {
            'Block Num.': self.block_num,
            'Alg. Time': end_page_rank - start_page_rank,
            'Total Time': end - start
        }
        return statistics

    def load_and_process_data(self):
        if self.block_num == 1:
            data = np.loadtxt(self.data_path, dtype=int)
            for edge in data:
                self.out_links.setdefault(edge[0], [0, []])
                self.out_links[edge[0]][1].append(edge[1])
                self.out_links[edge[0]][0] += 1

                # 统计节点
                self.N = edge[0] if edge[0] > self.N else self.N
                self.N = edge[1] if edge[1] > self.N else self.N

        else:
            with open(self.data_path, 'r') as f:
                # 统计所有的节点个数
                edge = f.readline()
                while edge:
                    edge = np.array(edge.split()).astype(int)
                    self.N = edge[0] if edge[0] > self.N else self.N
                    self.N = edge[1] if edge[1] > self.N else self.N
                    edge = f.readline()
                # 为节点分块并将块存储到磁盘
                step = int(np.ceil(self.N / self.block_num))
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

    def page_rank(self):
        self.load_and_process_data()
        self.old_rank = np.full(self.N, 1 / self.N, dtype=float)
        for i in range(self.max_iter):
            self.new_rank = np.zeros(self.N, dtype=float)
            if self.block_num == 1:
                # cython here
                for node, [degree, links] in self.out_links.items():
                    for link in links:
                        self.new_rank[link - 1] += self.beta * self.old_rank[node - 1] / degree
            else:
                for block_path in self.blocks:
                    block = np.load(block_path, allow_pickle=True).item()
                    # cython here       
                    for node, links in block.items():
                        for link in links:
                            self.new_rank[link - 1] += self.beta * self.old_rank[node - 1] / self.out_degree[node]
            # cython here
            self.new_rank += (1 - np.sum(self.new_rank)) / self.N
            convergence = np.sum(np.fabs(self.old_rank - self.new_rank))
            print('iteration times:', i + 1, ', convergence:', np.round(convergence, 3))
            if convergence < self.tol:
                break
            self.old_rank = self.new_rank

    def save_result(self):
        result = sorted(zip(self.new_rank, range(1, self.N + 1)), reverse=True)
        with open(self.result_path, 'w') as f:
            for i in range(self.report_top_num):
                f.write(
                    str(result[i][1]) + ' ' + str(result[i][0]) + '\n'
                )


if __name__ == '__main__':
    '''
    此处是可供您检查的代码，若您更改block_num参数，即可以调试分块的情况
    '''
    pr = PageRank(
        beta=0.85,
        max_iter=100,
        tol=1e-12,
        block_num=1,
        data_path='WikiData.txt',
        result_path='result.txt',
        report_top_num=100
    )

    pr.run_workflow().get('Alg. Time')

    '''
    下面是我们跑实验的代码，如果您需要检查最终结果是否正确，请不必解开注释。
    若您想检查我们对于遍历<tol>和<blocks_num>对实验结果的检查，请解开注释
    
    '''
    # block_num_range = range(1, 11)
    # tol_2_time = {
    #     1e-4: [],
    #     1e-8: [],
    #     1e-16: []
    # }
    #
    # for tol in tol_2_time.keys():
    #     for block_num in block_num_range:
    #         pr = PageRank(
    #             beta=0.85,
    #             max_iter=100,
    #             tol=tol,
    #             block_num=block_num,
    #             data_path='WikiData.txt',
    #             result_path='result.txt',
    #             report_top_num=100
    #         )
    #
    #         tol_2_time.get(tol).append(pr.run_workflow().get('Alg. Time'))
    #     plt.plot(list(range(1, 11)), tol_2_time.get(tol), 'o-')
    #
    # print(tol_2_time)
    # plt.legend(tol_2_time.keys())
    # plt.xlabel("Block Num.")
    # plt.ylabel("Time(s)")
    # plt.title("Alg.(Total) Runtime over Block Num. Changing")
    # plt.show()


