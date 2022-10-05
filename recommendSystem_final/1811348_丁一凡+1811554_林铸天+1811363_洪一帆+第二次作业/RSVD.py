import time
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
林铸天 丁一凡 洪一帆 小组
关于使用RSVD算法进行推荐
"""

def build_lr(total_steps, lr_init=0.0, lr_end=0.0, lr_max=0.1, warmup_steps=0, decay_type='cosine'):
    """
    Applies cosine decay to generate learning rate array.

    Args:
       total_steps(int): all steps in training.
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       lr_max(float): max learning rate.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       list, learning rate array.
    """
    lr_init, lr_end, lr_max = float(lr_init), float(lr_end), float(lr_max)
    decay_steps = total_steps - warmup_steps
    lr_all_steps = []
    inc_per_step = (lr_max - lr_init) / warmup_steps if warmup_steps else 0
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + inc_per_step * (i + 1)
        else:
            if decay_type == 'cosine':
                cosine_decay = 0.5 * (1 + math.cos(math.pi * (i - warmup_steps) / decay_steps))
                lr = (lr_max - lr_end) * cosine_decay + lr_end
            elif decay_type == 'square':
                frac = 1.0 - float(i - warmup_steps) / (total_steps - warmup_steps)
                lr = (lr_max - lr_end) * (frac * frac) + lr_end
            else:
                lr = lr_max
        lr_all_steps.append(lr)

    return lr_all_steps


def txt2df(path):
    print("读取数据集:", path)
    data = []
    with open(path, 'r') as f:

        user = 0
        for line in f:
            if '|' in line:
                user = int(line.split('|')[0])
            else:
                tmp = line.strip()
                tmp = tmp.split(' ')
                item = int(tmp[0])
                if len(tmp) == 3:
                    score = int(tmp[2])
                else:
                    score = np.nan
                data.append([user, item, score])
    return pd.DataFrame(data, columns=['user', 'item', 'score'])


def pred_val(P, Q, test_df):
    print("开始预测")
    for _i, line in enumerate(test_df.values):
        test_df.iloc[_i, 2] = np.matmul(P[int(line[0]), :], Q[:, int(line[1])])
    return test_df


def df2txt(P, Q, test_path, output_path):
    print("开始输出，输出参考user和item来自", test_path, "，输出目标为", output_path)
    with open(test_path, 'r') as f1:
        with open(output_path, 'w') as f2:
            user = 0
            for line in f1:
                if '|' in line:
                    f2.write(line)
                    user = int(line.split('|')[0])
                else:
                    tmp = line.strip()
                    tmp = tmp.split(' ')
                    item = int(tmp[0])
                    row_res = np.matmul(P[user, :], Q[:, item])
                    if row_res >= 100:
                        res = 100
                    elif row_res <= 0:
                        res = 0
                    else:
                        res = int(row_res)
                    f2.write(str(item) + ' ' + str(res) + '\n')

def gradient_descent(P, Q, search_df, lambda_p, lambda_q, lr=0.01):
    start = time.time()

    for _, line in enumerate(search_df.values):  # enumerate(search_df.iterrows()):
        sys.stdout.write('\r梯度下降更新进度-->' + str(round(_ / len(search_df), 4)))
        sys.stdout.flush()
        u = line[0]
        i = line[1]
        s = line[2]

        eui = s - np.matmul(P[u, :], Q[:, i])

        Q[:, i] += lr * eui * P[u, :].T - lambda_q * Q[:, i] * lr
        P[u, :] += lr * eui * Q[:, i].T - lambda_p * P[u, :] * lr

    end = time.time()
    print()
    print("本轮梯度下降运行时间", round(end - start, 2), "s")
    print("梯度下降完成")
    return P, Q, lambda_p, lambda_q


def cal_rmse(ds, P, Q):
    print("开始计算RMSE")
    start = time.time()
    true_val = np.array(ds['score'])
    pred_val = []
    for _i, line in enumerate(ds.values):
        sys.stdout.write('\r计算RMSE进度-->' + str(round(_i / len(ds), 4)))
        sys.stdout.flush()

        u = line[0]
        i = line[1]
        pred_val.append(np.matmul(P[u, :], Q[:, i]))

    rmse = (np.sum((np.array(pred_val) - true_val) ** 2) / len(true_val)) ** 0.5
    end = time.time()
    print()
    print("本轮梯度RMSE计算时间为", round(end - start, 2), "s")
    print("RMSE完成")
    return rmse


if __name__ == '__main__':
    # 参数配置
    factor = 1000
    epoch = 10
    lr = 0.001
    lambda_p = 0
    lambda_q = 0

    # 训练，持续约10min
    train_df = txt2df("data/train.txt")
    test_df = txt2df("data/test.txt")
    U = max(max(train_df['user']), max(test_df['user'])) + 1
    I = max(max(train_df['item']), max(test_df['item'])) + 1
    print("当前参数：", [factor, lr, lambda_p, lambda_q])
    P = np.ones((U, factor)) * (np.sqrt(np.mean(train_df['score']) / factor))
    Q = np.ones((factor, I)) * (np.sqrt(np.mean(train_df['score']) / factor))
    lr_list=build_lr(epoch,lr)
    for e in range(epoch):
        print("----------------------------------------第", e + 1, "轮(共", epoch,
              "轮)迭代开始----------------------------------------")
        P, Q, lambda_p, lambda_q = gradient_descent(P, Q, train_df, lambda_p, lambda_q, lr)


    # 由于参数数据过于庞大，我们并没有在实现RSVD算法时保存参数
    # 预测
    df2txt(P, Q, "./data/test.txt", "./data/result_RSVD.txt")
