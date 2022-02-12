# coding: utf-8
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np

# 打印log
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

# metric
def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, rmse, mape


def mae_loss(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.tensor(0.0), mask)

    loss = torch.abs(torch.sub(pred, label))
    loss *= mask
    loss = torch.where(torch.isnan(loss), torch.tensor(0.0), loss)

    loss = torch.mean(loss)
    return loss


def plot_train_val_loss(train_total_loss, val_total_loss, file_path):
    """绘制损失"""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
    plt.plot(range(1, len(val_total_loss) + 1), val_total_loss, c='r', marker='o', label='Validation')
    plt.legend(loc='best')
    plt.title('Train loss vs Validation loss')
    plt.savefig(file_path)


def save_test_result(trainPred, trainY, valPred, valY, testPred, testY):
    """保存测试结果"""
    with open('./figure/test_results.txt', 'w+') as f:
        for l in (trainPred, trainY, valPred, valY, testPred, testY):
            f.write(list(l))


def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seq2instance(data, num_his, num_pred):
    """生成样本"""
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims)
    y = torch.zeros(num_sample, num_pred, dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]

    return x, y


# 数据
def load_data(args):
    df = pd.read_hdf(args.traffic_file)
    traffic = torch.from_numpy(df.values)

    # train/val/test
    num_steps = df.shape[0]
    train_steps = round(args.train_ratio * num_steps)
    test_steps = round(args.test_ratio * num_steps)
    val_steps = num_steps - train_steps - test_steps

    train = traffic[:train_steps]
    val = traffic[train_steps: train_steps + val_steps]
    test = traffic[-test_steps:]

    # X, Y
    trainX, trainY = seq2instance(train, args.num_his, args.num_pred)
    valX, valY = seq2instance(val, args.num_his, args.num_pred)
    testX, testY = seq2instance(test, args.num_his, args.num_pred)

    # 归一化
    mean, std = torch.mean(trainX), torch.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # 空间嵌入，node2vec
    with open(args.SE_file, mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        num_vertex, dims = int(temp[0]), int(temp[1])  # 顶点数，维度
        SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = torch.tensor([float(ch) for ch in temp[1:]])

    # 时间嵌入，加入day_of_time和day_of_week作为嵌入表示
    time = pd.DatetimeIndex(df.index)  # 这个直接就获得时序戳
    dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))  # 获得每条数据的星期几数据

    timeofday = (time.hour*3600 + time.minute*60 + time.second) // (5 * 60)  # 获得每条数据是第几个5分钟
    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))

    time = torch.cat((dayofweek, timeofday), -1)

    train = time[:train_steps]
    val = time[train_steps:train_steps + val_steps]
    test = time[-test_steps:]

    trainTE = seq2instance(train, args.num_his, args.num_pred)
    # shape(num_sample, num_his or num_pred, 2)
    trainTE = torch.cat(trainTE, 1).type(torch.int32)
    # shape(num_sample, num_his + num_pred, 2)

    valTE = seq2instance(val, args.num_his, args.num_pred)
    valTE = torch.cat(valTE, 1).type(torch.int32)

    testTE = seq2instance(test, args.num_his, args.num_pred)
    testTE = torch.cat(testTE, 1).type(torch.int32)

    return trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std


class dataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.len = data_x.shape[0]

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.len









