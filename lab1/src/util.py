import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def generate_data(N: int) -> Tensor:
    x = torch.linspace(1, 16, N).unsqueeze(1)  # 生成N个在[1, 16]范围内均匀分布的点作为x
    y = torch.log2(x) + torch.cos(torch.tensor(np.pi/2) * x)  # 计算对应的y值
    return x, y


def split_dataset(x: Tensor, y: Tensor, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    total_size = x.size(0)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    # 打乱数据集
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:(train_size + val_size)]
    test_indices = indices[(train_size + val_size):]
    # 返回划分后的数据集
    return x[train_indices, :], y[train_indices, :], x[val_indices, :], y[val_indices, :], \
        x[test_indices, :], y[test_indices, :]


def visualize_set(val_y, pred_y, x_range):
    val_y = val_y.squeeze(1).detach().numpy()
    pred_y = pred_y.squeeze(1).detach().numpy()
    plt.scatter(x_range, val_y, label='Original data')
    plt.scatter(x_range, pred_y, label='Predicted data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Original vs. Predicted')
    plt.legend()
    plt.show()
