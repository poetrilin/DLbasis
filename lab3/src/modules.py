import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
import numpy as np
from typing import Optional, Tuple, Union
from torch_geometric.typing import Adj, SparseTensor


class PairNorm(nn.Module):
    def __init__(self, scale=1, eps=1e-5):
        super(PairNorm, self).__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        return self.scale * x / (norm + self.eps)


class DropEdge:
    def __init__(self, drop_prob):
        self.drop_prob = drop_prob

    def __call__(self, edge_index):
        if self.training and self.drop_prob > 0:
            edge_index = self.drop_edges(edge_index, self.drop_prob)
        return edge_index

    @staticmethod
    def drop_edges(edge_index, drop_prob):
        edge_mask = torch.rand(edge_index.size(1)) > drop_prob
        edge_index = edge_index[:, edge_mask]
        return edge_index


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        bias: bool = True,
    ):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self._cached_adj_t = None

        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        # 将lin的权重初始化为xavier
        init.xavier_uniform_(self.lin.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

    def edge2adj(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        A = torch.zeros(num_nodes, num_nodes)
        for i in range(edge_index.size(1)):
            A[edge_index[0, i], edge_index[1, i]] += 1

        D = A.sum(dim=1)
        D = D + 1
        A = A+torch.eye(num_nodes)
        D_inv_sqrt = torch.sqrt(1.0/D)
        D_inv = D_inv_sqrt.view(-1, 1) * D_inv_sqrt.view(1, -1)
        adj_t = D_inv * A  # element-wise product
        return adj_t

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        根据edge_index计算邻接矩阵,如果在cache中有缓存则直接使用
        """
        if self._cached_adj_t is None:
            self._cached_adj_t = self.edge2adj(edge_index, x.size(0))
        else:
            self._cached_adj_t = self._cached_adj_t.to(x.device)
        adj = self._cached_adj_t
        x = self.lin(x)
        x = torch.matmul(adj, x)

        if self.bias is not None:
            x = x + self.bias
        return x
