import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
import numpy as np
from typing import Optional, Tuple, Union


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
        add_loops: bool = True,
    ):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self._cached_adj_t = None
        self.add_loops = add_loops

        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        # 将lin的权重初始化为xavier
        init.xavier_uniform_(self.lin.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

    def edge2adj(self, edge_index: Tensor, num_nodes: int, add_self_loops: bool = True) -> Tensor:
        """
        get and normalize the adjacency matrix from edge_index whose shape is (2, num_edges)
        if add_self_loops:
            \hat A = A + I, \hat D = D + I, return \hat D^{-1/2} \hat A \hat D^{-1/2}
        else: 
            return I+D^{-1/2} A D^{-1/2}
        default add self loops
        """
        A = torch.zeros(num_nodes, num_nodes)
        for i in range(edge_index.size(1)):
            A[edge_index[0, i], edge_index[1, i]] += 1

        D = A.sum(dim=1)
        if add_self_loops:
            D = D + 1
            A = A+torch.eye(num_nodes)

        D_inv_sqrt = torch.sqrt(1.0/D)
        D_inv = D_inv_sqrt.view(-1, 1) * D_inv_sqrt.view(1, -1)
        adj_t = D_inv * A  # element-wise product

        if add_self_loops:
            return adj_t
        else:
            return torch.eye(num_nodes) + adj_t

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        根据edge_index计算邻接矩阵,如果在cache中有缓存则直接使用
        """
        if self._cached_adj_t is None:
            self._cached_adj_t = self.edge2adj(
                edge_index, x.size(0)).to(x.device)
        else:
            self._cached_adj_t = self._cached_adj_t.to(x.device)
        adj = self._cached_adj_t
        x = self.lin(x)
        x = torch.matmul(adj, x)

        if self.bias is not None:
            x = x + self.bias
        return x
