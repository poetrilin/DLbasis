import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import GraphConvolution
from torch_geometric.nn import GCNConv, PairNorm
from torch_geometric.data import Data
import random


def drop_edge(edge_index, keep_ratio: float = 1.):
    num_keep = int(keep_ratio * edge_index.shape[1])
    temp = [True] * num_keep + [False] * (edge_index.shape[1] - num_keep)
    random.shuffle(temp)
    return edge_index[:, temp]


class GCN(nn.Module):
    def __init__(self, num_node_features: int,
                 num_classes: int, *, hidden_dims: int = 16, num_layers: int = 3,
                 dropout: float = 0.1, use_pair_norm: bool = True, dropedge: float = 0):
        super(GCN, self).__init__()
        # self.conv1 = GraphConvolution(num_node_features, hidden_dims)
        self.convs = torch.nn.ModuleList(
            [GraphConvolution(num_node_features, hidden_dims)]
            + [GCNConv(in_channels=hidden_dims, out_channels=hidden_dims)
               for i in range(num_layers - 2)]
        )
        self.conv2 = GraphConvolution(hidden_dims, num_classes)
        self.use_pairnorm = use_pair_norm
        self.pairnorm = PairNorm()
        self.Dropedge = dropedge
        self.activation = nn.ReLU()
        self.drop_edge = dropedge

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        for l in self.convs:
            edges = drop_edge(edge_index, self.drop_edge)
            x = l(x, edges)
            if self.use_pairnorm:
                x = self.pairnorm(x)
            x = self.activation(x)

        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCN_LP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # z所有节点的表示向量
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print(dst.size())   # (7284, 64)
        r = (src * dst).sum(dim=-1)
        # print(r.size())   (7284)
        return r

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
