import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import GraphConvolution
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GCN(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dims=16):
        super(GCN, self).__init__()
        # self.conv1 = GraphConvolution(num_node_features, hidden_dims)
        # self.conv2 = GraphConvolution(hidden_dims, num_classes)
        self.conv1 = GCNConv(num_node_features, hidden_dims)
        self.conv2 = GCNConv(hidden_dims, num_classes)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
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
