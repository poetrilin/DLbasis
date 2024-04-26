import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


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

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / torch.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
