import torch
import torch.nn as nn


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
    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation=F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()

        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs):
        # print('inputs:', inputs)
        x, support = inputs

        if self.training and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless:  # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight

        out = torch.sparse.mm(support, xw)

        if self.bias is not None:
            out += self.bias

        return self.activation(out), support
