import torch
import torch.nn as nn
from utils import sparse_dropout
import torch.nn.functional as F
from modules import GraphConvolution


class GCN(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GraphConvolution(num_node_features, 16)
        self.conv2 = GraphConvolution(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
