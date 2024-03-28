import torch
import torch.nn as nn
import torch.nn.functional as F


class my_net(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_depth: int = 1):
        super(my_net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc_mid = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer_depth = hidden_depth
        self.active_func = nn.LeakyReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.active_func(out)

        for _ in range(self.hidden_layer_depth):
            out = self.fc_mid(out)
            out = self.active_func(out)

        out = self.fc2(out)
        out = self.active_func(out)
        return out
