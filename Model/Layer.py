import math

import torch
import torch.nn as nn

class CustomLinearLayer(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()

        self.size_in = size_in
        self.size_out = size_out

        self.weights = nn.Parameter(torch.Tensor(size_out, size_in))
        self.bias = nn.Parameter(torch.Tensor(size_out))

        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.matmul(x, self.weights.t())
        x = torch.add(x, self.bias)
        return x
