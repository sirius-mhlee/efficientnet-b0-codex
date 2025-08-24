import torch
import torch.nn as nn

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss
