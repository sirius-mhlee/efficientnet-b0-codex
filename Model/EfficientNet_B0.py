import torch
import torch.nn as nn

import torchvision.models as models

from Model.Layer import CustomLinearLayer

class EfficientNet_B0(nn.Module):
    def __init__(self, class_num):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = CustomLinearLayer(in_features, class_num)

    def forward(self, x):
        return self.backbone(x)
