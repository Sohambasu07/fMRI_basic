""" Transfer Learning using EfficientNet B4 """

import torch.nn as nn
import torchvision.models as models

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNetB4, self).__init__()
        self.PreTrained = models.efficientnet_b4(weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        for param in self.PreTrained.parameters():
            param.requires_grad = False
        
        self.PreTrained.classifier = nn.Linear(960, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.PreTrained(x)
        x = self.sigmoid(x)
        return x