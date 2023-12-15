""" Transfer Learning using MobileNetv3 """

import torch.nn as nn
import torchvision.models as models

class MobileNetv3(nn.Module):
    def __init__(self, num_classes=1):
        super(MobileNetv3, self).__init__()
        self.PreTrained = models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        for param in self.PreTrained.parameters():
            param.requires_grad = False
        
        self.PreTrained.classifier = nn.Linear(960, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.PreTrained(x)
        x = self.sigmoid(x)
        return x