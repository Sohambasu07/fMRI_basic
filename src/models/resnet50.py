""" Transfer Learning using resnet50 """

import torch.nn as nn
import torchvision.models as models

class Resnet50(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet50, self).__init__()
        self.PreTrained = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)
        for param in self.PreTrained.parameters():
            param.requires_grad = False
        
        self.PreTrained.fc = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.PreTrained(x)
        x = self.sigmoid(x)
        return x