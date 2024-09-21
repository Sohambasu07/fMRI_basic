""" Transfer Learning using MobileNetv3 """

import torch.nn as nn
import torchvision.models as models

class MobileNetv3(nn.Module):
    def __init__(self, num_classes=1):
        super(MobileNetv3, self).__init__()
        self.PreTrained = models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        for param in self.PreTrained.parameters():
            param.requires_grad = False
        in_features = self.PreTrained.classifier[0].in_features
        self.PreTrained.classifier = nn.Sequential(
            nn.Linear(in_features, 256),  
            nn.ReLU(),                    
            nn.Dropout(p=0.5),           
            nn.Linear(256, num_classes) 
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.PreTrained(x)
        x = self.sigmoid(x)
        return x