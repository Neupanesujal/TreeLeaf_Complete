# model.py

import torch
import torch.nn as nn
import torchvision.models as models

class TesseractOCR(nn.Module):
    def __init__(self, num_classes):
        super(TesseractOCR, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  
        
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
        
    def forward(self, x):
        features = self.resnet(x)
        features = features.view(-1, 512, 1, 1)
        features = nn.functional.relu(self.bn1(self.conv1(features)))
        output = self.conv2(features).squeeze(2)
        return output.transpose(1, 2)  
def get_model(num_classes):
    return TesseractOCR(num_classes)