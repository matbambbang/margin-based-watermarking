import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1)
        )
        
        self.layer_sequence = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32, 10)

    def forward(self, inputs):
        out = self.downsample(inputs)
        out = self.layer_sequence(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


class BigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1)
        )
        
        self.layer_sequence = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, 3, 1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, 10)

    def forward(self, inputs):
        out = self.downsample(inputs)
        out = self.layer_sequence(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


class SmallGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.GroupNorm(8,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1)
        )
        
        self.layer_sequence = nn.Sequential(
            nn.GroupNorm(16,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.GroupNorm(16,32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32, 10)

    def forward(self, inputs):
        out = self.downsample(inputs)
        out = self.layer_sequence(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


class BigGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.GroupNorm(8,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1)
        )
        
        self.layer_sequence = nn.Sequential(
            nn.GroupNorm(16,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.GroupNorm(32,64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, 3, 1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, 10)

    def forward(self, inputs):
        out = self.downsample(inputs)
        out = self.layer_sequence(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out



models = {
    'big': BigModel,
    'small': SmallModel,
    'bigg': BigGModel,
    'smallg': SmallGModel
}