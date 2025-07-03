import torch.nn as nn
import torch.nn.functional as F
import os

class MaskCNN(nn.Module):
    def __init__(self):
        super(MaskCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 54 * 54, 64)  # 自动推算后是 54x54 而非 53x53
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> [B, 16, 111, 111]
        x = self.pool(F.relu(self.conv2(x)))   # -> [B, 32, 54, 54]
        x = self.flatten(x)                    # -> [B, 32*54*54]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

