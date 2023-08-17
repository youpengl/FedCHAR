import torch
import torch.nn.functional as F
from torch import nn
batch_size = 10

class HARCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=6, conv_kernel_size=(1, 5), pool_kernel_size=(1, 2), dim=3008):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out
