import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):

    def __init__(self, n_channel_in, n_class_out):
        super().__init__()

        self.resnet = models.resnet18(pretrained=False, num_classes=256)

        # Adapted resnet from:
        # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        self.resnet.conv1 = nn.Conv2d(n_channel_in,
                                      64,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=False)
        self.fc = nn.Linear(256, n_class_out)

    def forward(self, x):
        z = self.resnet(x)
        y_pred = self.fc(z)

        return y_pred
