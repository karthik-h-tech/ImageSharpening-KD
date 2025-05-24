import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F

class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.conv = nn.Conv2d(512, 3, kernel_size=1)  # Map to 3-channel output

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        # Upsample to input size (assumed 256x256)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x
