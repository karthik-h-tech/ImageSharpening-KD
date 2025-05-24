import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        resnet = models.resnet18(pretrained=True)

        # Encoder: Use ResNet layers up to layer3 (output: 256 channels)
        self.encoder = nn.Sequential(*list(resnet.children())[:7])  # up to layer3

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder: Upsample back to original size
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Ensure output is in range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)              # downsample via ResNet layers
        x = self.bottleneck(x)           # compress channels
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)  # upsample
        x = self.decoder(x)              # refine features and output 3-channel image
        return x
