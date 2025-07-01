import torch
import torch.nn as nn
from torchvision import transforms

# Depthwise Separable Convolution Block
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.depthwise(x))
        x = self.relu(self.pointwise(x))
        return x

# StudentNet for Mobile/Real-Time (balanced for 30+ FPS)
class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = DSConv(3, 20)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DSConv(20, 40)
        self.pool2 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = DSConv(40, 80)
        # Decoder
        self.up2 = nn.ConvTranspose2d(80, 40, 2, stride=2)
        self.dec2 = DSConv(40 + 40, 20)
        self.up1 = nn.ConvTranspose2d(20, 20, 2, stride=2)
        self.dec1 = DSConv(20 + 20, 20)
        self.final = nn.Conv2d(20, 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return self.sigmoid(out)

# === Optional Utility ===
def get_normalize():
    return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def denormalize(tensor):
    return torch.clamp(tensor, 0, 1)
