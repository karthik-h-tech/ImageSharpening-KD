import torch
import torch.nn as nn
from torchvision import transforms

# Depthwise Separable Convolution Block (optional for future use)
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        x = self.relu(self.depthwise(x))
        x = self.relu(self.pointwise(x))
        return x

# Residual UNet Block with LeakyReLU
class ResidualUNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

        # Skip connection adjustment if in_ch != out_ch
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.act1(self.conv1(x))
        out = self.conv2(out)
        return self.act2(out + identity)

# --- Add UNetBlock definition ---
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# Slightly Enhanced StudentNet
class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = UNetBlock(3, 24)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(24, 48)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = UNetBlock(48, 96)
        self.pool3 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = UNetBlock(96, 96)
        # Decoder
        self.up3 = nn.ConvTranspose2d(96, 96, 2, stride=2)
        self.dec3 = UNetBlock(96 + 96, 48)
        self.up2 = nn.ConvTranspose2d(48, 48, 2, stride=2)
        self.dec2 = UNetBlock(48 + 48, 24)
        self.up1 = nn.ConvTranspose2d(24, 24, 2, stride=2)
        self.dec1 = UNetBlock(24 + 24, 24)
        self.final = nn.Conv2d(24, 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
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
