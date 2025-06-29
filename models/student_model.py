import torch
import torch.nn as nn
from torchvision import transforms

# === Depthwise Separable Convolution ===
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)  # ✅ Helps convergence and stability
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.relu(x)

# === Safe Crop for Skip Connections ===
def center_crop(src, target_shape):
    _, _, h, w = src.shape
    target_h, target_w = target_shape
    dh, dw = h - target_h, w - target_w
    return src[:, :, dh // 2:dh // 2 + target_h, dw // 2:dw // 2 + target_w]

# === Student Network ===
class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DepthwiseSeparableConv(3, 4)
        self.enc2 = DepthwiseSeparableConv(4, 8)
        self.enc3 = DepthwiseSeparableConv(8, 16)

        # Bottleneck
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DepthwiseSeparableConv(16, 32)

        # Decoder
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec3 = DepthwiseSeparableConv(32, 16)

        self.up2 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec2 = DepthwiseSeparableConv(16, 8)

        self.up1 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)
        self.dec1 = DepthwiseSeparableConv(8, 4)

        self.final = nn.Conv2d(4, 3, kernel_size=1)
        self.output_act = nn.Sigmoid()  # ✅ Ensures [0, 1] output

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        b = self.bottleneck(p3)

        d3 = self.up3(b)
        e3 = center_crop(e3, d3.shape[2:])
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2 = center_crop(e2, d2.shape[2:])
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1 = center_crop(e1, d1.shape[2:])
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return self.output_act(out)  # Output in [0, 1] range

# === Utility for Normalization (if needed elsewhere) ===
def get_normalize():
    return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def denormalize(tensor):
    return torch.clamp(tensor, 0, 1)
