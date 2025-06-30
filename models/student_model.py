import torch
import torch.nn as nn
from torchvision import transforms

# === Depthwise Separable Convolution ===
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.relu(x)

# === Enhanced Student Network ===
class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder - More channels for better feature extraction
        self.enc1 = DepthwiseSeparableConv(3, 16)
        self.enc2 = DepthwiseSeparableConv(16, 32)
        self.enc3 = DepthwiseSeparableConv(32, 64)

        # Bottleneck
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DepthwiseSeparableConv(64, 128)
        self.bottleneck2 = DepthwiseSeparableConv(128, 128)  # Additional bottleneck layer

        # Decoder
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DepthwiseSeparableConv(128, 64)  # 128 = 64 + 64 (skip connection)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DepthwiseSeparableConv(64, 32)  # 64 = 32 + 32 (skip connection)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DepthwiseSeparableConv(32, 16)  # 32 = 16 + 16 (skip connection)

        # Final output layers
        self.final_conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(16, 3, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Remove sigmoid - let the model learn the proper output range
        # The loss function will handle the clamping

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        # Bottleneck
        b = self.bottleneck(p3)
        b = self.bottleneck2(b)

        # Decoder with skip connections
        d3 = self.up3(b)
        e3 = self._center_crop(e3, d3.shape[2:])
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2 = self._center_crop(e2, d2.shape[2:])
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1 = self._center_crop(e1, d1.shape[2:])
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        # Final output
        out = self.final_conv1(d1)
        out = self.relu(out)
        out = self.final_conv2(out)
        
        # Apply sigmoid to ensure output is in [0, 1] range
        return torch.sigmoid(out)

    def _center_crop(self, src, target_shape):
        _, _, h, w = src.shape
        target_h, target_w = target_shape
        dh, dw = h - target_h, w - target_w
        return src[:, :, dh // 2:dh // 2 + target_h, dw // 2:dw // 2 + target_w]

# === Utility for Normalization (if needed elsewhere) ===
def get_normalize():
    return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def denormalize(tensor):
    return torch.clamp(tensor, 0, 1)
