import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()

        # Pretrained MobileNetV2 as encoder
        mobilenet = models.mobilenet_v2(pretrained=True).features

        # Encoder layers
        self.enc1 = mobilenet[0:2]    # Output: 16 channels
        self.enc2 = mobilenet[2:4]    # Output: 24 channels
        self.enc3 = mobilenet[4:7]    # Output: 32 channels
        self.enc4 = mobilenet[7:14]   # Output: 96 channels

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32 + 24, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(24 + 16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final output layer with Tanh to map output to [-1, 1]
        self.final = nn.Conv2d(8, 3, kernel_size=3, padding=1)
        self.output_act = nn.Tanh()

    def crop_to_match(self, src, tgt):
        """Crop src tensor to match spatial size of tgt tensor"""
        h_diff = src.size(2) - tgt.size(2)
        w_diff = src.size(3) - tgt.size(3)
        return src[:, :, h_diff // 2: src.size(2) - (h_diff - h_diff // 2),
                         w_diff // 2: src.size(3) - (w_diff - w_diff // 2)]

    def forward(self, x):
        # Encoder
        x0 = self.enc1(x)
        x1 = self.enc2(x0)
        x2 = self.enc3(x1)
        x3 = self.enc4(x2)

        # Bottleneck
        x = self.bottleneck(x3)

        # Decoder with skip connections
        x2_cropped = self.crop_to_match(x2, x)
        x = self.up3(torch.cat([x, x2_cropped], dim=1))

        x1_cropped = self.crop_to_match(x1, x)
        x = self.up2(torch.cat([x, x1_cropped], dim=1))

        x0_cropped = self.crop_to_match(x0, x)
        x = self.up1(torch.cat([x, x0_cropped], dim=1))

        x = self.up0(x)
        x = self.final(x)
        x = self.output_act(x)  # Tanh scales to [-1, 1]

        return x


# Normalization for input
def get_normalize():
    return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # for [-1, 1]

# Denormalization for output
def denormalize(tensor):
    return (tensor + 1) / 2  # maps [-1, 1] back to [0, 1]
