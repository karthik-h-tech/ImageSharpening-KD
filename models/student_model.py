import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        resnet = models.resnet18(pretrained=True)

        # Encoder: ResNet18 initial layers
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 channels, downsampled x2
        self.pool0 = resnet.maxpool                                        # downsampled x4
        self.layer1 = resnet.layer1  # 64 channels, x4
        self.layer2 = resnet.layer2  # 128 channels, x8
        self.layer3 = resnet.layer3  # 256 channels, x16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder with upsampling and skip connections
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # output range [0, 1]
        )
    
    def crop_to_match(self, src, tgt):
        """
        Center-crop src tensor to match the spatial size of tgt tensor.
        src and tgt are 4D tensors [B, C, H, W]
        """
        src_h, src_w = src.size(2), src.size(3)
        tgt_h, tgt_w = tgt.size(2), tgt.size(3)
        diff_h = src_h - tgt_h
        diff_w = src_w - tgt_w
        # Crop src to tgt size
        src = src[:, :, diff_h // 2 : diff_h // 2 + tgt_h, diff_w // 2 : diff_w // 2 + tgt_w]
        return src

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)     # 64 channels, x2 downsample
        x1 = self.pool0(x0)     # downsample x4
        x2 = self.layer1(x1)    # 64 channels, x4
        x3 = self.layer2(x2)    # 128 channels, x8
        x4 = self.layer3(x3)    # 256 channels, x16

        # Bottleneck
        x = self.bottleneck(x4)

        # Decoder with skip connections (crop to match spatial sizes)
        x3_cropped = self.crop_to_match(x3, x)
        x = self.up3(torch.cat([x, x3_cropped], dim=1))

        x2_cropped = self.crop_to_match(x2, x)
        x = self.up2(torch.cat([x, x2_cropped], dim=1))

        x0_cropped = self.crop_to_match(x0, x)
        x = self.up1(torch.cat([x, x0_cropped], dim=1))

        x = self.up0(x)
        x = self.final(x)

        return x
