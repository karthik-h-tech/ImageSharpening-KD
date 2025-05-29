import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        resnet = models.resnet34(pretrained=True)

        # Encoder
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # output channels=64
        self.encoder2 = resnet.layer1  # output channels=64
        self.encoder3 = resnet.layer2  # output channels=128
        self.encoder4 = resnet.layer3  # output channels=256
        self.encoder5 = resnet.layer4  # output channels=512

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(64, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        def crop_to_match(tensor, target_tensor):
            _, _, h, w = tensor.size()
            _, _, target_h, target_w = target_tensor.size()
            delta_h = h - target_h
            delta_w = w - target_w
            crop_top = delta_h // 2
            crop_left = delta_w // 2
            return tensor[:, :, crop_top:(crop_top + target_h), crop_left:(crop_left + target_w)]

        # Encoder
        e1 = self.encoder1(x)     # 64 channels
        e2 = self.encoder2(e1)    # 64 channels
        e3 = self.encoder3(e2)    # 128 channels
        e4 = self.encoder4(e3)    # 256 channels
        e5 = self.encoder5(e4)    # 512 channels

        # Decoder with skip connections
        d4 = self.up4(e5)  # upsample to e4 size approx
        d4 = self.dec4(torch.cat([d4, crop_to_match(e4, d4)], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, crop_to_match(e3, d3)], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, crop_to_match(e2, d2)], dim=1))

        d1 = self.up1(d2)
        # Resize e1 to match d1 size before concatenation (instead of crop)
        e1_resized = F.interpolate(e1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1_resized], dim=1))

        out = self.final(d1)
        return self.sigmoid(out)  # Output range [0, 1]
