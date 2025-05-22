import torch
import torch.nn as nn
import torchvision.models as models

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        # A simpler model than TeacherNet, for example using fewer layers
        resnet = models.resnet18(pretrained=True)
        # Use fewer layers or modify as needed for student model
        self.features = nn.Sequential(*list(resnet.children())[:-3])  # remove last 3 layers
        self.conv = nn.Conv2d(256, 3, kernel_size=1)  # Map to 3-channel output

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        return x
