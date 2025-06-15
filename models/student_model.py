import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms

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

        # Final conv outputs 3 channels (no Sigmoid, output will be denormalized later)
        self.final = nn.Conv2d(16, 3, kernel_size=3, padding=1)

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


def distillation_loss(student_output, teacher_output, criterion_pix=torch.nn.L1Loss(), alpha=0.7):
    """
    Compute distillation loss between student and teacher outputs.

    Args:
        student_output: Output tensor from student network, range [0,1]
        teacher_output: Output tensor from teacher network, range [0,1]
        criterion_pix: Pixel-wise loss function (default: L1Loss)
        alpha: weight for pixel loss (default 0.7), (1-alpha) for other losses if added

    Returns:
        Loss scalar tensor
    """
    # Pixel-wise L1 loss between student and teacher images
    loss_pix = criterion_pix(student_output, teacher_output)

    # You can add other losses here (e.g., perceptual loss, feature losses) for better distillation
    
    return loss_pix


# Normalization/denormalization utilities (match teacher)
def get_normalize():
    return torch.nn.Sequential(
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )

def denormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean


# Example training step function showing distillation usage
def train_step(student_model, teacher_model, optimizer, input_images):
    """
    Perform one training step for the student model distilling from teacher model.

    Args:
        student_model: the StudentNet instance
        teacher_model: pretrained and eval teacher model
        optimizer: optimizer for student model parameters
        input_images: input batch tensor [B, 3, H, W], normalized [0,1]

    Returns:
        loss value
    """
    student_model.train()
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(input_images)  # teacher prediction

    # Forward pass student
    student_outputs = student_model(input_images)

    # Compute distillation loss
    loss = distillation_loss(student_outputs, teacher_outputs)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# In your training loop, ensure you normalize inputs and denormalize outputs for both teacher and student.
# Example usage:
#   input_tensor = normalize(transform(image)).unsqueeze(0).to(device)
#   teacher_out = teacher_model(input_tensor)
#   student_out = student_model(input_tensor)
#   teacher_out = denormalize(teacher_out)
#   student_out = denormalize(student_out)
#   loss = distillation_loss(student_out, teacher_out)
