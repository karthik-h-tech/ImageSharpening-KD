# train.py

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
from pytorch_msssim import ssim
import kornia
from tqdm import tqdm

from models.teacher_model import TeacherNet
from models.student_model import StudentNet

TARGET_SIZE = (256, 256)  # Increased size for better quality
FP16_ENABLED = torch.cuda.is_available()
DEBUG_OUTPUT_DIR = "debug_output"

# --- Dataset ---
class ImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_images = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(hr_dir)
            for f in files if f.lower().endswith(('.jpg', '.png', '.bmp'))
        ])
        self.lr_images = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(lr_dir)
            for f in files if f.lower().endswith(('.jpg', '.png', '.bmp'))
        ])
        assert len(self.hr_images) == len(self.lr_images), "Mismatch in dataset size."

        self.to_tensor = transforms.Compose([
            transforms.Resize(TARGET_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_images[idx]).convert("RGB")
        lr = Image.open(self.lr_images[idx]).convert("RGB")
        
        hr_tensor = self.to_tensor(hr).clamp(0, 1)
        lr_tensor = self.to_tensor(lr).clamp(0, 1)
        return lr_tensor, hr_tensor

# --- Perceptual Loss ---
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, x, y):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        return F.l1_loss(self.vgg(x), self.vgg(y))

# --- Edge Loss ---
def edge_loss(img1, img2):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(img1.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(img1.device)
    sx = sobel_x.repeat(img1.size(1), 1, 1, 1)
    sy = sobel_y.repeat(img1.size(1), 1, 1, 1)
    grad1 = torch.sqrt(F.conv2d(img1, sx, padding=1, groups=img1.size(1)) ** 2 +
                       F.conv2d(img1, sy, padding=1, groups=img1.size(1)) ** 2 + 1e-6)
    grad2 = torch.sqrt(F.conv2d(img2, sx, padding=1, groups=img2.size(1)) ** 2 +
                       F.conv2d(img2, sy, padding=1, groups=img2.size(1)) ** 2 + 1e-6)
    return F.l1_loss(grad1, grad2)

# --- LAB Loss ---
def lab_loss(img1, img2):
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    lab1 = kornia.color.rgb_to_lab(img1)
    lab2 = kornia.color.rgb_to_lab(img2)
    return F.l1_loss(lab1[:, 1:], lab2[:, 1:])  # Ignore L-channel

# --- Distillation Loss ---
def distillation_loss(student, teacher, target, vgg_loss_fn,
                      alpha=1.0, beta=0.5, gamma=0.1, delta=0.1, epsilon=0.1, zeta=0.05):
    recon = F.l1_loss(student, target)
    feat = F.l1_loss(student, teacher)
    vgg = vgg_loss_fn(student, target)
    edge = edge_loss(student, target)
    ssim_loss = 1 - ssim(student, target, data_range=1.0, size_average=True)
    lab = lab_loss(student, target)
    total = alpha * recon + beta * feat + gamma * vgg + delta * edge + epsilon * ssim_loss + zeta * lab
    return total, {
        'recon': recon.item(), 'feat': feat.item(), 'vgg': vgg.item(),
        'edge': edge.item(), 'ssim': ssim_loss.item(), 'lab': lab.item()
    }

# --- Weight Initialization ---
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# --- Training ---
def train():
    torch.backends.cudnn.benchmark = True
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    dataset = ImageDataset("data/train/target", "data/train/input")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    teacher = TeacherNet().to(device).eval()
    student = StudentNet().to(device)
    student.apply(init_weights)

    vgg_loss_fn = VGGPerceptualLoss(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-4)  # Increased learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=FP16_ENABLED)

    best_loss = float("inf")
    patience, no_improve = 10, 0

    for epoch in range(1, 1001):
        student.train()
        total_loss = 0
        logs = {k: 0.0 for k in ['recon', 'feat', 'vgg', 'edge', 'ssim', 'lab']}
        start = time.time()

        for i, (lr_img, hr_img) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            with torch.no_grad():
                teacher_out = torch.clamp(teacher(lr_img), 0, 1)

            if epoch == 1 and i == 0:
                print(f"[Debug] Input min/max: {lr_img.min():.3f} / {lr_img.max():.3f}")
                print(f"[Debug] HR min/max: {hr_img.min():.3f} / {hr_img.max():.3f}")
                print(f"[Debug] Teacher out min/max: {teacher_out.min():.3f} / {teacher_out.max():.3f}")

            with torch.cuda.amp.autocast(enabled=FP16_ENABLED):
                student_out = student(lr_img)  # Student model already has sigmoid

                loss, parts = distillation_loss(student_out, teacher_out, hr_img, vgg_loss_fn,
                                                alpha=1.0, beta=0.5, gamma=0.1, delta=0.1, epsilon=0.1, zeta=0.05)

            if torch.isnan(loss): continue
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            for k in logs:
                logs[k] += parts[k]

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"✅ Epoch {epoch} | Avg Loss: {avg_loss:.6f} | Time: {time.time() - start:.1f}s")
        for k in logs:
            print(f"  {k}: {logs[k]/len(dataloader):.4f}", end=" | ")
        print()

        # Save output image
        save_image(torch.clamp(student_out[0], 0, 1), f"{DEBUG_OUTPUT_DIR}/epoch_{epoch:03}_out.png")
        save_image(torch.clamp(hr_img[0], 0, 1), f"{DEBUG_OUTPUT_DIR}/epoch_{epoch:03}_gt.png")

        # Save model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model": student.state_dict()}, "student_model_trained.pth")
            print("💾 Model improved and saved.")
            no_improve = 0
        else:
            no_improve += 1
            print(f"📉 No improvement ({no_improve}/{patience})")

        if best_loss <= 0.001 or no_improve >= patience:
            print("⛔ Early stopping.")
            break

    print("🎉 Training complete.")

if __name__ == "__main__":
    train()
