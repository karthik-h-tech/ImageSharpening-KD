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

# --- Config ---
TARGET_SIZE = (128, 128)
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
        return self.to_tensor(lr).clamp(0, 1), self.to_tensor(hr).clamp(0, 1)

# --- Losses ---
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

def edge_loss(img1, img2):
    # Sobel kernels (expanded for 3 channels)
    sobel_x = torch.tensor([[[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]]], dtype=torch.float32)
    sobel_y = torch.tensor([[[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]]], dtype=torch.float32)
    
    # Expand for RGB grouped conv
    sx = sobel_x.expand(3, 1, 3, 3).to(img1.device)
    sy = sobel_y.expand(3, 1, 3, 3).to(img1.device)

    grad1 = torch.sqrt(F.conv2d(img1, sx, padding=1, groups=3) ** 2 + F.conv2d(img1, sy, padding=1, groups=3) ** 2 + 1e-6)
    grad2 = torch.sqrt(F.conv2d(img2, sx, padding=1, groups=3) ** 2 + F.conv2d(img2, sy, padding=1, groups=3) ** 2 + 1e-6)

    return F.l1_loss(grad1, grad2)

def lab_loss(img1, img2):
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    lab1 = kornia.color.rgb_to_lab(img1)
    lab2 = kornia.color.rgb_to_lab(img2)
    return F.l1_loss(lab1[:, 1:], lab2[:, 1:])

# --- Weight Initialization ---
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# --- Training Loop ---
def train():
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    student = StudentNet().to(device)
    student.apply(init_weights)
    teacher = TeacherNet().to(device).eval()

    vgg_loss_fn = VGGPerceptualLoss(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=FP16_ENABLED)

    def run_training(stage_size, stage_epochs, stage_batch):
        nonlocal student, optimizer, scheduler
        global TARGET_SIZE
        TARGET_SIZE = stage_size
        dataset = ImageDataset("data/train/target_patches", "data/train/input_patches")
        dataloader = DataLoader(dataset, batch_size=stage_batch, shuffle=True, num_workers=8, pin_memory=True)

        best_loss, no_improve, patience = float("inf"), 0, 8

        for epoch in range(1, stage_epochs + 1):
            student.train()
            total_loss = 0
            logs = {k: 0.0 for k in ['recon', 'feat', 'vgg', 'ssim', 'edge', 'lab']}
            start = time.time()

            for i, (lr_img, hr_img) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} ({stage_size[0]}x{stage_size[1]})")):
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                with torch.no_grad():
                    teacher_out = torch.clamp(teacher(lr_img), 0, 1)

                with torch.autocast(device_type='cuda', enabled=FP16_ENABLED):
                    student_out = student(lr_img)
                    recon = F.l1_loss(student_out, hr_img)
                    feat = F.l1_loss(student_out, teacher_out)
                    vgg = vgg_loss_fn(student_out, hr_img)
                    ssim_loss = 1 - ssim(student_out, hr_img, data_range=1.0, size_average=True)
                    edge = edge_loss(student_out, hr_img)
                    lab = lab_loss(student_out, hr_img)

                    loss = 0.4 * recon + 0.4 * feat + 0.2 * vgg + 0.2 * ssim_loss + 0.1 * edge + 0.1 * lab

                if torch.isnan(loss): continue
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                for k, v in zip(logs.keys(), [recon, feat, vgg, ssim_loss, edge, lab]):
                    logs[k] += v.item()

                if epoch == 1 and i == 0:
                    save_image(student_out[0], f"{DEBUG_OUTPUT_DIR}/stage_{stage_size[0]}_epoch1_out.png")
                    save_image(hr_img[0], f"{DEBUG_OUTPUT_DIR}/stage_{stage_size[0]}_epoch1_gt.png")
                    save_image(lr_img[0], f"{DEBUG_OUTPUT_DIR}/stage_{stage_size[0]}_epoch1_in.png")

            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            print(f"✅ Epoch {epoch} | Avg Loss: {avg_loss:.6f} | Time: {time.time() - start:.1f}s")
            for k in logs:
                print(f"  {k}: {logs[k]/len(dataloader):.4f}", end=" | ")
            print()

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

    # Stage-wise training
    run_training(stage_size=(128, 128), stage_epochs=12, stage_batch=32)
    run_training(stage_size=(256, 256), stage_epochs=20, stage_batch=8)
    run_training(stage_size=(288, 288), stage_epochs=5, stage_batch=4)

    print("🎉 Training complete. Best model saved as student_model_trained.pth")

if __name__ == "__main__":
    train()
