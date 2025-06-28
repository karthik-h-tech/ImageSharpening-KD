import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from pytorch_msssim import ssim

from models.teacher_model import TeacherNet
from models.student_model import StudentNet

# === Dataset ===
class ImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.hr_images = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(hr_dir)
            for f in files if f.lower().endswith(valid_exts)
        ])
        self.lr_images = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(lr_dir)
            for f in files if f.lower().endswith(valid_exts)
        ])
        assert len(self.hr_images) == len(self.lr_images), "HR/LR count mismatch"
        self.transform = transform

    def __len__(self): return len(self.hr_images)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_images[idx]).convert("RGB")
        lr = Image.open(self.lr_images[idx]).convert("RGB")
        if self.transform:
            hr = self.transform(hr)
            lr = self.transform(lr)
        return lr, hr

# === Perceptual Loss ===
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].to(device).eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg

    def forward(self, x, y):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        return F.l1_loss(self.vgg(x), self.vgg(y))

# === Edge Loss ===
def edge_loss(img1, img2):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    device = img1.device
    sx = sobel_x.repeat(img1.size(1), 1, 1, 1).to(device)
    sy = sobel_y.repeat(img1.size(1), 1, 1, 1).to(device)
    grad1 = torch.sqrt(F.conv2d(img1, sx, padding=1, groups=img1.size(1))**2 +
                       F.conv2d(img1, sy, padding=1, groups=img1.size(1))**2 + 1e-6)
    grad2 = torch.sqrt(F.conv2d(img2, sx, padding=1, groups=img2.size(1))**2 +
                       F.conv2d(img2, sy, padding=1, groups=img2.size(1))**2 + 1e-6)
    return F.l1_loss(grad1, grad2)

# === Distillation Loss ===
def distillation_loss(student, teacher, target, vgg_loss_fn,
                      alpha=0.25, beta=0.2, gamma=0.25, delta=0.15, epsilon=0.15):
    l1 = nn.L1Loss()
    recon = l1(student, target)
    feat = l1(student, teacher)
    vgg = vgg_loss_fn(student, target)
    edge = edge_loss(student, target)
    ssim_loss = 1 - ssim(student, target, data_range=2.0, size_average=True)  # data_range=2.0 due to Tanh

    total = alpha * recon + beta * feat + gamma * vgg + delta * edge + epsilon * ssim_loss
    return total, {
        'recon': recon.item(),
        'feat': feat.item(),
        'vgg': vgg.item(),
        'edge': edge.item(),
        'ssim': ssim_loss.item()
    }

# === Helpers ===
def load_best_loss(path="best_loss.txt"):
    return float(open(path).read().strip()) if os.path.exists(path) else float('inf')

def save_best_loss(val, path="best_loss.txt"):
    with open(path, "w") as f:
        f.write(str(val))

def load_last_epoch(path="checkpoint.txt"):
    return int(open(path).read().strip()) + 1 if os.path.exists(path) else 1

def save_last_epoch(epoch, path="checkpoint.txt"):
    with open(path, "w") as f:
        f.write(str(epoch))

# === Training ===
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    lr_dir = "data/inputc_crops"
    hr_dir = "data/target_crops"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Tanh [-1, 1]
    ])

    dataset = ImageDataset(hr_dir, lr_dir, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    teacher = TeacherNet().to(device).eval()
    student = StudentNet().to(device)
    vgg_loss_fn = VGGPerceptualLoss(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-5)

    best_loss = load_best_loss()
    start_epoch = load_last_epoch()
    print(f"📉 Resuming from best loss: {best_loss:.6f}, starting at epoch {start_epoch}")

    if os.path.exists("student_model_trained.pth"):
        print("🔁 Loading existing student_model_trained.pth")
        student.load_state_dict(torch.load("student_model_trained.pth"))

    patience, no_improve, max_epochs = 5, 0, 1000

    for epoch in range(start_epoch, max_epochs + 1):
        student.train()
        running_loss, logs, count = 0.0, {'recon': 0, 'feat': 0, 'vgg': 0, 'edge': 0, 'ssim': 0}, 0

        for lr_img, hr_img in tqdm(dataloader, desc=f"Epoch {epoch}"):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            with torch.no_grad():
                teacher_out = torch.tanh(teacher(lr_img))  # match StudentNet range

            student_out = student(lr_img)

            # Align sizes
            if student_out.shape != teacher_out.shape:
                size = student_out.shape[2:]
                teacher_out = F.interpolate(teacher_out, size=size, mode='bilinear', align_corners=False)
                hr_img = F.interpolate(hr_img, size=size, mode='bilinear', align_corners=False)

            loss, parts = distillation_loss(student_out, teacher_out, hr_img, vgg_loss_fn)

            if torch.isnan(loss) or loss.item() < 1e-8:
                print("⚠️ Skipping NaN or very low loss")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            for k in logs: logs[k] += parts[k]
            count += 1

        if count == 0:
            print("⚠️ No valid batches processed. Skipping epoch.")
            continue

        avg_loss = running_loss / count
        print(f"🔁 Epoch {epoch} - Avg Loss: {avg_loss:.6f} | " +
              " | ".join([f"{k}: {logs[k]/count:.4f}" for k in logs]))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), "student_model_trained.pth")
            save_best_loss(best_loss)
            save_last_epoch(epoch)
            no_improve = 0
            print("💾 New best model saved!")
        else:
            no_improve += 1
            print(f"📉 No improvement ({no_improve}/{patience})")

        if best_loss <= 0.001:
            print("✅ Target loss reached. Training complete.")
            break
        if no_improve >= patience:
            print("⛔ Early stopping triggered.")
            break

    print("🎉 Training complete.")

if __name__ == "__main__":
    train()
