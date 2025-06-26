import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from PIL import Image
import random
from tqdm import tqdm

from models.teacher_model import TeacherNet
from models.student_model import StudentNet, denormalize


# ✅ Image Dataset
class ImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform_hr=None, transform_lr=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        self.hr_images = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(hr_dir)
            for f in files if f.lower().endswith(valid_extensions)
        ])
        self.lr_images = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(lr_dir)
            for f in files if f.lower().endswith(valid_extensions)
        ])
        assert len(self.hr_images) == len(self.lr_images), "HR and LR datasets must match."
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image = Image.open(self.hr_images[idx]).convert("RGB")
        lr_image = Image.open(self.lr_images[idx]).convert("RGB")
        if self.transform_hr:
            hr_image = self.transform_hr(hr_image)
        if self.transform_lr:
            lr_image = self.transform_lr(lr_image)
        return lr_image, hr_image


# ✅ Load best loss
def load_best_loss(path="best_loss.txt"):
    return float(open(path).read().strip()) if os.path.exists(path) else float('inf')

def save_best_loss(val, path="best_loss.txt"):
    with open(path, "w") as f:
        f.write(str(val))


# ✅ Perceptual Loss using VGG16
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, x, y):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        return F.l1_loss(self.vgg(x), self.vgg(y))


# ✅ Edge Gradient Loss using Sobel
def edge_loss(img1, img2):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    device = img1.device
    sx = sobel_x.repeat(img1.size(1), 1, 1, 1).to(device)
    sy = sobel_y.repeat(img1.size(1), 1, 1, 1).to(device)

    grad1 = torch.sqrt(F.conv2d(img1, sx, padding=1, groups=img1.size(1)) ** 2 +
                       F.conv2d(img1, sy, padding=1, groups=img1.size(1)) ** 2)
    grad2 = torch.sqrt(F.conv2d(img2, sx, padding=1, groups=img2.size(1)) ** 2 +
                       F.conv2d(img2, sy, padding=1, groups=img2.size(1)) ** 2)

    return F.l1_loss(grad1, grad2)


# ✅ Distillation Loss (Composite)
vgg_loss_fn = VGGPerceptualLoss()

def distillation_loss(student, teacher, ground_truth,
                      alpha=0.3, beta=0.2, gamma=0.3, delta=0.2):
    """
    alpha: reconstruction (GT), beta: teacher distillation, gamma: perceptual, delta: edge
    """
    l1 = nn.L1Loss()
    student = torch.clamp(student, 0.0, 1.0)
    teacher = torch.clamp(teacher, 0.0, 1.0)
    ground_truth = torch.clamp(ground_truth, 0.0, 1.0)

    recon_loss = l1(student, ground_truth)
    feat_loss = l1(student, teacher)
    perc_loss = vgg_loss_fn(student, ground_truth)
    edge_grad_loss = edge_loss(student, ground_truth)

    return alpha * recon_loss + beta * feat_loss + gamma * perc_loss + delta * edge_grad_loss


# ✅ Training Loop
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    # Directories
    lr_dir = "data/train/inputC"
    hr_dir = "data/train/target"

    normalize = transforms.Normalize([0.5]*3, [0.5]*3)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        normalize
    ])

    # Dataset and Loader
    dataset = ImageDataset(hr_dir, lr_dir, transform, transform)
    indices = random.sample(range(len(dataset)), min(500, len(dataset)))
    dataloader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=True, num_workers=2)

    # Models
    teacher = TeacherNet().to(device).eval()
    student = StudentNet().to(device)

    # Load pretrained student if available
    if os.path.exists("student_model_trained.pth"):
        student.load_state_dict(torch.load("student_model_trained.pth", map_location=device))
        print("✅ Loaded saved student weights")

    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    best_loss = load_best_loss()
    print(f"📉 Starting from best loss: {best_loss:.6f}")

    patience, no_improve, max_epochs = 3, 0, 1000

    for epoch in range(1, max_epochs + 1):
        student.train()
        running_loss = 0.0

        for lr_img, hr_img in tqdm(dataloader, desc=f"Epoch {epoch}"):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            with torch.no_grad():
                teacher_out = teacher(lr_img)

            student_out = student(lr_img)

            # Denormalize
            t_out = denormalize(teacher_out)
            s_out = denormalize(student_out)
            gt_out = denormalize(hr_img)

            # Match size
            if s_out.shape != t_out.shape:
                t_out = F.interpolate(t_out, size=s_out.shape[2:], mode='bilinear', align_corners=False)
                gt_out = F.interpolate(gt_out, size=s_out.shape[2:], mode='bilinear', align_corners=False)

            loss = distillation_loss(s_out, t_out, gt_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"🔁 Epoch {epoch} - Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), "student_model_trained.pth")
            save_best_loss(best_loss)
            no_improve = 0
            print(f"💾 Saved best model with loss {best_loss:.6f}")
        else:
            no_improve += 1
            print(f"⚠️ No improvement ({no_improve}/{patience})")

        if best_loss <= 0.001:
            print("✅ Loss target reached.")
            break
        if no_improve >= patience:
            print("⛔ Early stopping.")
            break

    print("🎉 Training Complete.")


if __name__ == "__main__":
    train()
