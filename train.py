import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from PIL import Image
import os
import random
from tqdm import tqdm
import torch.nn.functional as F

from models.teacher_model import TeacherNet
from models.student_model import StudentNet, denormalize  # Ensure denormalize is implemented in student_model


# Dataset Class
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


# Utilities
def load_best_loss(filepath="best_loss.txt"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            try:
                return float(f.read().strip())
            except:
                return float('inf')
    return float('inf')

def save_best_loss(loss, filepath="best_loss.txt"):
    with open(filepath, "w") as f:
        f.write(str(loss))


# Perceptual Loss with VGG
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.resize = resize

    def forward(self, x, y):
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        return F.l1_loss(self.vgg(x), self.vgg(y))


# Edge loss using Sobel filters
sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

def edge_loss(img1, img2):
    device = img1.device
    sx = sobel_x.repeat(img1.size(1), 1, 1, 1).to(device)
    sy = sobel_y.repeat(img1.size(1), 1, 1, 1).to(device)

    grad1 = torch.sqrt(F.conv2d(img1, sx, padding=1, groups=img1.size(1))**2 +
                       F.conv2d(img1, sy, padding=1, groups=img1.size(1))**2)
    grad2 = torch.sqrt(F.conv2d(img2, sx, padding=1, groups=img2.size(1))**2 +
                       F.conv2d(img2, sy, padding=1, groups=img2.size(1))**2)

    return F.l1_loss(grad1, grad2)


# Final Distillation Loss
vgg_loss = VGGPerceptualLoss().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

def distillation_loss(student_out, teacher_out, ground_truth, alpha=0.3, beta=0.2, gamma=0.2, delta=0.2, eta=0.1):
    l1 = nn.L1Loss()

    student_out = torch.clamp(student_out, 0.0, 1.0)
    teacher_out = torch.clamp(teacher_out, 0.0, 1.0)
    ground_truth = torch.clamp(ground_truth, 0.0, 1.0)

    loss_pixel = l1(student_out, ground_truth)
    loss_distill = l1(student_out, teacher_out)
    loss_perc = vgg_loss(student_out, ground_truth)
    loss_edge = edge_loss(student_out, ground_truth)

    return (alpha * loss_pixel +
            beta * loss_distill +
            gamma * loss_perc +
            delta * loss_edge)


# Training Loop
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📟 Using device: {device}")

    lr_dir = "data/train/inputC"
    hr_dir = "data/train/target"

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        normalize
    ])

    full_dataset = ImageDataset(hr_dir, lr_dir, transform, transform)
    subset_indices = random.sample(range(len(full_dataset)), min(500, len(full_dataset)))
    dataset = Subset(full_dataset, subset_indices)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    teacher = TeacherNet().to(device)
    student = StudentNet().to(device)

    if os.path.exists("student_model_trained.pth"):
        print("⚠️ Found saved model file, trying to load...")
        try:
            student.load_state_dict(torch.load("student_model_trained.pth", map_location=device))
            print("✅ Loaded student model weights.")
        except RuntimeError as e:
            print(f"❌ Failed to load: {e}")

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(student.parameters(), lr=1e-4)

    last_loss = load_best_loss()
    patience = 3
    epochs_no_improve = 0
    max_epochs = 1000
    epoch = 0

    print(f"📉 Loaded best loss: {last_loss:.6f}")

    while True:
        epoch += 1
        student.train()
        running_loss = 0.0

        for i, (lr_imgs, hr_imgs) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher(lr_imgs)

            student_outputs = student(lr_imgs)

            teacher_outputs_dn = denormalize(teacher_outputs)
            student_outputs_dn = denormalize(student_outputs)
            hr_imgs_dn = denormalize(hr_imgs)

            if teacher_outputs_dn.shape != student_outputs_dn.shape:
                teacher_outputs_dn = F.interpolate(teacher_outputs_dn, size=student_outputs_dn.shape[2:], mode='bilinear', align_corners=False)
                hr_imgs_dn = F.interpolate(hr_imgs_dn, size=student_outputs_dn.shape[2:], mode='bilinear', align_corners=False)

            student_outputs_dn = torch.clamp(student_outputs_dn, 0.0, 1.0)
            teacher_outputs_dn = torch.clamp(teacher_outputs_dn, 0.0, 1.0)
            hr_imgs_dn = torch.clamp(hr_imgs_dn, 0.0, 1.0)

            loss = distillation_loss(student_outputs_dn, teacher_outputs_dn, hr_imgs_dn)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / (i + 1)
        print(f"📝 Epoch [{epoch}] Avg Loss: {avg_loss:.6f}")

        if avg_loss < last_loss:
            last_loss = avg_loss
            epochs_no_improve = 0
            torch.save(student.state_dict(), "student_model_trained.pth")
            save_best_loss(avg_loss)
            print(f"💾 Model saved with loss {avg_loss:.6f}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s)")

        if last_loss <= 0.001:
            print("✅ Loss threshold reached. Stopping.")
            break
        if epochs_no_improve >= patience:
            print("⚠️ Early stopping.")
            break
        if epoch >= max_epochs:
            print("🛑 Max epochs reached.")
            break

    print("🎉 Training complete.")


if __name__ == "__main__":
    train()
