import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from models.teacher_model import TeacherNet
from models.student_net import StudentNet
import numpy as np
from utils.metrics import calculate_ssim
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform_hr=None, transform_lr=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        # Filter only image files with common extensions recursively
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.hr_images = []
        for root, _, files in os.walk(hr_dir):
            for f in files:
                if f.lower().endswith(valid_extensions):
                    self.hr_images.append(os.path.join(root, f))
        self.hr_images = sorted(self.hr_images)

        self.lr_images = []
        for root, _, files in os.walk(lr_dir):
            for f in files:
                if f.lower().endswith(valid_extensions):
                    self.lr_images.append(os.path.join(root, f))
        self.lr_images = sorted(self.lr_images)

        self.transform_hr = transform_hr
        self.transform_lr = transform_lr

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_path = self.hr_images[idx]
        lr_path = self.lr_images[idx]

        hr_image = Image.open(hr_path).convert("RGB")
        lr_image = Image.open(lr_path).convert("RGB")

        if self.transform_hr:
            hr_image = self.transform_hr(hr_image)
        if self.transform_lr:
            lr_image = self.transform_lr(lr_image)

        return lr_image, hr_image

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories for high-res and low-res images
    hr_dir = "data/train/sharp/Sign-Language-Digits-Dataset-master/Dataset"
    # For low-res images, we simulate by downscaling and upscaling on the fly in transform_lr

    transform_hr = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    transform_lr = transforms.Compose([
        transforms.Resize((64, 64)),  # downscale
        transforms.Resize((256, 256)),  # upscale back to original size
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(hr_dir=hr_dir, lr_dir=hr_dir, transform_hr=transform_hr, transform_lr=transform_lr)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    teacher = TeacherNet().to(device)
    student = StudentNet().to(device)

    # Freeze teacher parameters
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    criterion_reconstruction = nn.MSELoss()
    criterion_distillation = nn.MSELoss()

    optimizer = optim.Adam(student.parameters(), lr=1e-4)

    num_epochs = 10

    for epoch in range(num_epochs):
        student.train()
        running_loss = 0.0
        for lr_imgs, hr_imgs in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            optimizer.zero_grad()

            # Teacher output (no grad)
            with torch.no_grad():
                teacher_outputs = teacher(hr_imgs)

            # Student output
            student_outputs = student(lr_imgs)

            # Upsample student output to match hr_imgs size
            student_outputs_upsampled = nn.functional.interpolate(student_outputs, size=hr_imgs.shape[2:], mode='bilinear', align_corners=False)

            # Reconstruction loss (student output vs high-res ground truth)
            loss_reconstruction = criterion_reconstruction(student_outputs_upsampled, hr_imgs)

            # Upsample teacher output to match hr_imgs size
            teacher_outputs_upsampled = nn.functional.interpolate(teacher_outputs, size=hr_imgs.shape[2:], mode='bilinear', align_corners=False)

            # Distillation loss (student output vs teacher output)
            loss_distillation = criterion_distillation(student_outputs_upsampled, teacher_outputs_upsampled)

            # Total loss: weighted sum
            loss = loss_reconstruction + 0.5 * loss_distillation

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

        # Optional: Evaluate SSIM on a small validation set or training batch here

    # Save the trained student model
    torch.save(student.state_dict(), "student_model_trained.pth")
    print("Training complete. Model saved as student_model_trained.pth")

if __name__ == "__main__":
    train()
