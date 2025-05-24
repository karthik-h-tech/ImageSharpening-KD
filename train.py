import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import os
import random
from models.teacher_model import TeacherNet
from models.student_model import StudentNet
from tqdm import tqdm
from utils.metrics import combined_loss
from pytorch_msssim import SSIM  # Added SSIM loss

class ImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform_hr=None, transform_lr=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
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

        assert len(self.hr_images) == len(self.lr_images), "HR and LR datasets must have the same number of images."

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

def load_best_loss(filepath="best_loss.txt"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            try:
                return float(f.read().strip())
            except:
                return float('inf')
    else:
        return float('inf')

def save_best_loss(loss, filepath="best_loss.txt"):
    with open(filepath, "w") as f:
        f.write(str(loss))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_dir = "data/train/sharp"
    lr_dir = "data/train/blurry"

    transform_hr = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    transform_lr = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    full_dataset = ImageDataset(hr_dir=hr_dir, lr_dir=lr_dir, transform_hr=transform_hr, transform_lr=transform_lr)
    
    # 🔁 Subsample dataset (e.g., 1000 images)
    subset_size = min(1000, len(full_dataset))
    subset_indices = random.sample(range(len(full_dataset)), subset_size)
    dataset = Subset(full_dataset, subset_indices)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    teacher = TeacherNet().to(device)
    student = StudentNet().to(device)

    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    criterion_reconstruction = nn.MSELoss()
    criterion_distillation = nn.MSELoss()
    criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=3)

    optimizer = optim.Adam(student.parameters(), lr=1e-4)

    loss_threshold = 0.001
    patience = 5
    best_loss = load_best_loss()
    print(f"Loaded best loss: {best_loss}")

    epochs_no_improve = 0
    max_epochs = 1000
    epoch = 0

    while True:
        epoch += 1
        student.train()
        running_loss = 0.0

        for lr_imgs, hr_imgs in tqdm(dataloader, desc=f"Epoch {epoch}"):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher(hr_imgs)

            student_outputs = student(lr_imgs)

            loss_reconstruction = criterion_reconstruction(student_outputs, hr_imgs)
            loss_distillation = criterion_distillation(student_outputs, teacher_outputs)
            loss_ssim = 1 - criterion_ssim(student_outputs, hr_imgs)

            loss = 0.5 * loss_reconstruction + 0.2 * loss_distillation + 0.3 * loss_ssim
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch}] Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(student.state_dict(), "student_model_trained.pth")
            save_best_loss(best_loss)
            print(f"New best model saved with loss {best_loss:.6f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")

        if best_loss <= loss_threshold:
            print(f"Loss threshold {loss_threshold} reached, stopping training.")
            break
        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs, stopping training.")
            break
        if epoch >= max_epochs:
            print(f"Reached max epochs {max_epochs}, stopping training.")
            break

    print("Training complete.")

if __name__ == "__main__":
    train()
