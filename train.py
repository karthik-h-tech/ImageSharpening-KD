import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.teacher_model import TeacherNet
from models.student_model import StudentNet
from utils.metrics import calculate_ssim
import numpy as np
from tqdm import tqdm
import cv2

# Dataset with downscale-upscale transform to simulate video conferencing conditions
class DownscaleUpscaleDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, downscale_size=(64, 64)):
        self.dataset = datasets.ImageFolder(root_dir=root_dir, transform=transform)
        self.downscale_size = downscale_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Downscale and then upscale image to simulate low-res input
        img_np = np.array(img.permute(1, 2, 0))  # CxHxW to HxWxC
        low_res = cv2.resize(img_np, self.downscale_size, interpolation=cv2.INTER_CUBIC)
        upscaled = cv2.resize(low_res, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_CUBIC)
        upscaled = torch.tensor(upscaled).permute(2, 0, 1).float() / 255.0
        return upscaled, img

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms for high-res images
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = DownscaleUpscaleDataset(root_dir='data/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    teacher = TeacherNet().to(device)
    student = StudentNet().to(device)

    # Freeze teacher parameters
    for param in teacher.parameters():
        param.requires_grad = False

    criterion = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=1e-4)

    epochs = 10
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs_student = student(inputs)
            with torch.no_grad():
                outputs_teacher = teacher(targets)

            loss = criterion(outputs_student, outputs_teacher)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    torch.save(student.state_dict(), 'student_model.pth')

if __name__ == "__main__":
    train()
