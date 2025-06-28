import os
import torch
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
import cv2
from skimage.metrics import structural_similarity as ssim_func
from models.student_model import StudentNet

# 🔧 Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 🔹 Dataset for paired blurry and sharp images
class PairedImageDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform_lr=None, transform_hr=None):
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

        lr_files = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(lr_dir)
            for f in files if f.lower().endswith(('png', 'jpg', 'jpeg'))
        ])
        hr_files = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(hr_dir)
            for f in files if f.lower().endswith(('png', 'jpg', 'jpeg'))
        ])

        lr_dict = {os.path.basename(path): path for path in lr_files}
        hr_dict = {os.path.basename(path): path for path in hr_files}
        common = sorted(set(lr_dict.keys()) & set(hr_dict.keys()))

        self.lr_images = [lr_dict[f] for f in common]
        self.hr_images = [hr_dict[f] for f in common]

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        try:
            lr = Image.open(self.lr_images[idx]).convert('RGB')
            hr = Image.open(self.hr_images[idx]).convert('RGB')

            if self.transform_lr:
                lr = self.transform_lr(lr)
            if self.transform_hr:
                hr = self.transform_hr(hr)

            return lr, hr
        except Exception as e:
            print(f"⚠️ Skipping corrupted image pair: {self.lr_images[idx]} | Error: {e}")
            return self.__getitem__((idx + 1) % len(self.lr_images))

# 🔹 Denormalize
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean

# 🔹 Perceptual Loss with denormalization
class PerceptualLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:9].to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.device = device

    def forward(self, x, y):
        x = denormalize(x)
        y = denormalize(y)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        return F.l1_loss(self.vgg(x), self.vgg(y))

# 🔹 Safe SSIM
def ensure_min_size(img_np, min_size=7):
    h, w = img_np.shape[:2]
    if h < min_size or w < min_size:
        img_np = cv2.resize(img_np, (max(w, min_size), max(h, min_size)), interpolation=cv2.INTER_LINEAR)
    return img_np

def calculate_ssim(img1, img2):
    img1 = ensure_min_size(img1)
    img2 = ensure_min_size(img2)
    min_dim = min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
    return ssim_func(img1, img2, channel_axis=2, win_size=win_size, data_range=1.0)

# 🔹 Load student model
def load_student_model(device, path="student_model_trained.pth"):
    model = StudentNet().to(device)
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    cleaned_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_dict)
    model.eval()
    return model

# 🔹 Main Evaluation
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️ Device:", device)

    lr_dir = "data/inputC_crops"
    hr_dir = "data/target_crops"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # Normalized later in loss
    ])

    dataset = PairedImageDataset(lr_dir, hr_dir, transform, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    student = load_student_model(device)
    perceptual_loss_fn = PerceptualLoss(device)

    ssim_scores = []
    perceptual_losses = []

    with torch.no_grad():
        for lr, hr in tqdm(loader, desc="📊 Evaluating"):
            lr, hr = lr.to(device), hr.to(device)
            out = torch.sigmoid(student(lr))  # ✅ Add sigmoid

            # Denormalize both for perceptual loss
            p_loss = perceptual_loss_fn(out, hr).item()
            perceptual_losses.append(p_loss)

            # Clamp and convert to NumPy for SSIM
            out_np = denormalize(out).squeeze().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            hr_np = denormalize(hr).squeeze().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

            if out_np.shape != hr_np.shape:
                out_np = cv2.resize(out_np, (hr_np.shape[1], hr_np.shape[0]), interpolation=cv2.INTER_LINEAR)

            ssim_score = calculate_ssim(out_np, hr_np)
            ssim_scores.append(ssim_score)

    print("\n✅ Avg SSIM:", np.mean(ssim_scores))
    print("✅ Avg Perceptual Loss:", np.mean(perceptual_losses))

if __name__ == "__main__":
    evaluate()
