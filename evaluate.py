import os
import torch
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torchvision.utils import make_grid
from models.student_model import StudentNet

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------------------------
# Dataset
# ------------------------------
class PairedImageDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.transform = transform
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
        common_files = sorted(set(os.path.basename(f) for f in lr_files) & set(os.path.basename(f) for f in hr_files))
        lr_map = {os.path.basename(f): f for f in lr_files}
        hr_map = {os.path.basename(f): f for f in hr_files}
        self.lr_images = [lr_map[f] for f in common_files]
        self.hr_images = [hr_map[f] for f in common_files]
        self.filenames = common_files

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr = Image.open(self.lr_images[idx]).convert('RGB')
        hr = Image.open(self.hr_images[idx]).convert('RGB')
        filename = self.filenames[idx]
        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)
        return lr, hr, filename

# ------------------------------
# Utility
# ------------------------------
def denormalize(tensor):
    return tensor * 0.5 + 0.5

def save_image(tensor, path):
    grid = make_grid(tensor, nrow=1, padding=0)
    ndarr = (grid.mul(255).clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy())
    Image.fromarray(ndarr).save(path)

# ------------------------------
# Perceptual Loss
# ------------------------------
class PerceptualLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:9].to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, x, y):
        x = denormalize(x)
        y = denormalize(y)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        return F.l1_loss(self.vgg(x), self.vgg(y))

# ------------------------------
# Load Student Model
# ------------------------------
def load_student_model(device, path="student_model_trained.pth"):
    model = StudentNet().to(device)
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    model.eval()
    return model

# ------------------------------
# Evaluation
# ------------------------------
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️ Using device:", device)

    lr_dir = "data/test/input"
    hr_dir = "data/test/target"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dataset = PairedImageDataset(lr_dir, hr_dir, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = load_student_model(device)
    perceptual = PerceptualLoss(device)

    ms_ssim_scores = []
    p_losses = []

    with torch.no_grad():
        for lr, hr, _ in tqdm(loader, desc="📊 Evaluating"):
            lr, hr = lr.to(device), hr.to(device)
            pred = torch.clamp(model(lr), -1, 1)

            # Perceptual loss
            p_loss = perceptual(pred, hr).item()
            p_losses.append(p_loss)

            # MS-SSIM
            pred_denorm = denormalize(pred).clamp(0, 1)
            hr_denorm = denormalize(hr).clamp(0, 1)
            ms_ssim_val = ms_ssim(pred_denorm, hr_denorm, data_range=1.0, size_average=True).item()
            ms_ssim_scores.append(ms_ssim_val)

    avg_ssim = np.mean(ms_ssim_scores)
    avg_ploss = np.mean(p_losses)

    print("\n✅ Evaluation Complete")
    print("📏 Avg MS-SSIM: {:.4f}".format(avg_ssim))
    print("📉 Avg Perceptual Loss: {:.4f}".format(avg_ploss))

if __name__ == "__main__":
    evaluate()
