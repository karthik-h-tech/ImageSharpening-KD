import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from models.student_model import StudentNet
from skimage.metrics import structural_similarity as ssim_func
import numpy as np
from tqdm import tqdm
import cv2  # For resizing images if needed

# 🔹 Paired dataset for blurry → sharp mapping
class PairedImageDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform_lr=None, transform_hr=None):
        self.lr_images = []
        self.hr_images = []
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

        for root, _, files in os.walk(lr_dir):
            for f in files:
                if f.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.lr_images.append(os.path.join(root, f))

        for root, _, files in os.walk(hr_dir):
            for f in files:
                if f.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.hr_images.append(os.path.join(root, f))

        lr_dict = {os.path.basename(path): path for path in self.lr_images}
        hr_dict = {os.path.basename(path): path for path in self.hr_images}
        common_filenames = sorted(set(lr_dict.keys()) & set(hr_dict.keys()))

        self.lr_images = [lr_dict[f] for f in common_filenames]
        self.hr_images = [hr_dict[f] for f in common_filenames]

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_img = Image.open(self.lr_images[idx]).convert('RGB')
        hr_img = Image.open(self.hr_images[idx]).convert('RGB')

        if self.transform_lr:
            lr_img = self.transform_lr(lr_img)
        if self.transform_hr:
            hr_img = self.transform_hr(hr_img)

        return lr_img, hr_img

# 🔹 Remove training prefixes if any
def remove_prefix(state_dict, prefix='model.'):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[len(prefix):] if k.startswith(prefix) else k
        new_state_dict[new_key] = v
    return new_state_dict

# 🔹 Ensure image has minimum size 7x7 for SSIM
def ensure_min_size(img_np, min_size=7):
    h, w = img_np.shape[:2]
    if h < min_size or w < min_size:
        img_np = cv2.resize(img_np, (max(w, min_size), max(h, min_size)), interpolation=cv2.INTER_LINEAR)
    return img_np

# 🔹 Calculate SSIM with safe window size and channel axis
def calculate_ssim(img1, img2):
    min_dim = min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1  # make it odd

    return ssim_func(
        img1,
        img2,
        multichannel=True,
        channel_axis=2,
        win_size=win_size,
        data_range=img2.max() - img2.min()
    )

# 🔹 Main evaluation logic
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr_dir = 'data/train/blurry'   # Low-res / blurry images
    hr_dir = 'data/train/sharp'    # Ground truth sharp images

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = PairedImageDataset(lr_dir, hr_dir, transform_lr=transform, transform_hr=transform)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    student = StudentNet().to(device)
    checkpoint = torch.load('student_model_trained.pth', map_location=device)

    state_dict = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
    state_dict = remove_prefix(state_dict)
    student.load_state_dict(state_dict)
    student.eval()

    ssim_scores = []

    with torch.no_grad():
        for lr_img, hr_img in tqdm(test_loader, desc="Evaluating"):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            output = student(lr_img)

            # Convert tensors to numpy images (H,W,C)
            output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            hr_np = hr_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            # Ensure minimum size
            output_np = ensure_min_size(output_np)
            hr_np = ensure_min_size(hr_np)

            # Resize output_np to match hr_np shape if needed
            if output_np.shape != hr_np.shape:
                output_np = cv2.resize(output_np, (hr_np.shape[1], hr_np.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Calculate SSIM safely
            ssim_val = calculate_ssim(output_np, hr_np)
            ssim_scores.append(ssim_val)

    avg_ssim = np.mean(ssim_scores)
    print(f"\n✅ Average SSIM on test set: {avg_ssim:.4f}")

if __name__ == "__main__":
    evaluate()
