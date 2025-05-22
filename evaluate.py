import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from models.student_net import StudentNet
from utils.metrics import calculate_ssim
import numpy as np
from tqdm import tqdm

# Custom Dataset for Paired Images with recursive loading and filename matching
class PairedImageDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform_lr=None, transform_hr=None):
        self.lr_images = []
        self.hr_images = []
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

        # Recursively collect low-res images
        for root, _, files in os.walk(lr_dir):
            for f in files:
                if f.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.lr_images.append(os.path.join(root, f))

        # Recursively collect high-res images
        for root, _, files in os.walk(hr_dir):
            for f in files:
                if f.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.hr_images.append(os.path.join(root, f))

        # Create dicts for quick lookup by filename
        lr_dict = {os.path.basename(path): path for path in self.lr_images}
        hr_dict = {os.path.basename(path): path for path in self.hr_images}

        # Find common filenames
        common_filenames = sorted(set(lr_dict.keys()) & set(hr_dict.keys()))

        # Keep only matched pairs
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

# Helper function to clean up state dict keys
def remove_prefix(state_dict, prefix='model.'):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[len(prefix):] if k.startswith(prefix) else k
        new_state_dict[new_key] = v
    return new_state_dict

# Evaluation function
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define paths to test data
    lr_dir = 'data/test/4'   # directory containing low-resolution or blurry test images
    hr_dir = 'data/train/sharp/Sign-Language-Digits-Dataset-master/Dataset/2'    # directory containing corresponding high-resolution sharp images

    # Define transforms
    transform_lr = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    transform_hr = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load dataset and dataloader
    dataset = PairedImageDataset(lr_dir, hr_dir, transform_lr, transform_hr)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load the trained student model
    student = StudentNet().to(device)
    checkpoint = torch.load('student_model_trained.pth', map_location=device)

    # Extract the correct state dict
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
    else:
        state_dict = checkpoint
    state_dict = remove_prefix(state_dict)

    student.load_state_dict(state_dict)
    student.eval()

    # Run evaluation
    ssim_scores = []

    with torch.no_grad():
        for lr_img, hr_img in tqdm(test_loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            output = student(lr_img)

            # Calculate SSIM between output and ground-truth sharp image
            ssim_score = calculate_ssim(output[0], hr_img[0])
            ssim_scores.append(ssim_score)

    avg_ssim = np.mean(ssim_scores)
    print(f"\n✅ Average SSIM on test set: {avg_ssim:.4f}")

# Entry point
if __name__ == "__main__":
    evaluate()
