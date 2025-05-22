import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    return ssim(img1, img2, multichannel=True, data_range=img2.max() - img2.min())
