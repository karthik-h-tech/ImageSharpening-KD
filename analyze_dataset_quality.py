import os
import cv2
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt

def compute_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sharpness = np.sqrt(grad_x ** 2 + grad_y ** 2).mean()
    return sharpness

def analyze_dataset(input_dir, target_dir):
    input_files = natsorted(glob(os.path.join(input_dir, "*.png")))
    target_files = natsorted(glob(os.path.join(target_dir, "*.png")))

    sharpness_vals = []
    ssim_vals = []

    print(f"🔍 Evaluating {len(input_files)} image pairs...")
    for in_f, tg_f in tqdm(zip(input_files, target_files), total=len(input_files)):
        img_input = cv2.imread(in_f)
        img_target = cv2.imread(tg_f)

        sharpness = compute_sharpness(img_target)
        sharpness_vals.append(sharpness)

        ssim = compare_ssim(cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY))
        ssim_vals.append(ssim)

    print("\n📊 Dataset Stats:")
    print(f"🔸 Avg Sharpness (target): {np.mean(sharpness_vals):.3f}")
    print(f"🔸 Min-Max Sharpness: {np.min(sharpness_vals):.2f} - {np.max(sharpness_vals):.2f}")
    print(f"🔸 Avg SSIM (input vs target): {np.mean(ssim_vals):.3f}")
    print(f"🔸 Min-Max SSIM: {np.min(ssim_vals):.3f} - {np.max(ssim_vals):.3f}")

    plt.hist(sharpness_vals, bins=30, alpha=0.7, label="Sharpness")
    plt.hist(ssim_vals, bins=30, alpha=0.7, label="SSIM")
    plt.title("📈 Dataset Sharpness & SSIM Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 🔧 Replace with your actual patch dirs
if __name__ == "__main__":
    analyze_dataset("data/train/input_patches", "data/train/target_patches")
