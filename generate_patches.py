import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from natsort import natsorted

### CONFIGURATION ###
patch_size = [512, 512]
stride = [204, 204]
to_remove_ratio = 0.3
max_patch_pairs = 5000
patch_counter = 0

# Paths
train_input_dir = "data/inputC"
train_target_dir = "data/target"
train_input_patch_dir = "data/inputC_crops"
train_target_patch_dir = "data/target_crops"

os.makedirs(train_input_patch_dir, exist_ok=True)
os.makedirs(train_target_patch_dir, exist_ok=True)

######################## FUNCTIONS ########################

def sharpness_measure(img, ksize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.sum(grad_mag)

def filter_by_sharpness(inputs, targets):
    sharpness_scores = [
        np.prod([
            (sharpness_measure(img, k) - np.min([sharpness_measure(im, k) for im in targets])) /
            (np.ptp([sharpness_measure(im, k) for im in targets]) + 1e-8)
            for k in [3, 7, 11, 15]
        ])
        for img in targets
    ]
    threshold_count = int(to_remove_ratio * len(targets))
    keep_indices = np.argsort(sharpness_scores)[threshold_count:]
    return [inputs[i] for i in keep_indices], [targets[i] for i in keep_indices]

def extract_patches(img_lr, img_hr):
    patches_lr, patches_hr = [], []
    H, W = img_lr.shape[:2]
    for i in range(0, H, stride[0]):
        for j in range(0, W, stride[1]):
            if i + patch_size[0] <= H and j + patch_size[1] <= W:
                patches_lr.append(img_lr[i:i+patch_size[0], j:j+patch_size[1]])
                patches_hr.append(img_hr[i:i+patch_size[0], j:j+patch_size[1]])
    return patches_lr, patches_hr

########################### MAIN ###########################

train_input_files = natsorted(glob(os.path.join(train_input_dir, "*.png")))
train_target_files = natsorted(glob(os.path.join(train_target_dir, "*.png")))

train_pairs = list(zip(train_input_files, train_target_files))

print(f"🛠️ Creating up to {max_patch_pairs} training patches from {len(train_pairs)} image pairs...")

for lr_file, hr_file in tqdm(train_pairs):
    if patch_counter >= max_patch_pairs:
        break

    name = os.path.splitext(os.path.basename(lr_file))[0]
    img_lr = cv2.imread(lr_file, -1)
    img_hr = cv2.imread(hr_file, -1)

    lr_patches, hr_patches = extract_patches(img_lr, img_hr)
    lr_patches, hr_patches = filter_by_sharpness(lr_patches, hr_patches)

    for plr, phr in zip(lr_patches, hr_patches):
        patch_counter += 1
        if patch_counter > max_patch_pairs:
            break

        cv2.imwrite(os.path.join(train_input_patch_dir, f"{name}-{patch_counter}.png"), plr)
        cv2.imwrite(os.path.join(train_target_patch_dir, f"{name}-{patch_counter}.png"), phr)

print(f"✅ Done. Total patches saved: {patch_counter}")
print(f"  📁 {train_input_patch_dir}")
print(f"  📁 {train_target_patch_dir}")
