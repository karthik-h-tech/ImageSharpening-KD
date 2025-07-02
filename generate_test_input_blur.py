import cv2
import numpy as np
import os
import random

def degrade_image(img):
    # Apply motion blur
    if random.random() < 0.7:
        k = random.choice([5, 7, 9, 11])
        angle = random.uniform(0, np.pi)
        kernel = np.zeros((k, k))
        x0, y0 = k // 2, k // 2
        x1 = int(x0 + (k // 2) * np.cos(angle))
        y1 = int(y0 + (k // 2) * np.sin(angle))
        cv2.line(kernel, (x0, y0), (x1, y1), 1, 1)
        kernel /= np.sum(kernel)
        img = cv2.filter2D(img, -1, kernel)

    # Add Gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 3, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img

# === Example usage ===
input_dir = 'data/test/target'
output_dir = 'data/test/input'
os.makedirs(output_dir, exist_ok=True)

for fname in sorted(os.listdir(input_dir)):
    if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue

    degraded = degrade_image(img)
    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, degraded)

print("✅ Degraded images saved to:", output_dir)
