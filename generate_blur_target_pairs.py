import os
from PIL import Image, ImageFilter
from tqdm import tqdm

# === Configuration ===
source_dir = "data/DIV2K_train_HR"
inputc_dir = "data/inputc"   # blurred images
target_dir = "data/target"   # original sharp images
blur_radius = 5

# === Create output directories if they don't exist ===
os.makedirs(inputc_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)

# === Get all image file paths ===
valid_exts = ('.png', '.jpg', '.jpeg')
image_files = [
    f for f in os.listdir(source_dir)
    if f.lower().endswith(valid_exts)
]

print(f"📦 Found {len(image_files)} images. Applying blur with radius {blur_radius}...")

# === Process images ===
for img_name in tqdm(image_files, desc="Blurring images"):
    img_path = os.path.join(source_dir, img_name)
    img = Image.open(img_path).convert("RGB")

    # Save original to target
    img.save(os.path.join(target_dir, img_name))

    # Apply blur and save to inputc
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blurred.save(os.path.join(inputc_dir, img_name))

print("✅ Done.")
print(f"🔸 Blurred images saved to → {inputc_dir}")
print(f"🔸 Sharp target images saved to → {target_dir}")
