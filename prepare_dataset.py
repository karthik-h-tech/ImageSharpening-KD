import os
import shutil
import zipfile

# Temporary extraction directory
download_dir = 'datasets_tmp'
os.makedirs(download_dir, exist_ok=True)

# Final dataset directories
input_dir = 'data/inputc'
target_dir = 'data/target'
os.makedirs(input_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)

# Path to the manually downloaded train.zip
train_zip_path = 'train.zip'

# Ensure the train.zip file exists
if not os.path.exists(train_zip_path):
    raise FileNotFoundError("❌ train.zip not found. Please place it in the project directory.")

# Extract train.zip
print("📦 Extracting train.zip...")
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(download_dir)

# Move blur → inputc and sharp → target
blur_path = os.path.join(download_dir, 'train', 'blur')
sharp_path = os.path.join(download_dir, 'train', 'sharp')

if os.path.exists(blur_path):
    for file in os.listdir(blur_path):
        shutil.move(os.path.join(blur_path, file), os.path.join(input_dir, file))

if os.path.exists(sharp_path):
    for file in os.listdir(sharp_path):
        shutil.move(os.path.join(sharp_path, file), os.path.join(target_dir, file))

# Cleanup temporary extraction folder
shutil.rmtree(download_dir)

print("✅ Dataset ready:")
print("   🔸 Blurry images → data/inputc")
print("   🔸 Sharp images  → data/target")
