import os
import shutil

def extract_zip(zip_path, extract_to):
    print(f"📦 Extracting {zip_path} to {extract_to}")
    try:
        shutil.unpack_archive(zip_path, extract_to)
        print(f"✅ Extraction complete: {extract_to}")
    except shutil.ReadError:
        print(f"❌ Error: {zip_path} is not a valid zip file or is corrupted.")

def move_contents(src_folder, dest_folder):
    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dest_path = os.path.join(dest_folder, item)
        if os.path.isdir(src_path):
            shutil.move(src_path, dest_folder)
        else:
            os.makedirs(dest_folder, exist_ok=True)
            shutil.move(src_path, dest_path)

# -------------------------------
# Extract & prepare train data
# -------------------------------
train_zip = 'data/downloads/train.zip'
train_extract_tmp = 'data/downloads/train_tmp'
train_final_dir = 'data/train'

if os.path.exists(train_zip):
    extract_zip(train_zip, train_extract_tmp)

    # Create final dirs
    os.makedirs(train_final_dir, exist_ok=True)
    os.makedirs(os.path.join(train_final_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(train_final_dir, "target"), exist_ok=True)

    # Move input and target folders
    input_dir = os.path.join(train_extract_tmp, "input")
    target_dir = os.path.join(train_extract_tmp, "target")

    if os.path.exists(input_dir):
        move_contents(input_dir, os.path.join(train_final_dir, "input"))
    else:
        print("⚠️ No 'input' folder found in extracted zip.")

    if os.path.exists(target_dir):
        move_contents(target_dir, os.path.join(train_final_dir, "target"))
    else:
        print("⚠️ No 'target' folder found in extracted zip.")

    shutil.rmtree(train_extract_tmp)
    print("📁 Train data organized in data/train/")
else:
    print("⚠️ train.zip not found!")

print("🚀 Done preparing train data.")
