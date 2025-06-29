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
train_zip = 'data/train.zip'
train_extract_tmp = 'data/train_tmp'
train_final_dir = 'data/train'

if os.path.exists(train_zip):
    extract_zip(train_zip, train_extract_tmp)

    # Create final dirs
    os.makedirs(train_final_dir, exist_ok=True)
    os.makedirs(os.path.join(train_final_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(train_final_dir, "target"), exist_ok=True)

    # Move input and target folders
    if os.path.exists(os.path.join(train_extract_tmp, "input")):
        move_contents(os.path.join(train_extract_tmp, "input"), os.path.join(train_final_dir, "input"))
    if os.path.exists(os.path.join(train_extract_tmp, "target")):
        move_contents(os.path.join(train_extract_tmp, "target"), os.path.join(train_final_dir, "target"))

    shutil.rmtree(train_extract_tmp)
    print("📁 Train data organized in data/train/")
else:
    print("⚠️ train.zip not found!")

# -------------------------------
# Extract & prepare test data
# -------------------------------
test_zip = 'data/test.zip'
test_extract_tmp = 'data/test_tmp'
test_final_dir = 'data/test'

if os.path.exists(test_zip):
    extract_zip(test_zip, test_extract_tmp)

    # Create final dirs
    os.makedirs(test_final_dir, exist_ok=True)
    os.makedirs(os.path.join(test_final_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(test_final_dir, "target"), exist_ok=True)

    # Move input and target folders
    if os.path.exists(os.path.join(test_extract_tmp, "input")):
        move_contents(os.path.join(test_extract_tmp, "input"), os.path.join(test_final_dir, "input"))
    if os.path.exists(os.path.join(test_extract_tmp, "target")):
        move_contents(os.path.join(test_extract_tmp, "target"), os.path.join(test_final_dir, "target"))

    shutil.rmtree(test_extract_tmp)
    print("📁 Test data organized in data/test/")
else:
    print("⚠️ test.zip not found!")

print("🚀 All done!")
