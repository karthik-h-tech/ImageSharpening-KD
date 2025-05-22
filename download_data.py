import os
import zipfile
import requests
from tqdm import tqdm

# ✅ Working small ZIP file (~6MB of test images)
url = "https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/refs/heads/master.zip"
output_path = "data/sample.zip"
extract_to = "data/train/sharp/"

os.makedirs("data", exist_ok=True)

def download(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    with open(output_path, 'wb') as file, tqdm(
        desc="Downloading Sample Dataset",
        total=total_size,
        unit='iB',
        unit_scale=True,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

# Step 1: Download
print("Starting download...")
download(url, output_path)

# Step 2: Extract
print("Extracting...")
with zipfile.ZipFile(output_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("✅ Done! Images are in:", extract_to)
