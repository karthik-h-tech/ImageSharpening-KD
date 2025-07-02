import zipfile
import os

# Path to the existing ZIP file
zip_path = "test (2).zip"

# Extraction directory
extract_to = "data/"

# Ensure the directory exists
os.makedirs(extract_to, exist_ok=True)

# Extract the ZIP file
print("📦 Extracting ZIP...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("✅ Done! Extracted to:", extract_to)
