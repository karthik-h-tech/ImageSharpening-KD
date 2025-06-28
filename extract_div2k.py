import os
import shutil

# === Configuration ===
zip_filename = 'DIV2K_train_HR.zip'  # Name of the zip file in the main folder
base_dir = 'data'                    # Target base directory

# === Extract Logic ===
def extract_zip_to_data(zip_filename):
    # Full path to the zip file
    zip_path = os.path.join(os.getcwd(), zip_filename)

    # Folder name = zip name without .zip
    folder_name = os.path.splitext(os.path.basename(zip_filename))[0]

    # Final extraction path
    extract_to = os.path.join(base_dir, folder_name)

    # Make sure data/target folder exists
    os.makedirs(extract_to, exist_ok=True)

    print(f"\n📦 Extracting {zip_filename} to {extract_to}...")
    shutil.unpack_archive(zip_path, extract_to)
    print(f"✅ Extraction completed to: {extract_to}")

# === Run ===
if __name__ == "__main__":
    extract_zip_to_data(zip_filename)
