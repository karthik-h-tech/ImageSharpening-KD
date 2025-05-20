import os
import urllib.request
import zipfile

def download_and_extract(url, extract_to='.'):
    filename = url.split('/')[-1]
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    else:
        print(f"{filename} already downloaded.")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {filename} to {extract_to}")

def prepare_div2k_dataset():
    # URLs for DIV2K dataset (low resolution and high resolution)
    div2k_url = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'

    data_dir = 'data/train'
    os.makedirs(data_dir, exist_ok=True)

    download_and_extract(div2k_url, data_dir)

    print("DIV2K dataset downloaded and extracted to data/train.")

if __name__ == "__main__":
    prepare_div2k_dataset()
