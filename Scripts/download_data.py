import os
import gdown

# Google Drive file ID for GoPro training data
GoPro_train = '1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI'

# Ensure the download directory exists
os.makedirs('data/downloads', exist_ok=True)

print('Downloading GoPro training zip...')
gdown.download(id=GoPro_train, output='data/downloads/train.zip', quiet=False)

print('✅ Download complete. Zip saved to data/downloads/train.zip')
