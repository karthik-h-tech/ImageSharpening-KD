import os
import shutil
import argparse
import gdown

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='train, test or train-test')
parser.add_argument('--dataset', type=str, default='GoPro', help='all, GoPro, HIDE, RealBlur_R, RealBlur_J')
args = parser.parse_args()

### Google drive file IDs ######
GoPro_train = '1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI'
GoPro_test  = '1k6DTSHu4saUgrGTYkkZXTptILyG9RRll'
HIDE_test   = '1XRomKYJF1H92g1EuD06pCQe4o6HlwB7A'
RealBlurR_test = '1glgeWXCy7Y0qWDc0MXBTUlZYJf8984hS'
RealBlurJ_test = '1Rb1DhhXmX7IXfilQ-zL9aGjQfAAvQTrW'

dataset = args.dataset

for data in args.data.split('-'):
    if data == 'train':
        print('Downloading GoPro Training Data...')
        os.makedirs(os.path.join('data', 'Downloads'), exist_ok=True)
        gdown.download(id=GoPro_train, output='data/Downloads/train.zip', quiet=False)
        print('Extracting GoPro data...')
        shutil.unpack_archive('data/Downloads/train.zip', 'data/Downloads')
        os.rename(os.path.join('data', 'Downloads', 'train'), os.path.join('data', 'Downloads', 'GoPro'))
        os.remove('data/Downloads/train.zip')

    if data == 'test':
        if dataset == 'all' or dataset == 'GoPro':
            print('Downloading GoPro Testing Data...')
            gdown.download(id=GoPro_test, output='data/test.zip', quiet=False)
            print('Extracting GoPro Data...')
            shutil.unpack_archive('data/test.zip', 'data')
            os.remove('data/test.zip')

        if dataset == 'all' or dataset == 'HIDE':
            print('Downloading HIDE Testing Data...')
            gdown.download(id=HIDE_test, output='data/test.zip', quiet=False)
            print('Extracting HIDE Data...')
            shutil.unpack_archive('data/test.zip', 'data')
            os.remove('data/test.zip')

        if dataset == 'all' or dataset == 'RealBlur_R':
            print('Downloading RealBlur_R Testing Data...')
            gdown.download(id=RealBlurR_test, output='data/test.zip', quiet=False)
            print('Extracting RealBlur_R Data...')
            shutil.unpack_archive('data/test.zip', 'data')
            os.remove('data/test.zip')

        if dataset == 'all' or dataset == 'RealBlur_J':
            print('Downloading RealBlur_J Testing Data...')
            gdown.download(id=RealBlurJ_test, output='data/test.zip', quiet=False)
            print('Extracting RealBlur_J Data...')
            shutil.unpack_archive('data/test.zip', 'data')
            os.remove('data/test.zip')
