# Dataset Preparation Instructions

This folder should contain the training, validation, and test datasets organized as follows:

- data/
  - train/
    - class1/
    - class2/
    - ...
  - val/
    - class1/
    - class2/
    - ...
  - test/
    - class1/
    - class2/
    - ...

Each class folder should contain images belonging to that class.

## How to add your own images

1. Create subfolders inside train, val, and test directories for each category/class of images.
2. Add your images (preferably high-resolution) into the respective folders.
3. The training script will automatically load images from these folders.

## Sample Dataset

If you want to test the pipeline quickly, you can download a sample dataset such as:

- [BSD500 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
- [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

Download and extract the images into the respective folders.

## Notes

- Images will be downscaled and upscaled during training to simulate video conferencing conditions.
- Ensure images are in formats supported by PIL (e.g., JPG, PNG).
