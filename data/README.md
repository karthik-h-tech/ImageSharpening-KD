
## Description

- **train/input/**: Contains the original blurred images used for training.
- **train/input_patches/**: Contains small patches cropped from the blurred training images. These are used to speed up and diversify training.
- **train/target/**: Contains the corresponding sharp (ground truth) images for the training set.
- **train/target_patches/**: Contains patches cropped from the sharp training images, aligned with `input_patches`.

- **test/input/**: Contains blurred images for testing and evaluation.
- **test/target/**: Contains the corresponding sharp ground truth images for the test set.

- **test.zip**: A compressed archive of the test set for easy sharing or submission.

## Notes

- All images are in standard formats (`.jpg`, `.png`).
- Patches are typically used for efficient training and to increase dataset diversity.
- Ensure that the number and order of images in `input` and `target` (and their patch folders) match for correct training and evaluation.

## Usage

- Use the `train/` folders for model training.
- Use the `test/` folders for model evaluation and benchmarking.
- If you need to regenerate patches, use the provided `generate_patches.py` script.

---

**Contact:**  
For questions about the dataset or structure, please contact the project maintainer.
