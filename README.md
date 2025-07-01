
---

## Working & Workflow

### 1. **Data Preparation**

- **Download Data:**  
  Run:
  ```bash
  python download_data.py
  ```
  This script downloads and organizes public datasets (e.g., DIV2K, Unsplash) into the `data/` folder.

- **Extract Data:**  
  If your data is in compressed or raw format, run:
  ```bash
  python extract_data.py
  ```
  This will extract and organize the images into the correct subfolders.

- **Generate Patches:**  
  For efficient training, generate patches from the full images:
  ```bash
  python generate_patches.py
  ```
  This creates `input_patches/` and `target_patches/` in `data/train/`.

- **Simulate Blur:**  
  To create blurred images for training/testing, use:
  ```bash
  python blur.py
  ```
  or
  ```bash
  python generate_test_input_blur.py
  ```

### 2. **Restormer Teacher Model**

- The `Restormer/` directory is sourced from the official [Restormer repository](https://github.com/swz30/Restormer) ([CVPR 2022](https://arxiv.org/abs/2111.09881)).
- It contains the code and pretrained weights for the teacher model, which is state-of-the-art for image restoration tasks such as motion deblurring, deraining, and denoising.
- For more details, see the [Restormer README](https://github.com/swz30/Restormer/blob/main/README.md).

### 3. **Model Training**

- **Train the Student Model:**  
  ```bash
  python train.py
  ```
  This script performs multi-stage training using knowledge distillation from the teacher (Restormer). It automatically handles checkpointing and resumes if interrupted.

### 4. **Testing & Evaluation**

- **Test the Student Model:**  
  ```bash
  python test_student.py
  ```

- **Test the Teacher Model:**  
  ```bash
  python test_restormer_teacher.py
  ```

- **Evaluate Performance:**  
  ```bash
  python evaluate.py
  ```
  Computes SSIM, PSNR, and other metrics on the test set.

- **Benchmark Speed:**  
  ```bash
  python benchmark_model_fps.py
  ```

### 5. **Subjective Evaluation (MOS Study)**

- Conduct a Mean Opinion Score (MOS) study using the outputs and the provided Excel or script templates in `MOS_evaluation/`.

---

## File Descriptions

- **train.py**: Main training loop for student-teacher distillation.
- **models/student_model.py**: Lightweight student model for real-time inference.
- **models/teacher_model.py**: High-capacity teacher model (Restormer).
- **blur.py**: Simulates realistic video call blur.
- **generate_patches.py**: Splits images into patches for efficient training.
- **evaluate.py**: Computes objective metrics (SSIM, PSNR).
- **benchmark_model_fps.py**: Measures model inference speed.
- **test_student.py**: Runs the student model on sample images.
- **test_restormer_teacher.py**: Runs the teacher model on sample images.
- **optimize_fps.py**: (Optional) Script to further optimize model speed.
- **rename.py**: Utility for renaming files in bulk.
- **download_data.py**: Script to download public datasets.
- **extract_data.py**: Organizes and extracts raw datasets.

---

## Requirements

- Python 3.8+
- PyTorch, torchvision, kornia, pytorch_msssim, tqdm, opencv-python, scikit-image, and other dependencies in `requirements.txt`.
- For Restormer, see [Restormer/INSTALL.md](https://github.com/swz30/Restormer/blob/main/INSTALL.md) for any additional requirements.

---

## Results & Certification

- The student model achieves real-time performance and high SSIM on diverse test images.
- For certification, both objective (SSIM > 0.90) and subjective (MOS) evaluations are provided.
- See the `MOS_evaluation/` folder for subjective study templates and results.

---

## Citation

If you use the Restormer teacher model, please cite:
