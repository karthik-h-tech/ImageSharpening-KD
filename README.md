# 🧠 Student-Teacher Framework for Real-Time Video Restoration

This project implements a lightweight **student model** trained using a high-performing **Restormer teacher model** through knowledge distillation. The system is designed to restore degraded video frames caused by low-bandwidth conditions such as motion blur, compression, and resolution loss—common in online classrooms or low-quality calls.

The student model is real-time and optimized for deployment on low-end devices, while the teacher model ensures high restoration fidelity.

---

## 📦 Prerequisites

### ✅ Clone Restormer Repository

To train or test using the teacher model, you **must clone the official Restormer repo**:

```bash
git clone https://github.com/swz30/Restormer.git
```

Make sure the cloned folder structure looks like this:

```
your_project/
├── Restormer/
│   └── Motion_Deblurring/
├── train.py
├── test_student.py
└── ...
```

---

### 📥 Restormer Checkpoint (Required)

To train or evaluate using the teacher model (`Restormer`), you must download the pretrained weights:

🔗 [Download Restormer Deblurring `.pth` file](https://drive.google.com/file/d/1TDzcqvoNJS54yk7RSC-pco__HB4E32pz/view?usp=drive_link)

Place the file here:

```
Restormer/Motion_Deblurring/pretrained_models/
```

---

## 🧪 Working & Workflow

### 1. **Data Preparation**

- **Download Data:**  
  ```bash
  python download_data.py
  ```

- **Extract Data:**  
  ```bash
  python extract_data.py
  ```

- **Generate Patches:**  
  ```bash
  python generate_patches.py
  ```

- **Simulate Blur:**  
  ```bash
  python blur.py
  ```
  or
  ```bash
  python generate_test_input_blur.py
  ```

---

### 2. **Restormer Teacher Model**

- The `Restormer/` directory is sourced from the official [Restormer repository](https://github.com/swz30/Restormer) ([CVPR 2022](https://arxiv.org/abs/2111.09881)).
- It contains the code and pretrained weights for the teacher model, which is state-of-the-art for image restoration tasks such as motion deblurring, deraining, and denoising.
- For more details, see the [Restormer README](https://github.com/swz30/Restormer/blob/main/README.md).

---

### 3. **Model Training**

To train the student model using the Restormer teacher:

```bash
python train.py
```

- Performs multi-stage training with knowledge distillation.
- Automatically resumes if interrupted.

---

### 4. **Testing & Evaluation**

- **Test the Student Model:**
  ```bash
  python test_student.py
  ```

- **Test the Teacher Model:**
  ```bash
  python test_restormer_teacher.py
  ```

- **Evaluate Performance (PSNR, SSIM):**
  ```bash
  python evaluate.py
  ```

- **Benchmark Speed (FPS):**
  ```bash
  python benchmark_model_fps.py
  ```

---

### 5. **Subjective Evaluation (MOS Study)**

Use the `MOS_evaluation/` folder to conduct a Mean Opinion Score (MOS) study. Tools and templates for Excel-based or automated surveys are provided.

---

## 📁 File Descriptions

| File                          | Description                                              |
|------------------------------|----------------------------------------------------------|
| `train.py`                   | Trains student model using teacher supervision           |
| `models/student_model.py`    | Lightweight model for deployment                         |
| `models/teacher_model.py`    | Wrapper for Restormer model                              |
| `blur.py`                    | Simulates blur for training                              |
| `generate_patches.py`        | Prepares image patches for training                      |
| `evaluate.py`                | Computes SSIM and PSNR                                   |
| `benchmark_model_fps.py`     | Benchmarks inference time                                |
| `test_student.py`            | Runs student model for inference                         |
| `test_restormer_teacher.py`  | Runs teacher model (Restormer)                           |
| `optimize_fps.py`            | Optional: optimize student model for speed               |
| `rename.py`                  | Utility script to rename files                           |
| `download_data.py`           | Downloads training/test data                             |
| `extract_data.py`            | Unzips and organizes datasets                            |

---

## 📦 Requirements

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- For Restormer-specific dependencies, see:
  [Restormer/INSTALL.md](https://github.com/swz30/Restormer/blob/main/INSTALL.md)

---

## ✅ Results & Certification

- The student model:
  - Achieves real-time performance on low-end hardware
  - Reaches SSIM > 0.90 on benchmark datasets
- Includes:
  - Objective results (PSNR, SSIM)
  - Subjective evaluation (MOS)
- See the `results/` and `MOS_evaluation/` folders for full metrics and visual comparisons.

---

## 📚 Citation

If you use this framework or the Restormer teacher model, please cite:

```bibtex
@inproceedings{Restormer,
  author    = {Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat and Fahad Shahbaz Khan and Ming-Hsuan Yang and Ling Shao},
  title     = {Restormer: Efficient Transformer for High-Resolution Image Restoration},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022},
  pages     = {5728--5739}
}
```
