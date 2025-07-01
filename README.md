# 🧠 Real-Time Video Frame Restoration via Student-Teacher Knowledge Distillation

## 📝 Abstract

This project proposes a lightweight real-time video frame restoration framework based on **knowledge distillation**. A compact **student model**, capable of running at **30+ FPS** on low-end hardware (e.g., Jetson Nano, Raspberry Pi 4, or mid-range CPUs), is trained using supervision from a high-capacity **Restormer teacher model**.

The framework is designed for use in bandwidth-constrained or computationally limited environments such as **online classrooms**, **telemedicine**, **low-cost surveillance**, and **video conferencing**, where video frames are often degraded by motion blur, compression, or low resolution.

By distilling knowledge from the Restormer model (CVPR 2022), the student model balances **efficiency and visual fidelity**, achieving strong quantitative (PSNR, SSIM) and subjective (MOS) results while maintaining real-time performance.

---

## 📦 Prerequisites

### ✅ Clone Restormer Repository

To train or test with the teacher model, clone the official Restormer repository:

```bash
git clone https://github.com/swz30/Restormer.git
```

Ensure your project folder has the following structure:

```
your_project/
├── Restormer/
│   └── Motion_Deblurring/
├── train.py
├── test_student.py
├── models/
│   ├── student_model.py
│   └── teacher_model.py
└── ...
```

---

### 📥 Download Restormer Checkpoint

You must download the pretrained Restormer Deblurring checkpoint:

🔗 [Download `.pth` file](https://drive.google.com/file/d/1TDzcqvoNJS54yk7RSC-pco__HB4E32pz/view?usp=drive_link)

Then place it under:

```
Restormer/Motion_Deblurring/pretrained_models/
```

---

## 🧪 Workflow

### 1. 📁 Data Preparation

- **Download datasets**:
  ```bash
  python download_data.py
  ```

- **Extract archives**:
  ```bash
  python extract_data.py
  ```

- **Generate training patches**:
  ```bash
  python generate_patches.py
  ```

- **Apply synthetic blur**:
  ```bash
  python generate_test_input_blur.py
  ```

---

### 2. 🧠 Teacher Model (Restormer)

- The teacher model is based on the [Restormer architecture](https://github.com/swz30/Restormer) (CVPR 2022).
- It is pretrained and provides high-quality restoration performance.
- Used only for training the student model or evaluation baseline.

---

### 3. 🏋️ Train the Student Model

Train using knowledge distillation from the teacher:

```bash
python train.py
```

- Supports automatic checkpointing and resume.
- Learns via loss functions that include MSE, perceptual loss, and teacher supervision.

---

### 4. ✅ Testing & Evaluation

- **Test the student model**:
  ```bash
  python test_student.py
  ```

- **Test the teacher (Restormer) model**:
  ```bash
  python test_restormer_teacher.py
  ```

- **Evaluate SSIM, PSNR**:
  ```bash
  python evaluate.py
  ```

- **Benchmark student model speed**:
  ```bash
  python benchmark_model_fps.py
  ```

---

### 5. 👁️ Subjective Evaluation (MOS)

Use the `MOS_evaluation/` folder to:

- Conduct human visual rating (Mean Opinion Score).
- Use included Excel templates or Python-based surveys.

---

## 📁 File Descriptions

| File/Folder                   | Purpose                                                   |
|------------------------------|------------------------------------------------------------|
| `train.py`                   | Trains student model using teacher supervision             |
| `models/student_model.py`    | Lightweight model for real-time inference (30+ FPS)        |
| `models/teacher_model.py`    | Wrapper for pretrained Restormer model                     |
| `blur.py`                    | Simulates realistic motion blur                            |
| `generate_patches.py`        | Splits high-res images into training patches               |
| `evaluate.py`                | Computes SSIM and PSNR metrics                             |
| `benchmark_model_fps.py`     | Measures FPS of student model on target hardware           |
| `test_student.py`            | Runs student model on sample test data                     |
| `test_restormer_teacher.py`  | Runs teacher model for comparison                          |
| `download_data.py`           | Downloads datasets (e.g., DIV2K, Unsplash)                 |
| `extract_data.py`            | Unzips and organizes datasets                              |
| `optimize_fps.py`            | (Optional) Prunes or quantizes model for faster inference  |
| `rename.py`                  | Renames files in bulk (utility)                            |
| `MOS_evaluation/`            | Subjective visual rating templates                         |

---

## ⚙️ Requirements

- Python 3.8+
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- For Restormer-specific dependencies:

  [Restormer/INSTALL.md](https://github.com/swz30/Restormer/blob/main/INSTALL.md)

---

## ✅ Results Summary

| Metric        | Student Model       | Teacher Model (Restormer) |
|---------------|---------------------|----------------------------|
| SSIM          | ≥ 0.90              | 0.94–0.96                  |
| PSNR          | 28–32 dB            | 32–35 dB                   |
| Inference FPS | **30–60+ (Real-Time)** | 3–6 FPS (Non-real-time)     |

- **Visual quality** is nearly indistinguishable from the teacher.
- **MOS results** show strong human preference over raw input.
- **Runs on edge devices** such as Raspberry Pi 4, Jetson Nano, or mid-range laptops.

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

---

## 📬 Contact

For questions, bug reports, or collaborations, please open an issue or contact via the GitHub page.
