# 🧠 Real-Time Video Frame Restoration via Student-Teacher Knowledge Distillation

## 📝 Abstract

This project proposes a lightweight real-time video frame restoration framework based on **knowledge distillation**. A compact **student model**, capable of running at **30+ FPS** on low-end hardware (e.g., Jetson Nano, Raspberry Pi 4, or mid-range CPUs), is trained using supervision from a high-capacity **Restormer teacher model**.

The framework is designed for use in bandwidth-constrained or computationally limited environments such as **online classrooms**, **telemedicine**, **low-cost surveillance**, and **video conferencing**, where video frames are often degraded by **realistic motion blur, noise**, or mild compression artifacts.

By distilling knowledge from the Restormer model (CVPR 2022), the student model balances **efficiency and visual fidelity**, achieving strong quantitative (PSNR, SSIM) and subjective (MOS) results while maintaining real-time performance.

✅ **The student model can also restore realistically blurred Full HD (1920×1080) video frames — typical of video conferencing — at 30+ FPS**, making it ideal for edge device deployment in real-world streaming applications.

---

## 📦 Prerequisites

### ✅ Clone Restormer Repository

To train or test with the teacher model, clone the official Restormer repository:

```bash
git clone https://github.com/swz30/Restormer.git
```

Ensure your project folder has the following structure:

```
ImageSharpening-KD/
├── data/
├── models/
├── outputs/
│   ├── student_output/
│   └── teacher_output/
├── scripts/
│   ├── download_data.py
│   ├── extract_data.py
│   ├── extract_test.py
│   ├── generate_patches.py
│   ├── generate_test_input.py
│   ├── degrade_video.py
│   └── analyze_dataset_quality.py
├── evaluation/
│   ├── evaluate.py
│   ├── benchmark_model_fps.py
│   └── MOS_evaluation/
├── train/
│   ├── train.py
│   └── student_model_trainer.py
├── test/
│   ├── test_student.py
│   ├── test_student_video.py
│   └── test_restorer_teacher.py
├── requirements.txt
├── README.md
└── Restormer/
    └── Motion_Deblurring/
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
  python Scripts/download_data.py
  ```

- **Extract archives**:
  ```bash
  python Scripts/extract_data.py
  ```

- **Generate training patches**:
  ```bash
  python Scripts/generate_patches.py
  ```

- **Apply synthetic blur**:
  ```bash
  python Scripts/generate_test_input_blur.py
  ```

---

### 2. 🧠 Teacher Model (Restormer)

- The teacher model is based on the [Restormer architecture](https://github.com/swz30/Restormer) (CVPR 2022).
- It is pretrained and provides high-quality restoration performance.
- Used only for training the student model or as a comparison baseline.

#### 🗂️ Output Folder Structure

```bash
outputs/
├── student_output/
│   ├── input/     # Blurred input frames
│   ├── target/    # Ground truth (clean) frames
│   └── output/    # Restored frames from StudentNet
│
├── teacher_output/
│   ├── input/     # Blurred input frames
│   ├── target/    # Ground truth (clean) frames
│   └── output/    # Restored frames from Restormer
```

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

- **Download and extract test set**:  
  Manually download the ZIP from:  
  https://drive.google.com/file/d/1VPZTQGoDkdzldFk7PH1KH_vfIg1KwRq8/view?usp=drive_link  
  Place it in the project root directory and run:  
  ```bash
  python Scripts/extract_test.py
  ```
 
- **Test the student model**:
  ```bash
  python test/test_student.py
  ```

- **Test the teacher (Restormer) model**:
  ```bash
  python test/test_restormer_teacher.py
  ```

- **Evaluate SSIM, PSNR**:
  ```bash
  python evaluate/evaluate.py
  ```

- **Benchmark student model speed**:
  ```bash
  python evaluate/benchmark_model_fps.py
  ```

---

## 5. ✅ Video Testing & Degradation

- **Download the example videos:**  
  Manually download from:  
  [https://drive.google.com/drive/folders/1jWuOOYjfB6ELCkwQsJ9l4uX6LwxesY_6?usp=drive_link](https://drive.google.com/drive/folders/1jWuOOYjfB6ELCkwQsJ9l4uX6LwxesY_6?usp=drive_link)  
  Place them under:
```bash
  Output/student_output/input_student/
  Output/student_output/target_student/
  Output/student_output/output_student/
 ```
- Test the Model on Video
 ```bash
python test/test_student_video.py
 ```
- Output:
 ```bash
 Output/student_output/output_student/output_student_video_side_by_side.mp4**     
```
- **Simulate Realistic Conferencing Blur**
 ```bash
python Scripts/degrade_video.py
 ```

### 6.  👁️ Subjective Evaluation (MOS)

Subjective quality was assessed using the **Mean Opinion Score (MOS)** method.

The evaluation was performed using the `mos_evaluation.xlsx` file located in the `evaluation/MOS_evaluation/` directory. This spreadsheet includes:

- Ratings from **multiple human raters**
- Individual scores for **each test image**
- Automatically computed **average MOS score**

📊 The overall average MOS achieved by the student model was:

```
⭐ Average MOS: 4.13375 / 5.0
```

This indicates **high perceptual quality** of the restored outputs compared to degraded inputs, closely aligning with teacher-level performance as perceived by human observers.


---

## 📁 File Descriptions

| File/Folder                   | Purpose                                                   |
|------------------------------|------------------------------------------------------------|
| `train.py`                   | Trains student model using teacher supervision             |
| `models/student_model.py`    | Lightweight model for real-time inference (30+ FPS)        |
| `models/teacher_model.py`    | Wrapper for pretrained Restormer model                     |
| `Scripts/generate_patches.py`        | Splits high-res images into training patches               |
| `evaluation/evaluate.py`                | Computes SSIM and PSNR metrics                             |
| `evaluation/benchmark_model_fps.py`     | Measures FPS of student model on target hardware           |
| `test/test_student.py`            | Runs student model on sample test data                     |
| `test/test_restormer_teacher.py`  | Runs teacher model and saves input, target, output images  |
| `Scripts/download_data.py`           | Downloads datasets (e.g., DIV2K, Unsplash)                 |
| `Scripts/extract_data.py`            | Unzips and organizes datasets                              |
| `evaluation/MOS_evaluation.xlsx/`            | Contains image-wise ratings from multiple raters and computed average MOS (4.13375)                       |
| `outputs/student_output/`   | Contains input (blurred), target (ground truth), and restored outputs from the student model |
| `output/teacher_output/`            | Contains input, target, and restored outputs from the teacher model (Restormer)      |

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
| MS-SSIM          | ≥ 0.90           | 0.94–0.96                  |
| PSNR          | 28–32 dB            | 32–35 dB                   |
| Inference FPS | **30–60+ (Real-Time)** | 3–6 FPS (Non-real-time)     |


- **Visual quality** is nearly indistinguishable from the teacher.
- **MOS results** show strong human preference over raw input.
- **Runs on edge devices** such as Raspberry Pi 4, Jetson Nano, or mid-range laptops.
- **Teacher output** is organized into `teacher_output/input/`, `target/`, and `output/` for visual and quantitative comparisons.

## 🎥 Video Testing Results

| Metric     | Full HD Video (1920×1080) |
|------------|---------------------------|
| SSIM       | 0.95                       |
| MS-SSIM    | 0.97                       |
| FPS        | 30+ (Real-time)            |

- **Tested on video sequences** using `python test_student_video.py`.
- Achieves high structural similarity on realistic conferencing degradation.

## 📈 Why MS-SSIM?

We use **Multi-Scale Structural Similarity Index (MS-SSIM)** as our primary evaluation metric.

✅ Unlike standard SSIM (which measures similarity at a single scale), MS-SSIM evaluates image quality across multiple resolutions.  
This makes it more aligned with human visual perception, especially for assessing sharpness and structural fidelity.

✅ MS-SSIM is now standard in image restoration research because it better captures how humans perceive improvements in texture and detail.

---

## 📝 Performance Highlights

| Metric     | Single Image | 100-Image Average | Full HD Video |
|------------|--------------|-------------------|---------------|
| MS-SSIM    | 0.95         | 0.90              | 0.97          |
| SSIM       | 0.93         | 0.89              | 0.95          |
| FPS        | 30+          | Real-time         | Real-time     |


## 🧩 Deployment Note for Real-World Scenarios

This framework is optimized for **real-time restoration of lightly degraded video frames**, especially under conditions typical of:

- 🎥 **Video conferencing**
- 👩‍🏫 **Online classrooms**
- 🏥 **Telemedicine**
- 🎦 **Lightweight surveillance**
- 📱 **Mobile and edge streaming**

In these scenarios, degradations are generally **minor**, including:

- Slight motion blur from camera shake
- Light Gaussian noise from compression or bandwidth constraints
- Mild JPEG artifacts


## 🚀 Future Scope
- Integrate temporal consistency to stabilize multi-frame sequences and reduce flicker.
- Apply quantization and pruning for ultra-constrained microcontroller deployment.
- Extend to handle severe motion blur and low-light scenarios.
- Explore multi-modal approaches with depth or inertial sensor data.
