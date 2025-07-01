# 🧠 Real-Time Video Frame Restoration via Student-Teacher Knowledge Distillation

## 📝 Abstract

This project proposes a lightweight real-time video frame restoration framework based on **knowledge distillation**. A compact **student model**, capable of running at **30+ FPS** on low-end hardware (e.g., Jetson Nano, Raspberry Pi 4, or mid-range CPUs), is trained using supervision from a high-capacity **Restormer teacher model**.

The framework is designed for use in bandwidth-constrained or computationally limited environments such as **online classrooms**, **telemedicine**, **low-cost surveillance**, and **video conferencing**, where video frames are often degraded by motion blur, compression, or low resolution.

By distilling knowledge from the Restormer model (CVPR 2022), the student model balances **efficiency and visual fidelity**, achieving strong quantitative (PSNR, SSIM) and subjective (MOS) results while maintaining real-time performance.

---

## 📦 Prerequisites

### ✅ Clone Restormer Repository

```bash
git clone https://github.com/swz30/Restormer.git
