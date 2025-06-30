import time
import torch
import numpy as np
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ✅ Import the student model
from models.student_model import StudentNet

def measure_fps(device='cuda', runs=100):
    # Create model and move to device
    model = StudentNet().to(device)
    model.eval()

    # Use input size that matches training (256x256) and is divisible by 32
    # For 1920x1080 testing, we'll use a size that's close to that
    dummy_input = torch.randn(1, 3, 1088, 1920).to(device)  # 1088 = 1080 + 8 to be divisible by 32

    # Warm-up runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Measure inference time
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    end = time.time()

    total_time = end - start
    avg_time_ms = (total_time / runs) * 1000
    fps = 1000 / avg_time_ms

    print(f"✅ Device: {device}")
    print(f"🖼️ Input size: {dummy_input.shape[2]}x{dummy_input.shape[3]}")
    print(f"🕒 Average inference time: {avg_time_ms:.2f} ms")
    print(f"🚀 Estimated FPS: {fps:.2f}")
    
    # Check if FPS meets requirements (30-60 FPS)
    if fps >= 30:
        print(f"✅ FPS requirement met: {fps:.2f} >= 30")
    else:
        print(f"❌ FPS requirement not met: {fps:.2f} < 30")

if __name__ == "__main__":
    # Automatically use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    measure_fps(device)
