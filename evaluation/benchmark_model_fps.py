import time
import torch
import numpy as np
import os
import sys

# -----------------------------
# Setup sys.path for imports
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))  # Go to project root
models_dir = os.path.join(project_root, 'models')

for path in [project_root, models_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# ✅ Import the student model
from student_model import StudentNet  # models/student_model.py

def measure_fps(device='cuda', runs=100):
    # Create model and move to device
    model = StudentNet().to(device)
    model.eval()

    # Input size: 1088x1920 to be divisible by 32, mimicking 1080p
    dummy_input = torch.randn(1, 3, 1088, 1920).to(device)

    # Warm-up runs (for stable timing)
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

    # Requirement check
    if fps >= 30:
        print(f"✅ FPS requirement met: {fps:.2f} >= 30")
    else:
        print(f"❌ FPS requirement not met: {fps:.2f} < 30")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    measure_fps(device)
