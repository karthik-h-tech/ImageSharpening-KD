import cv2
import numpy as np
import os
import random
from tqdm import tqdm


def realistic_conferencing_blur(frame):
    """
    Simulate realistic blur like low-bitrate video conferencing:
    1. Downscale + Upscale for resolution loss
    2. Optional motion blur
    3. Light Gaussian noise
    """
    h, w = frame.shape[:2]

    # Step 1: Downscale & Upscale
    scale = random.uniform(0.4, 0.6)  # simulate strong bandwidth downscale
    downscaled = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)

    # Step 2: Motion Blur
    if random.random() < 0.5:
        ksize = random.choice([3, 5])
        angle = random.uniform(0, np.pi)
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        x0, y0 = ksize // 2, ksize // 2
        x1 = int(x0 + (ksize // 2) * np.cos(angle))
        y1 = int(y0 + (ksize // 2) * np.sin(angle))
        cv2.line(kernel, (x0, y0), (x1, y1), 1, 1)
        kernel /= np.sum(kernel)
        blurred = cv2.filter2D(blurred, -1, kernel)

    # Step 3: Add slight Gaussian noise
    noise = np.random.normal(0, 2, frame.shape).astype(np.int16)
    noisy = np.clip(blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return noisy


def degrade_video(input_video_path, output_video_path):
    # === Load video ===
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {input_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # === Initialize writer ===
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # use 'XVID' for .avi if needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"❌ Failed to open video writer: {output_video_path}")
        return

    print(f"📥 Input: {input_video_path}")
    print(f"📤 Output: {output_video_path}")
    print(f"🛠️  Degrading frames...")

    # === Process each frame ===
    for _ in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        degraded_frame = realistic_conferencing_blur(frame)
        out.write(degraded_frame)

    cap.release()
    out.release()
    print(f"\n✅ Degraded video saved successfully.")


if __name__ == "__main__":
    degrade_video(
        input_video_path="student_output/target_student/sample_video.mp4",     # Replace with your high-quality video
        output_video_path="student_output/input_student/sample_degraded.mp4"  # Output degraded video
    )
