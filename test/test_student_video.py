import os
import sys
import time
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from skimage.metrics import structural_similarity as compare_ssim

# === Add project root to sys.path to resolve 'models' import ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now this import will work if 'models/student_model.py' exists
from models.student_model import StudentNet


def pad_image_reflect(img, base=32):
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    h, w = img_tensor.shape[2:]
    pad_h = (base - h % base) % base
    pad_w = (base - w % base) % base
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    padded = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    return padded, (h, w), (pad_top, pad_bottom, pad_left, pad_right)


def load_student_model(weight_path, device):
    model = StudentNet().to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
    model.eval()
    return model


def test_student_model_on_video(video_path, weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    model = load_student_model(weight_path, device)
    print("✅ Model loaded.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs("Output/student_output/output_student", exist_ok=True)
    out_path = "Output/student_output/output_student/output_student_video_side_by_side.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))

    if not out.isOpened():
        print("❌ Failed to initialize output video writer.")
        return

    ms_ssim_scores = []
    ssim_scores = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Prepare input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        padded_tensor, (orig_h, orig_w), (pt, pb, pl, pr) = pad_image_reflect(pil_img)
        padded_tensor = padded_tensor.to(device)

        with torch.no_grad():
            output = model(padded_tensor)
        output = torch.clamp(output, 0, 1)
        output_cropped = output[:, :, pt:pt + orig_h, pl:pl + orig_w]

        # Optional post-processing (sharpen)
        blurred = transforms.functional.gaussian_blur(output_cropped, kernel_size=3, sigma=1)
        sharpened = output_cropped + 0.3 * (output_cropped - blurred)
        output_cropped = torch.clamp(sharpened, 0, 1)

        # Convert tensors to numpy
        output_np = output_cropped[0].permute(1, 2, 0).cpu().numpy()
        output_np_255 = (output_np * 255).astype(np.uint8)

        input_tensor = transforms.ToTensor()(pil_img).unsqueeze(0).to(device)
        input_np = np.array(pil_img)

        # === Compute MS-SSIM ===
        ssim_m = ms_ssim(output_cropped, input_tensor, data_range=1.0, size_average=True).item()
        ms_ssim_scores.append(ssim_m)

        # === Compute SSIM ===
        ssim_s = compare_ssim(input_np, output_np_255, data_range=255, channel_axis=2)
        ssim_scores.append(ssim_s)

        # Create side-by-side video
        out_bgr = cv2.cvtColor(output_np_255, cv2.COLOR_RGB2BGR)
        side_by_side = np.concatenate((frame, out_bgr), axis=1)
        out.write(side_by_side)

        if frame_count % 10 == 0 or frame_count == total_frames:
            print(f"🎞️ Frame {frame_count}/{total_frames} | MS-SSIM: {ssim_m:.4f} | SSIM: {ssim_s:.4f}")

    cap.release()
    out.release()

    avg_ms_ssim = np.mean(ms_ssim_scores) * 100
    avg_ssim = np.mean(ssim_scores) * 100
    print(f"\n📊 Average MS-SSIM: {avg_ms_ssim:.2f}%")
    print(f"📊 Average SSIM:     {avg_ssim:.2f}%")
    print(f"✅ Side-by-side output saved at: {out_path}")


if __name__ == "__main__":
    test_student_model_on_video(
        video_path="Output/student_output/input_student/sample_degraded.mp4",
        weight_path="student_model_trained.pth"
    )
