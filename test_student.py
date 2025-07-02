import os
import time
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from skimage.exposure import match_histograms
import numpy as np
from pytorch_msssim import ms_ssim  # ✅ MS-SSIM import

from models.student_model import StudentNet

# === Reflect Padding for arbitrary input size ===
def pad_image_reflect(img, base=32):
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    h, w = img_tensor.shape[2:]
    pad_h = (base - h % base) % base
    pad_w = (base - w % base) % base
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    padded = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    return padded, (h, w), (pad_top, pad_bottom, pad_left, pad_right)

# === Load StudentNet with Trained Weights ===
def load_student_model(weight_path, device):
    model = StudentNet().to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
    model.eval()
    return model

# === Test Single Image ===
def test_student_model(blur_path, weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    model = load_student_model(weight_path, device)

    # Load and pad image
    blurred_img = Image.open(blur_path).convert("RGB")
    padded_tensor, (orig_h, orig_w), (pt, pb, pl, pr) = pad_image_reflect(blurred_img)
    padded_tensor = padded_tensor.to(device)

    # Inference
    with torch.no_grad():
        start_time = time.time()
        output = model(padded_tensor)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed_time = time.time() - start_time

    # Clamp and crop
    output = torch.clamp(output, 0, 1)
    output_cropped = output[:, :, pt:pt + orig_h, pl:pl + orig_w]

    # === Post-processing ===
    # [1] Adjust contrast and brightness
    output_cropped = transforms.functional.adjust_contrast(output_cropped, contrast_factor=1.05)
    output_cropped = transforms.functional.adjust_brightness(output_cropped, brightness_factor=1.02)

    # [2] Sharpening
    blurred = transforms.functional.gaussian_blur(output_cropped, kernel_size=3, sigma=1)
    output_cropped = torch.clamp(output_cropped + 0.3 * (output_cropped - blurred), 0, 1)

    # Save output
    output_dir = "student_output/output_student"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(blur_path))
    save_image(output_cropped[0], output_path)
    print(f"💾 Saved output as '{output_path}'")
    print(f"⏱️ Inference time: {elapsed_time:.3f} seconds")
    print(f"📊 Output range: [{output_cropped.min():.3f}, {output_cropped.max():.3f}]")

    # === Compute MS-SSIM ===
    target_path = blur_path.replace("input_student", "target_student")
    if os.path.exists(target_path):
        target_img = Image.open(target_path).convert("RGB")
        target_tensor = transforms.ToTensor()(target_img)[:, :orig_h, :orig_w]

        # Histogram Matching
        out_np = output_cropped[0].permute(1, 2, 0).cpu().numpy()
        tgt_np = target_tensor.permute(1, 2, 0).cpu().numpy()
        matched = match_histograms(out_np, tgt_np, channel_axis=-1)
        output_cropped = torch.tensor(matched).permute(2, 0, 1).unsqueeze(0).to(device).clamp(0, 1)

        # Convert target to batch and match shape
        tgt_tensor = target_tensor.unsqueeze(0).to(device)
        if tgt_tensor.shape[-2:] != output_cropped.shape[-2:]:
            tgt_tensor = F.interpolate(tgt_tensor, size=output_cropped.shape[-2:], mode='bilinear', align_corners=False)

        ms_ssim_val = ms_ssim(output_cropped, tgt_tensor, data_range=1.0, size_average=True).item()
        print(f"📏 MS-SSIM: {ms_ssim_val:.4f}")
    else:
        print(f"⚠️ Target image not found at '{target_path}'")

    # === Display input vs output ===
    input_tensor = transforms.ToTensor()(blurred_img)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(input_tensor.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title("Input (Blurred)")
    axs[0].axis("off")
    axs[1].imshow(output_cropped[0].permute(1, 2, 0).cpu().numpy())
    axs[1].set_title("Student Output")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_student_model(
        "student_output/input_student/img26.jpg",
        "student_model_trained.pth"
    )
