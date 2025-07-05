import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# -------------------------------
# Setup sys.path for imports
# -------------------------------
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
restormer_dir = os.path.join(root_dir, 'Restormer')
models_dir = os.path.join(root_dir, 'models')

for path in [restormer_dir, models_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# ------------------------
# Import model architectures
# ------------------------
from basicsr.models.archs.restormer_arch import Restormer
from student_model import StudentNet  # Assuming models/student_model.py exists

# ------------------------
# Helper functions
# ------------------------
def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return np.float32(img) / 255.0

def save_img(filepath, img):
    img = np.clip(img, 0, 1)
    img = (img * 255).round().astype(np.uint8)
    Image.fromarray(img).save(filepath)

# ------------------------
# CONFIG
# ------------------------
input_path = 'Output/teacher_output/input/teacher_input.png'
gt_path = 'Output/teacher_output/target/teacher_target.png'
result_dir = 'Output/teacher_output/output/'
weights_path = 'Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth'
use_teacher = True  # ✅ Set to False to test student model
os.makedirs(result_dir, exist_ok=True)

# ------------------------
# LOAD MODEL
# ------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🚀 Using device: {device}')
print(f'🧠 Using {"Teacher" if use_teacher else "Student"} Model')

if use_teacher:
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False
    ).to(device)
else:
    model = StudentNet().to(device)  # Adjust based on your student model's init args

checkpoint = torch.load(weights_path, map_location=device)
model.load_state_dict(checkpoint.get('params', checkpoint), strict=False)
model.eval()

# ------------------------
# PROCESS IMAGE
# ------------------------
print(f"🖼️ Processing image: {input_path}")
img_np = load_img(input_path)
img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

# Pad to multiple of 8
factor = 8
h, w = img_tensor.shape[2], img_tensor.shape[3]
H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
padh = H - h if h % factor != 0 else 0
padw = W - w if w % factor != 0 else 0
img_padded = F.pad(img_tensor, (0, padw, 0, padh), 'reflect')

# Inference
with torch.no_grad():
    restored = model(img_padded)
restored = restored[:, :, :h, :w]
restored_np = torch.clamp(restored, 0, 1).cpu().permute(0, 2, 3, 1).squeeze(0).numpy()

# ------------------------
# SAVE OUTPUT
# ------------------------
filename = os.path.splitext(os.path.basename(input_path))[0]
filename += '_teacher.png' if use_teacher else '_student.png'
save_path = os.path.join(result_dir, filename)
save_img(save_path, restored_np)
print(f"✅ Saved restored image to: {save_path}")

# ------------------------
# COMPUTE METRICS
# ------------------------
if os.path.exists(gt_path):
    gt_np = load_img(gt_path)
    psnr_val = compare_psnr(gt_np, restored_np, data_range=1)
    ssim_val = compare_ssim(gt_np, restored_np, channel_axis=-1, data_range=1)
    print(f"📈 PSNR: {psnr_val:.2f} dB")
    print(f"📈 SSIM: {ssim_val:.4f}")
else:
    print(f"⚠️ Ground truth file not found at: {gt_path}")

# ------------------------
# DISPLAY SIDE BY SIDE
# ------------------------
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_np)
plt.title("Input (Motion Blurred)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(restored_np)
plt.title("Restored Output")
plt.axis('off')
plt.tight_layout()
plt.show()
