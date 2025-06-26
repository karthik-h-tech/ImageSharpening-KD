import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
from collections import OrderedDict

# Add Restormer root to sys.path
restormer_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Restormer'))
if restormer_root not in sys.path:
    sys.path.insert(0, restormer_root)

from basicsr.models.archs.restormer_arch import Restormer

def pad_image(img):
    """Pad image to nearest multiple of 8 (Restormer requirement)."""
    w, h = img.size
    pad_w = (8 - w % 8) % 8
    pad_h = (8 - h % 8) % 8
    new_img = Image.new("RGB", (w + pad_w, h + pad_h), (0, 0, 0))
    new_img.paste(img, (0, 0))
    return new_img, w, h

def denormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean

def load_model(weight_path, device):
    print("📦 Initializing Restormer model...")
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
    )
    print("✅ Model created.")

    print("📂 Loading weights...")
    checkpoint = torch.load(weight_path, map_location=device)
    print("✅ Weights loaded.")

    state_dict = checkpoint.get('params', checkpoint.get('state_dict', checkpoint))

    clean_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace('module.', '') if k.startswith('module.') else k
        clean_dict[new_k] = v

    model.load_state_dict(clean_dict, strict=False)
    print("✅ Weights loaded into model.")

    model.eval().to(device)
    return model

def test_single_image(image_path, weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    if device.type != 'cuda':
        print("⚠️ CUDA (GPU) not available. Using CPU instead.")

    print("🚀 Loading model...")
    model = load_model(weight_path, device)

    print("🖼️ Loading input image...")
    image = Image.open(image_path).convert("RGB")
    padded, orig_w, orig_h = pad_image(image)

    print("📐 Image padded. Shape:", padded.size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    input_tensor = transform(padded).unsqueeze(0).to(device)

    print("🤖 Running inference...")
    with torch.no_grad():
        output = model(input_tensor)
    print("✅ Inference complete.")

    output = denormalize(output)
    output = torch.clamp(output, 0, 1)

    output_cropped = output[:, :, :orig_h, :orig_w]
    input_cropped = denormalize(input_tensor)[:, :, :orig_h, :orig_w]

    save_image(output_cropped, "restored_output.png")
    print("✅ Saved output image: restored_output.png")

    input_np = input_cropped.squeeze().permute(1, 2, 0).cpu().numpy()
    output_np = output_cropped.squeeze().permute(1, 2, 0).cpu().numpy()

    input_np = np.clip(input_np, 0, 1)
    output_np = np.clip(output_np, 0, 1)

    print("📊 Displaying side-by-side comparison...")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(input_np)
    axs[0].set_title("Input")
    axs[0].axis('off')

    axs[1].imshow(output_np)
    axs[1].set_title("Restored")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_single_image(
        "safe_input.jpg",
        "Restormer/Defocus_Deblurring/pretrained_models/single_image_defocus_deblurring.pth"
    )
