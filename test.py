import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image

# Import your StudentNet
from models.student_model import StudentNet, denormalize

def pad_image(img):
    """Pad image to nearest multiple of 8."""
    w, h = img.size
    pad_w = (8 - w % 8) % 8
    pad_h = (8 - h % 8) % 8
    new_img = Image.new("RGB", (w + pad_w, h + pad_h), (0, 0, 0))
    new_img.paste(img, (0, 0))
    return new_img, w, h

def load_student_model(weight_path, device):
    model = StudentNet().to(device)
    checkpoint = torch.load(weight_path, map_location=device)

    # Handle possible nested keys in checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    clean_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '') if k.startswith('module.') else k
        clean_dict[new_k] = v

    model.load_state_dict(clean_dict, strict=False)
    model.eval()
    return model

def test_single_image(image_path, weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    model = load_student_model(weight_path, device)

    image = Image.open(image_path).convert("RGB")
    padded_img, orig_w, orig_h = pad_image(image)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    input_tensor = transform(padded_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    output = denormalize(output)
    output = torch.clamp(output, 0, 1)

    # Crop back to original size
    output_cropped = output[:, :, :orig_h, :orig_w]
    input_cropped = denormalize(input_tensor)[:, :, :orig_h, :orig_w]
    input_cropped = torch.clamp(input_cropped, 0, 1)

    # Save output image
    save_image(output_cropped, "student_restored_output.png")
    print("✅ Saved output image: student_restored_output.png")

    # Convert to NumPy
    input_np = input_cropped.squeeze().permute(1, 2, 0).cpu().numpy()
    output_np = output_cropped.squeeze().permute(1, 2, 0).cpu().numpy()

    input_np = np.clip(input_np, 0, 1)
    output_np = np.clip(output_np, 0, 1)

    # Show side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(input_np)
    axs[0].set_title("Input")
    axs[0].axis('off')

    axs[1].imshow(output_np)
    axs[1].set_title("Student Output")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_single_image("degraded_test.JPG", "student_model_trained.pth")
