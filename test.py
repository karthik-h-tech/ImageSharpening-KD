import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image

from models.student_model import StudentNet  # Ensure your latest StudentNet is here


# === Denormalization for [-1, 1] => [0, 1] ===
def denormalize(tensor):
    return tensor * 0.5 + 0.5


# === Pad Image to Nearest Multiple of 32 ===
def pad_image(img, base=32):
    w, h = img.size
    pad_w = (base - w % base) % base
    pad_h = (base - h % base) % base
    new_img = Image.new("RGB", (w + pad_w, h + pad_h), (0, 0, 0))
    new_img.paste(img, (0, 0))
    return new_img, w, h


# === Load Student Model ===
def load_student_model(weight_path, device):
    model = StudentNet().to(device)
    checkpoint = torch.load(weight_path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
    else:
        state_dict = checkpoint

    clean_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '') if k.startswith('module.') else k
        clean_dict[new_k] = v

    model.load_state_dict(clean_dict, strict=False)
    model.eval()
    return model


# === Test Logic ===
def test_single_image(image_path, weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    model = load_student_model(weight_path, device)

    image = Image.open(image_path).convert("RGB")
    padded_img, orig_w, orig_h = pad_image(image, base=32)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # for [-1, 1] range
    ])

    input_tensor = transform(padded_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)  # already in [-1, 1]

    # Denormalize both input and output for saving/display
    output = denormalize(output)
    input_cropped = denormalize(input_tensor)

    # Clamp to [0, 1]
    output = torch.clamp(output, 0, 1)
    input_cropped = torch.clamp(input_cropped, 0, 1)

    # Crop to original image size
    output_cropped = output[:, :, :orig_h, :orig_w]
    input_cropped = input_cropped[:, :, :orig_h, :orig_w]

    # Save images
    save_image(output_cropped, "student_restored_output.png")
    save_image(input_cropped, "student_input_visual.png")
    print("✅ Saved output as 'student_restored_output.png'")

    # Visualize
    output_np = output_cropped.squeeze().permute(1, 2, 0).cpu().numpy()
    input_np = input_cropped.squeeze().permute(1, 2, 0).cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(input_np)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(output_np)
    axs[1].set_title("Student Output")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_single_image("0001.png", "student_model_trained.pth")
