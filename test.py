import time
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F

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

    # Post-processing
    output = torch.clamp(output, 0, 1)
    output_cropped = output[:, :, pt:pt + orig_h, pl:pl + orig_w]

    # Save and show
    save_image(output_cropped[0], "student_output_restored.png")
    print("💾 Saved output as 'student_output_restored.png'")
    print(f"⏱️ Inference time: {elapsed_time:.3f} seconds")
    print(f"📊 Output range: [{output_cropped.min():.3f}, {output_cropped.max():.3f}]\n")

    # Display input vs output
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
        "data/test/input/eileen-pan-5d5DSRQ5dUc-unsplash.jpg",
        "student_model_trained.pth"
    )
