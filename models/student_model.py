import torch
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import matplotlib.pyplot as plt

from student_model import StudentNet  # make sure path is correct

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StudentNet().to(device)
model.load_state_dict(torch.load("student_model.pth", map_location=device))
model.eval()

# Load and preprocess the image
img = Image.open("sample_test1.jpg").convert("RGB")  # use your test image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
input_tensor = transform(img).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    output = model(input_tensor)

# Postprocess the output
output_image = output.squeeze(0).cpu().clamp(0, 1)

# Convert to displayable images
original_image = to_pil_image(input_tensor.squeeze(0).cpu())
enhanced_image = to_pil_image(output_image)

# Display
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(enhanced_image)
plt.title("Enhanced by Student")
plt.axis("off")

plt.show()

# Save result
enhanced_image.save("output_results/fixed_output.jpg")
