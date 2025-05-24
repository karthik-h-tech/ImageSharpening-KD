import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models.student_model import StudentNet
from torchvision.transforms.functional import to_pil_image

# Load and preprocess input image
img_path = 'degraded_test.jpg'
img = Image.open(img_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

input_tensor = transform(img).unsqueeze(0)  # [1, 3, 256, 256]

# Setup device
device = torch.device("cpu")

# Load trained student model
model = StudentNet().to(device)

# Load checkpoint safely
checkpoint = torch.load('student_model_trained.pth', map_location=device)
if isinstance(checkpoint, dict):
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

# Move input tensor to device
input_tensor = input_tensor.to(device)

# Run inference
with torch.no_grad():
    output_tensor = model(input_tensor)

# Convert output to image for display and saving
output_tensor = output_tensor.squeeze().clamp(0, 1)  # [3, 256, 256]
output_image = output_tensor.permute(1, 2, 0).cpu().numpy()

# Display original vs enhanced
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img.resize((256, 256)))
plt.title("Original (Degraded)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_image)
plt.title("Enhanced by Student")
plt.axis('off')

plt.tight_layout()
plt.show()

# Save enhanced image
output_pil = to_pil_image(output_tensor.cpu())
output_pil.save("enhanced_output.jpg")
print("✅ Enhanced image saved as 'enhanced_output.jpg'")
