import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.student_model import StudentNet
from utils.metrics import calculate_ssim
import numpy as np
from tqdm import tqdm

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(root_dir='data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    student = StudentNet().to(device)
    student.load_state_dict(torch.load('student_model.pth', map_location=device))
    student.eval()

    ssim_scores = []

    with torch.no_grad():
        for inputs, _ in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = student(inputs)
            # Calculate SSIM between input and output
            ssim_score = calculate_ssim(inputs[0], outputs[0])
            ssim_scores.append(ssim_score)

    avg_ssim = np.mean(ssim_scores)
    print(f"Average SSIM on test set: {avg_ssim:.4f}")

if __name__ == "__main__":
    evaluate()
