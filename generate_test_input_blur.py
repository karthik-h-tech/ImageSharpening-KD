import cv2
import os
import mediapipe as mp
import numpy as np

# Define paths
target_path = 'data/test/target'
output_path = 'data/test/input'

# Create output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Sort the image filenames to maintain order
image_filenames = sorted([
    f for f in os.listdir(target_path)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

# Setup MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Process each image
for filename in image_filenames:
    img_path = os.path.join(target_path, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Failed to read {filename}")
        continue

    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = segmentation.process(img_rgb)

    # Generate mask from segmentation
    mask = results.segmentation_mask
    condition = mask > 0.5  # Person is foreground

    # Create blurred background
    blurred = cv2.GaussianBlur(img, (55, 55), 0)

    # Composite image: person in focus, background blurred
    output_img = np.where(condition[..., None], img, blurred)

    # Save image to output folder
    output_file = os.path.join(output_path, filename)
    cv2.imwrite(output_file, output_img)

print("Processing complete. Blurred images saved to:", output_path)
