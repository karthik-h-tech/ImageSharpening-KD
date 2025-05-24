import os
from PIL import Image
from collections import Counter

sharp_folder = "data/train/sharp"
blurry_folder = "data/train/blurry"

def get_all_shapes(folder):
    shapes = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            try:
                with Image.open(path) as img:
                    shapes.append(img.size)  # size is (width, height)
            except Exception as e:
                print(f"Failed to open {filename}: {e}")
    return shapes

# Get shapes from both folders
sharp_shapes = get_all_shapes(sharp_folder)
blurry_shapes = get_all_shapes(blurry_folder)

# Combine all shapes
all_shapes = sharp_shapes + blurry_shapes

# Find the most common shape
shape_counts = Counter(all_shapes)
most_common_shape, count = shape_counts.most_common(1)[0]

print(f"Most common image shape (width, height): {most_common_shape} with {count} occurrences")

def resize_images_to_shape(folder, target_shape):
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            try:
                with Image.open(path) as img:
                    if img.size != target_shape:
                        resized_img = img.resize(target_shape, Image.Resampling.LANCZOS)
                        resized_img.save(path)
                        print(f"Resized {filename} from {img.size} to {target_shape}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

print("Resizing sharp images to the most common shape...")
resize_images_to_shape(sharp_folder, most_common_shape)

print("Resizing blurry images to the most common shape...")
resize_images_to_shape(blurry_folder, most_common_shape)

print("✅ Done resizing all images to the same shape.")
