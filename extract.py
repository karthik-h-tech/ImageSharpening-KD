from PIL import Image

def preprocess_image(input_path, output_path="safe_input.jpg", max_dim=960):
    # Open and convert to RGB
    img = Image.open(input_path).convert("RGB")

    # Resize while maintaining aspect ratio, if image is too large
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Make dimensions a multiple of 8 (Restormer requirement)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8

        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"🔧 Resized image to {new_w}x{new_h}")

    # Save as clean RGB JPEG
    img.save(output_path, format="JPEG")
    print(f"✅ Image saved as {output_path}")

# Example usage
preprocess_image("new.jpg")
