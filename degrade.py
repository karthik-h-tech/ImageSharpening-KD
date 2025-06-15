from PIL import Image, ImageFilter
import os

def degrade_image(
    input_path, 
    output_path="degraded_test.jpg", 
    downscale_size=(64, 64),      # Slightly higher resolution
    final_size=(128, 128), 
    apply_blur=True,              # Apply slight blur
    blur_radius=0.6               # Light blur
):
    """
    Slightly degrades an image by downscaling and upscaling.
    Applies a mild blur to simulate real-world low-quality input.
    
    Args:
        input_path (str): Path to the high-resolution input image.
        output_path (str): Where to save the degraded image.
        downscale_size (tuple): Size to downscale to (simulate degradation).
        final_size (tuple): Final upscaled size (model input size).
        apply_blur (bool): Whether to apply Gaussian blur.
        blur_radius (float): Radius for blur if applied.
    """
    # Open original image
    img = Image.open(input_path).convert('RGB')

    # Downscale (simulate mild low-res)
    low_res = img.resize(downscale_size, Image.BICUBIC)

    # Optionally blur
    if apply_blur:
        low_res = low_res.filter(ImageFilter.GaussianBlur(blur_radius))

    # Upscale to model input size
    degraded_img = low_res.resize(final_size, Image.BICUBIC)

    # Save result
    degraded_img.save(output_path)
    print(f"✅ Degraded image saved to: {output_path}")


# Example usage
degrade_image(
    input_path='data/train/sharp/IMG_1199.JPG',
    output_path='degraded_test.jpg',
    downscale_size=(96, 96),       # Less aggressive downscale
    final_size=(128, 128),
    apply_blur=True,
    blur_radius=0.4                # Very slight blur
)

